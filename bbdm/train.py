"""Цикл обучения BBDM с EMA, детерминированной валидацией и резюмом."""

import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bbdm.config import (
    CHECKPOINT_DIR,
    EMA_DECAY,
    EMA_START,
    GRAD_CLIP,
    NUM_WORKERS,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SPECTRAL_WEIGHT,
    VAL_SEED,
)


class EMA:
    """Экспоненциальное скользящее среднее весов."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        # Буферы не усредняются, а копируются: они не обучаемы, а среднее
        # по ним (например, счётчиков) бессмысленно.
        for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(buf.data)

    @torch.no_grad()
    def copy_from(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.copy_(param.data)
        for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(buf.data)

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow.state_dict())


@torch.no_grad()
def _validate(bbdm, val_loader, device, spectral_weight, val_seed):
    """Валидационный лосс с фиксированными t и шумом.

    Без фиксации t и eps в q_sample тянутся заново каждую эпоху, и
    val-кривая гуляет на величину, сравнимую с реальным улучшением, --
    тогда и выбор "лучшего" чекпоинта, и ReduceLROnPlateau управляются
    шумом оценки, а не качеством модели.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(val_seed)

    bbdm.eval()
    total, total_mse, total_spec, n = 0.0, 0.0, 0.0, 0
    val_bar = tqdm(val_loader, desc="  Val", leave=False)
    for x0, y in val_bar:
        x0, y = x0.to(device), y.to(device)
        loss, terms = bbdm.loss(
            x0, y,
            spectral_weight=spectral_weight,
            generator=generator,
            return_terms=True,
        )
        total += loss.item()
        total_mse += terms["mse"].item()
        total_spec += terms["spec"].item()
        n += 1
        val_bar.set_postfix(loss=f"{loss.item():.6f}")

    n = max(n, 1)
    return total / n, total_mse / n, total_spec / n


def train(
    bbdm,
    train_dataset,
    val_dataset,
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    ema_start=EMA_START,
    spectral_weight=SPECTRAL_WEIGHT,
    device="cuda",
    checkpoint_dir=CHECKPOINT_DIR,
    num_workers=NUM_WORKERS,
    val_seed=VAL_SEED,
    resume=False,
):
    """Обучает BBDM и возвращает (bbdm, ema).

    Args:
        spectral_weight: вес L1(log RAPSD) в лоссе. 0 -- чистый MSE.
        resume: продолжить с `last.pt` в checkpoint_dir, если он есть.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device != "cpu"))
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Спектральный член лосса усредняет спектр по батчу; хвостовой
        # батч из 1-2 патчей дал бы заметно более шумную оценку с тем же
        # весом, что и полный.
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    bbdm = bbdm.to(device)
    optimizer = torch.optim.Adam(bbdm.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    ema = EMA(bbdm.model, decay=EMA_DECAY)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    last_path = os.path.join(checkpoint_dir, "last.pt")
    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        bbdm.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {start_epoch} (step {global_step})")

    epoch_bar = tqdm(range(start_epoch, n_epochs), desc="Epochs")

    for epoch in epoch_bar:
        bbdm.train()
        train_loss, train_mse, train_spec, n_batches = 0.0, 0.0, 0.0, 0
        train_bar = tqdm(train_loader, desc="  Train", leave=False)

        for x0, y in train_bar:
            x0, y = x0.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = bbdm.loss(
                x0, y, spectral_weight=spectral_weight, return_terms=True
            )
            loss.backward()
            nn.utils.clip_grad_norm_(bbdm.parameters(), GRAD_CLIP)
            optimizer.step()

            # На шаге ema_start в shadow кладутся ЖИВЫЕ веса. Иначе shadow
            # так и остался бы копией инициализации, и ранний "best"
            # чекпоинт сохранил бы в ветке "ema" почти случайные веса.
            if global_step == ema_start:
                ema.copy_from(bbdm.model)
            elif global_step > ema_start:
                ema.update(bbdm.model)

            train_loss += loss.item()
            train_mse += terms["mse"].item()
            train_spec += terms["spec"].item()
            n_batches += 1
            global_step += 1
            train_bar.set_postfix(
                loss=f"{loss.item():.6f}",
                mse=f"{terms['mse'].item():.6f}",
                spec=f"{terms['spec'].item():.4f}",
            )

        n_batches = max(n_batches, 1)
        train_loss /= n_batches
        train_mse /= n_batches
        train_spec /= n_batches

        val_loss, val_mse, val_spec = _validate(
            bbdm, val_loader, device, spectral_weight, val_seed
        )
        scheduler.step(val_loss)

        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}",
            val=f"{val_loss:.6f}",
            val_mse=f"{val_mse:.6f}",
            val_spec=f"{val_spec:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            step=global_step,
        )

        # Пока EMA не стартовала, её ветка чекпоинта должна содержать
        # живые веса, а не инициализацию.
        if global_step < ema_start:
            ema.copy_from(bbdm.model)

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model": bbdm.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_spec": val_spec,
            "train_loss": train_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            # Пишем гиперпараметры процесса рядом с весами: иначе при
            # инференсе легко молча взять другое T/s/eta.
            "hparams": {
                "T": bbdm.T,
                "s": bbdm.s,
                "eta": bbdm.eta,
                "spectral_weight": spectral_weight,
            },
        }
        torch.save(state, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, os.path.join(checkpoint_dir, "best.pt"))
            tqdm.write(
                f"Epoch {epoch+1} saved best "
                f"(val={val_loss:.6f}, mse={val_mse:.6f}, spec={val_spec:.4f})"
            )

    return bbdm, ema
