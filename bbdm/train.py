"""Цикл обучения BBDM с EMA, детерминированной валидацией, резюмом,
спектральным монитором настоящего сэмплера и (опционально) DDP на нескольких GPU.

Многопроцессный режим включается сам, если запущено через `torchrun` и
инициализирован process group (см. `bbdm.dist`); на одном процессе поведение
не меняется."""

import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from bbdm import dist as D
from bbdm.config import (
    CHECKPOINT_DIR,
    EMA_DECAY,
    EMA_START,
    ETA,
    GRAD_CLIP,
    GROUPS,
    MONITOR_EVERY,
    MONITOR_N,
    MONITOR_S,
    NUM_WORKERS,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SPECTRAL_WEIGHT,
    VAL_SEED,
)
from bbdm.evaluate import evaluate_spectra
from bbdm.model import BBDM, UNet


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
def _validate(bbdm, val_loader, device, spectral_weight, val_seed, distributed=False):
    """Валидационный лосс с фиксированными t и шумом.

    Без фиксации t и eps в q_sample тянутся заново каждую эпоху, и
    val-кривая гуляет на величину, сравнимую с реальным улучшением, --
    тогда и выбор "лучшего" чекпоинта, и ReduceLROnPlateau управляются
    шумом оценки, а не качеством модели.

    distributed=True: каждый ранг считает свой шард val, суммы по рангам
    сводятся all_reduce, и ВСЕ ранги получают одно и то же число -- иначе
    ReduceLROnPlateau и выбор best.pt разошлись бы между процессами. В этом
    режиме лосс усредняется по образцам (в одиночном -- по батчам, как раньше).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(val_seed)

    bbdm.eval()
    total, total_mse, total_spec, n = 0.0, 0.0, 0.0, 0
    val_bar = tqdm(val_loader, desc="  Val", leave=False, disable=not D.is_main())
    for x0, y in val_bar:
        x0, y = x0.to(device), y.to(device)
        loss, terms = bbdm.loss(
            x0, y,
            spectral_weight=spectral_weight,
            generator=generator,
            return_terms=True,
        )
        w = x0.shape[0] if distributed else 1
        total += loss.item() * w
        total_mse += terms["mse"].item() * w
        total_spec += terms["spec"].item() * w
        n += w
        val_bar.set_postfix(loss=f"{loss.item():.6f}")

    if distributed:
        t = torch.tensor([total, total_mse, total_spec, float(n)],
                         device=device, dtype=torch.float64)
        D.all_reduce_sum(t)
        total, total_mse, total_spec, n = (float(v) for v in t)

    n = max(n, 1)
    return total / n, total_mse / n, total_spec / n


def _run_monitor(bbdm, ema, dataset, device, n_patches, S, epoch):
    """TF / r_ell / r_in НАСТОЯЩЕГО сэмплера на нескольких val-патчах.

    Смысл -- поймать провал, невидимый одношаговому лоссу. В run 4
    val-лосс вёл себя нормально все 100 эпох, а TF на выходе сэмплера была
    9.5: избыток рождается только в цепочке обратного процесса. Считается
    на EMA-весах (до ema_start там лежат живые), то есть на том, что потом
    уйдёт в инференс. Патчи и сид фиксированы, поэтому числа сравнимы
    между эпохами.
    """
    monitor_bbdm = BBDM(ema.shadow, T=bbdm.T, s=bbdm.s, eta=bbdm.eta).to(device)
    n = min(n_patches, len(dataset))
    res = evaluate_spectra(
        monitor_bbdm, dataset,
        mu=getattr(dataset, "mu", 0.0), sigma=getattr(dataset, "sigma", 1.0),
        indices=list(range(n)), S=S, device=device,
        batch_size=min(4, n), seed=0, progress=False,
    )

    parts = []
    flagged = False
    for (lo, hi), tf, r, r_in in zip(res["bands"], res["tf_bands"], res["r_bands"],
                                      res["r_input_bands"]):
        parts.append(f"{lo:g}-{hi:g}: TF {tf:.2f} r {r:.2f} r_in {r_in:.2f}")
        # Флаг только на ИЗБЫТОК: это режим run 4, невидимый одношаговому
        # лоссу. Недобор (TF < 1) в первых эпохах нормален -- недообученный
        # MSE-денойзер гладкий -- и виден в самой строке; предупреждение на
        # него кричало бы каждую раннюю эпоху, и его перестали бы читать.
        if tf == tf and tf > 2.0:  # tf == tf отсекает nan
            flagged = True
    line = f"  [monitor ep {epoch + 1}, S={S}, n={n}] " + " | ".join(parts)
    if flagged:
        line += ("\n  [monitor] TF > 2 -- избыток мощности на выходе сэмплера, "
                 "как в run 4. Посмотрите до того, как тратить на прогон ещё часы.")
    tqdm.write(line)

    return {
        "epoch": epoch,
        "S": S,
        "n_patches": n,
        "bands": res["bands"],
        "tf_bands": res["tf_bands"],
        "r_bands": res["r_bands"],
        "r_input_bands": res["r_input_bands"],
    }


def _hparams(bbdm, spectral_weight):
    """Гиперпараметры процесса и архитектуры -- пишутся рядом с весами."""
    net = bbdm.model
    return {
        "T": bbdm.T,
        "s": bbdm.s,
        "eta": bbdm.eta,
        "spectral_weight": spectral_weight,
        "in_ch": getattr(net, "in_ch", None),
        "cond_ch": getattr(net, "cond_ch", 0),
        "base_ch": getattr(net, "base_ch", None),
        "time_dim": getattr(net, "time_dim", None),
        "groups": getattr(net, "groups", None),
        "upsample": getattr(net, "upsample", None),
    }


class TrainStep(nn.Module):
    """forward(x0, y) -> (loss, mse, spec). Обёртка под DDP.

    DDP оборачивает ЭТОТ модуль, а не `bbdm.model`: если подменить
    `bbdm.model` обёрткой, `BBDM.denoise` потеряет атрибут `cond_ch`
    (`getattr(self.model, "cond_ch", 0)` у DDP вернёт 0), и условие на Planck
    молча пропало бы; заодно ключи state_dict получили бы префикс `module.`
    и `load_checkpoint` перестал бы узнавать архитектуру. Так `bbdm.model`
    остаётся голым UNet-ом, а DDP синхронизирует градиенты всех параметров.
    """

    def __init__(self, bbdm, spectral_weight):
        super().__init__()
        self.bbdm = bbdm
        self.spectral_weight = spectral_weight

    def forward(self, x0, y):
        loss, terms = self.bbdm.loss(
            x0, y, spectral_weight=self.spectral_weight, return_terms=True
        )
        return loss, terms["mse"], terms["spec"]


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
    overwrite=False,
    monitor_dataset=None,
    monitor_every=MONITOR_EVERY,
    monitor_n=MONITOR_N,
    monitor_S=MONITOR_S,
    extra_state=None,
):
    """Обучает BBDM и возвращает (bbdm, ema).

    Args:
        batch_size: размер батча НА ОДИН процесс. При `torchrun` с k
            процессами глобальный батч = k * batch_size; чтобы число шагов
            на эпоху, `ema_start` (в шагах) и LR совпадали с одиночным
            прогоном, передавайте global_batch // k.
        spectral_weight: вес L1(log RAPSD) в лоссе. 0 -- чистый MSE.
        resume: продолжить с `last.pt` в checkpoint_dir, если он есть.
        overwrite: разрешить НОВЫЙ прогон в папке, где уже есть best.pt /
            last.pt. По умолчанию запрещено -- это единственная копия
            прошлого многочасового прогона.
        monitor_dataset: если задан, каждые `monitor_every` эпох на первых
            `monitor_n` его патчах прогоняется настоящий сэмплер
            (`monitor_S` шагов) и печатаются TF / r_ell / r_in по полосам
            (только на rank 0).
        extra_state: словарь, который кладётся в каждый чекпоинт под ключом
            "extra" (mu, sigma, списки тайлов, частота...). Нормализация и
            сплит должны жить рядом с весами: иначе их приходится пересчитывать
            из train-тайлов, и на другой машине они молча разойдутся.
    """
    main = D.is_main()
    world = D.world_size()
    distributed = world > 1

    last_path = os.path.join(checkpoint_dir, "last.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")

    if not resume and not overwrite and (
        os.path.exists(best_path) or os.path.exists(last_path)
    ):
        raise FileExistsError(
            f"В {checkpoint_dir} уже есть best.pt/last.pt. Новый прогон их "
            "перезапишет. Укажите новую checkpoint_dir, resume=True чтобы "
            "продолжить, или overwrite=True, если прошлый прогон не нужен."
        )
    if main:
        os.makedirs(checkpoint_dir, exist_ok=True)
    D.barrier()

    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device != "cpu"))
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if distributed:
        # Каждый ранг видит свою долю train; set_epoch() ниже перетасовывает
        # её по-разному каждую эпоху. drop_last: все ранги делают одинаковое
        # число шагов, иначе collective в backward повиснет.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world, rank=D.rank(), shuffle=True, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            drop_last=True, **loader_kwargs,
        )
        # val шардируется без повторов (в отличие от DistributedSampler,
        # который добивает хвост дублями и сместил бы среднее).
        val_loader = DataLoader(
            Subset(val_dataset, range(D.rank(), len(val_dataset), world)),
            batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # Хвостовой батч из 1-2 патчей дал бы заметно более шумную оценку
            # спектрального члена (если он включён) с тем же весом, что и полный.
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
    monitor_history = []

    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        bbdm.load_state_dict(ckpt["model"])
        ema.shadow.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        monitor_history = ckpt.get("monitor", [])
        if main:
            print(f"Resumed from epoch {start_epoch} (step {global_step})")

    step_module = TrainStep(bbdm, spectral_weight)
    if distributed:
        use_cuda = str(device).startswith("cuda")
        step_module = DDP(
            step_module,
            device_ids=[torch.device(device).index] if use_cuda else None,
            # m_t / delta_t -- константные буферы, рассылать их каждый шаг незачем.
            broadcast_buffers=False,
        )
        # При создании DDP веса rank 0 рассылаются остальным, так что старт
        # одинаковый, даже если модели строились с разными сидами.

    epoch_bar = tqdm(range(start_epoch, n_epochs), desc="Epochs", disable=not main)

    for epoch in epoch_bar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        bbdm.train()
        step_module.train()
        acc = torch.zeros(4, dtype=torch.float64)  # loss, mse, spec, n_batches
        train_bar = tqdm(train_loader, desc="  Train", leave=False, disable=not main)

        for x0, y in train_bar:
            x0, y = x0.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, mse, spec = step_module(x0, y)
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

            acc += torch.tensor([loss.item(), mse.item(), spec.item(), 1.0],
                                dtype=torch.float64)
            global_step += 1
            if main:
                train_bar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    mse=f"{mse.item():.6f}",
                    spec=f"{spec.item():.4f}",
                )

        acc = acc.to(device)
        D.all_reduce_sum(acc)  # no-op на одном процессе
        n_batches = max(float(acc[3]), 1.0)
        train_loss = float(acc[0]) / n_batches
        train_mse = float(acc[1]) / n_batches
        train_spec = float(acc[2]) / n_batches

        val_loss, val_mse, val_spec = _validate(
            bbdm, val_loader, device, spectral_weight, val_seed, distributed=distributed
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

        if main and monitor_dataset is not None and monitor_every and (epoch + 1) % monitor_every == 0:
            monitor_history.append(
                _run_monitor(bbdm, ema, monitor_dataset, device, monitor_n, monitor_S, epoch)
            )

        if main:
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
                # Гиперпараметры процесса и архитектуры рядом с весами: иначе при
                # инференсе легко молча взять другое T/s/eta или другую сеть.
                "hparams": _hparams(bbdm, spectral_weight),
                "monitor": monitor_history,
                "world_size": world,
                "extra": extra_state or {},
            }
            torch.save(state, last_path)
            if val_loss < best_val_loss:
                tqdm.write(
                    f"Epoch {epoch+1} saved best "
                    f"(val={val_loss:.6f}, mse={val_mse:.6f}, spec={val_spec:.4f})"
                )
                torch.save(state, best_path)

        # val_loss одинаков на всех рангах, так что best_val_loss обновляется
        # согласованно и без передачи от rank 0.
        best_val_loss = min(best_val_loss, val_loss)
        # Остальные ранги не должны убежать вперёд, пока rank 0 считает
        # монитор и пишет чекпоинт (иначе вечное ожидание в collective).
        D.barrier()

    return bbdm, ema


def load_checkpoint(path, device="cuda", use_ema=True):
    """Строит BBDM под архитектуру чекпоинта и загружает в неё веса.

    Архитектура (апсемплинг, канал условия, ширина) определяется по ключам
    и формам state_dict, а T и s -- по буферам m_t / delta_t, а не по
    hparams: у чекпоинтов run 3/4 этих полей ещё не было. Поэтому один и тот
    же вызов корректно открывает и старые чекпоинты (ConvTranspose2d, без
    условия), и новые.

    Args:
        use_ema: заменить веса сети EMA-тенью (так делается инференс).

    Returns:
        (bbdm в режиме eval на device, словарь чекпоинта).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"]
    hp = ckpt.get("hparams", {})

    upsample = "transpose" if "model.up1.up.weight" in state else "resize_conv"
    init_w = state["model.init_conv.weight"]
    in_ch = int(state["model.out_conv.weight"].shape[0])
    cond_ch = int(init_w.shape[1]) - in_ch
    base_ch = int(init_w.shape[0])
    time_dim = int(state["model.time_emb.net.0.weight"].shape[1])
    # Число групп GroupNorm по формам весов не восстановить.
    groups = hp.get("groups") or GROUPS

    T_ckpt = int(state["m_t"].numel())
    # delta_t при m = 1/2 равна s / 2.
    s_ckpt = round(2.0 * float(state["delta_t"][T_ckpt // 2 - 1]), 6)

    unet = UNet(in_ch=in_ch, base_ch=base_ch, time_dim=time_dim, groups=groups,
                cond_ch=cond_ch, upsample=upsample)
    bbdm = BBDM(unet, T=T_ckpt, s=s_ckpt, eta=hp.get("eta", ETA),
                spectral_weight=hp.get("spectral_weight", 0.0))
    bbdm.load_state_dict(state)
    if use_ema and "ema" in ckpt:
        bbdm.model.load_state_dict(ckpt["ema"])

    return bbdm.to(device).eval(), ckpt
