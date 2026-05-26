import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm.notebook import tqdm
from bbdm.config import CHECKPOINT_DIR, EMA_START


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = deepcopy(model).eval()

    @torch.no_grad()
    def update(self, model):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data = self.decay * s_param.data + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow.state_dict())


def train(
    bbdm,
    train_dataset,
    val_dataset,
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    ema_start=10000,
    device="cuda",
    checkpoint_dir=CHECKPOINT_DIR,
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    bbdm = bbdm.to(device)
    optimizer = torch.optim.Adam(bbdm.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3000
    )
    ema = EMA(bbdm.model, decay=0.995)

    global_step = 0
    best_val_loss = float("inf")
    epoch_bar = tqdm(range(n_epochs), desc="Epochs")

    for epoch in epoch_bar:
        bbdm.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="  Train", leave=False)

        for x0, y in train_bar:
            x0, y = x0.to(device), y.to(device)
            optimizer.zero_grad()
            loss = bbdm.loss(x0, y)
            loss.backward()
            nn.utils.clip_grad_norm_(bbdm.parameters(), 1.0)
            optimizer.step()

            if global_step >= ema_start:
                ema.update(bbdm.model)

            train_loss += loss.item()
            global_step += 1
            train_bar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss /= len(train_loader)

        bbdm.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="  Val", leave=False)

        with torch.no_grad():
            for x0, y in val_bar:
                x0, y = x0.to(device), y.to(device)
                l = bbdm.loss(x0, y).item()
                val_loss += l
                val_bar.set_postfix(loss=f"{l:.6f}")

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}",
            val=f"{val_loss:.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            step=global_step,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": bbdm.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(checkpoint_dir, "best.pt"),
            )
            tqdm.write(f"Epoch {epoch+1} saved best (val={val_loss:.6f})")

    return bbdm, ema
