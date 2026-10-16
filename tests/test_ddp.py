"""DDP-обучение на двух процессах (gloo, CPU): веса, лосс, чекпоинты.

GPU не нужны. Проверяется то, на что можно полагаться при запуске на двух A40:
  * после обучения веса идентичны на обоих рангах, хотя стартовали с разных сидов;
  * валидационный лосс, сведённый all_reduce, равен одиночному;
  * чекпоинт пишет один процесс, `load_checkpoint` его читает, архитектура на месте;
  * через DDP-обёртку не теряется канал условия `cond_ch`.

Запуск: `pytest tests/test_ddp.py` или `python tests/test_ddp.py` (~20-40 c).
"""

import json
import os
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm import dist as D  # noqa: E402
from bbdm.early_stop import EarlyStopper  # noqa: E402
from bbdm.model.bbdm import BBDM  # noqa: E402
from bbdm.model.unet import UNet  # noqa: E402
from bbdm.train import _validate, load_checkpoint, train  # noqa: E402

WORLD = 2


class _PairDataset:
    def __init__(self, n, size=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [
            (torch.randn(1, size, size, generator=g), torch.randn(1, size, size, generator=g))
            for _ in range(n)
        ]
        self.mu, self.sigma = 0.0, 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _DeterministicLoss:
    """Стенд для проверки сведения val-лосса: лосс зависит только от данных."""

    def eval(self):
        return self

    def loss(self, x0, y, spectral_weight=None, generator=None, return_terms=False):
        mse = ((x0 - y) ** 2).mean()
        zero = torch.zeros(())
        return (mse, {"mse": mse, "spec": zero}) if return_terms else mse


def _flat(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()])


def _worker(rank, port, tmpdir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank))
    r, w, _, device = D.init_distributed(timeout_minutes=5)
    out = {"rank": r, "world": w, "device": device}
    try:
        assert (r, w) == (rank, WORLD)

        # 1) val-лосс: сведение по рангам == одиночный подсчёт.
        ds = _PairDataset(8)
        shard = torch.utils.data.Subset(ds, range(rank, len(ds), WORLD))
        loader = torch.utils.data.DataLoader(shard, batch_size=2, shuffle=False)
        v_dist, _, _ = _validate(_DeterministicLoss(), loader, device, 0.0, 0, distributed=True)
        # одиночный подсчёт: среднее по всем 8 образцам (батчи равны по размеру)
        v_single = float(sum(((a - b) ** 2).mean() for a, b in
                             (ds[i] for i in range(len(ds)))) / len(ds))
        out["val_dist"], out["val_single"] = v_dist, v_single

        # 2) обучение с РАЗНЫМИ сидами на рангах (DDP обязан выровнять старт).
        torch.manual_seed(100 + rank)
        unet = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1)
        bbdm = BBDM(unet, T=100, s=0.5, eta=0.0)
        train_ds, val_ds = _PairDataset(8, seed=1), _PairDataset(4, seed=2)
        ckdir = os.path.join(tmpdir, "ck")
        bbdm, ema = train(
            bbdm, train_ds, val_ds, n_epochs=2, batch_size=2, lr=1e-3, ema_start=2,
            device=device, checkpoint_dir=ckdir, num_workers=0,
            monitor_dataset=val_ds, monitor_every=1, monitor_n=2, monitor_S=3,
            extra_state={"freq": 150, "mu": 1.5, "sigma": 2.5},
        )

        # 3) веса и EMA одинаковы на всех рангах.
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, (_flat(bbdm.model).tolist(), _flat(ema.shadow).tolist()))
        a, b = gathered
        out["max_param_diff"] = max(abs(x - y) for x, y in zip(a[0], b[0]))
        out["max_ema_diff"] = max(abs(x - y) for x, y in zip(a[1], b[1]))
        out["finite"] = all(map(lambda v: v == v, a[0]))
        out["moved"] = float(torch.tensor(a[0]).abs().sum()) > 0

        D.barrier()
        if rank == 0:
            assert os.path.exists(os.path.join(ckdir, "best.pt"))
            loaded, ck = load_checkpoint(os.path.join(ckdir, "last.pt"), device="cpu")
            out["cond_ch"] = loaded.model.cond_ch
            out["ckpt_world"] = ck["world_size"]
            out["ckpt_extra"] = ck["extra"]
            out["ckpt_monitor"] = len(ck["monitor"])
            out["keys_clean"] = not any(k.startswith("module.") for k in ck["model"])
            # last.pt -- это ровно то, что было в живых EMA-весах в конце
            out["ema_roundtrip_diff"] = float((_flat(loaded.model) - _flat(ema.shadow).cpu()).abs().max())
    finally:
        D.cleanup()
    with open(os.path.join(tmpdir, f"res_{rank}.json"), "w") as f:
        json.dump(out, f)


def _worker_stop(rank, port, tmpdir):
    """Останов решает rank 0, и ОБА ранга обязаны выйти из цикла на одной эпохе."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(WORLD), LOCAL_RANK=str(rank))
    r, w, _, device = D.init_distributed(timeout_minutes=2)   # завис бы -> ошибка, а не вечное ожидание
    out = {"rank": r}
    try:
        torch.manual_seed(100 + rank)
        unet = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1)
        bbdm = BBDM(unet, T=100, s=0.5, eta=0.0)
        ds = _PairDataset(8, seed=1)
        # tf_max < 0: любое положительное TF -- нарушение, останов на первой проверке
        stopper = EarlyStopper(tf_max=-1.0, tf_strikes=1, warmup_checks=0)
        ckdir = os.path.join(tmpdir, "ck_stop")
        bbdm, ema = train(
            bbdm, ds, ds, n_epochs=5, batch_size=2, lr=1e-3, ema_start=2,
            device=device, checkpoint_dir=ckdir, num_workers=0,
            monitor_dataset=ds, monitor_every=1, monitor_n=2, monitor_S=3,
            early_stop=stopper,
        )
        out["finished"] = True
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, _flat(bbdm.model).tolist())
        out["max_param_diff"] = max(abs(x - y) for x, y in zip(*gathered))
        D.barrier()
        if rank == 0:
            ck = torch.load(os.path.join(ckdir, "last.pt"), weights_only=False, map_location="cpu")
            out["epoch"] = ck["epoch"]
            out["reason"] = ck["stop_reason"]
            out["monitor_n"] = len(ck["monitor"])
    finally:
        D.cleanup()
    with open(os.path.join(tmpdir, f"stop_{rank}.json"), "w") as f:
        json.dump(out, f)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run():
    tmpdir = tempfile.mkdtemp(prefix="ddp_test_")
    mp.spawn(_worker, args=(_free_port(), tmpdir), nprocs=WORLD, join=True)
    res = []
    for r in range(WORLD):
        with open(os.path.join(tmpdir, f"res_{r}.json")) as f:
            res.append(json.load(f))
    return res


def test_ddp_two_processes():
    r0, r1 = _run()

    # val-лосс: все ранги получили одно и то же число, равное одиночному
    assert abs(r0["val_dist"] - r1["val_dist"]) < 1e-12
    assert abs(r0["val_dist"] - r0["val_single"]) < 1e-6

    # синхронность весов и EMA при разных стартовых сидах
    assert r0["max_param_diff"] < 1e-6, f"веса разъехались: {r0['max_param_diff']}"
    assert r0["max_ema_diff"] < 1e-6
    assert r0["finite"] and r0["moved"]

    # чекпоинт: пишет rank 0, архитектура и условие на месте, ключи без `module.`
    assert r0["cond_ch"] == 1
    assert r0["ckpt_world"] == WORLD
    assert r0["ckpt_extra"] == {"freq": 150, "mu": 1.5, "sigma": 2.5}
    assert r0["ckpt_monitor"] == 2
    assert r0["keys_clean"]
    assert r0["ema_roundtrip_diff"] < 1e-6


def test_ddp_early_stop_is_synchronised():
    tmpdir = tempfile.mkdtemp(prefix="ddp_stop_")
    mp.spawn(_worker_stop, args=(_free_port(), tmpdir), nprocs=WORLD, join=True)
    res = []
    for r in range(WORLD):
        with open(os.path.join(tmpdir, f"stop_{r}.json")) as f:
            res.append(json.load(f))
    r0, r1 = res
    assert r0["finished"] and r1["finished"], "оба ранга должны выйти из train()"
    assert r0["epoch"] == 0 and r0["monitor_n"] == 1, "стоп на первой же проверке"
    assert "TF guard" in r0["reason"]
    assert r0["max_param_diff"] < 1e-6


if __name__ == "__main__":
    failures = 0
    for fn in (test_ddp_two_processes, test_ddp_early_stop_is_synchronised):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
