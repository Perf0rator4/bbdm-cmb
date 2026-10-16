"""Минимальные помощники для обучения на нескольких GPU (torch.distributed).

Запуск: `torchrun --nproc_per_node=2 scripts/train_freq.py ...`. Без torchrun
(обычный `python`) всё это превращается в no-op, и обучение идёт как раньше на
одном процессе: `world_size() == 1`, `is_main() == True`.
"""

import datetime
import os

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if is_distributed() else 0


def world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return rank() == 0


def init_distributed(timeout_minutes=60):
    """Поднимает process group по переменным окружения torchrun.

    Returns:
        (rank, world_size, local_rank, device) -- device уже выбран
        (`cuda:<local_rank>` либо `cpu`). Без torchrun возвращает
        (0, 1, 0, "cuda" или "cpu") и ничего не инициализирует.

    Таймаут большой: пока rank 0 считает спектральный монитор или пишет
    чекпоинт, остальные стоят на следующем collective, и дефолтных минут
    для долгой финальной оценки не хватило бы.
    """
    has_cuda = torch.cuda.is_available()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, ("cuda" if has_cuda else "cpu")

    r = int(os.environ["RANK"])
    w = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", 0))

    if has_cuda:
        torch.cuda.set_device(lr)
        backend, device = "nccl", f"cuda:{lr}"
    else:
        backend, device = "gloo", "cpu"

    dist.init_process_group(
        backend=backend, rank=r, world_size=w,
        timeout=datetime.timedelta(minutes=timeout_minutes),
    )
    return r, w, lr, device


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_sum(tensor):
    """In-place сумма тензора по всем рангам (no-op на одном процессе)."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup():
    if is_distributed():
        dist.destroy_process_group()
