"""Distributed training helpers (DDP / DP). Mirrors UnReflectAnything DISTRIBUTE pattern."""

import os
from typing import Any, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Sampler

# Same string values as UnReflectAnything utilities.config
DISTRIBUTE_SINGLEGPU = "singlegpu"
DISTRIBUTE_DP = "dp"
DISTRIBUTE_DDP = "ddp"


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_distribute(value: Any) -> str:
    """Return one of singlegpu | dp | ddp."""
    if value is None:
        return DISTRIBUTE_SINGLEGPU
    s = str(value).strip().lower()
    if s in ("single", "singlegpu", "cpu"):
        return DISTRIBUTE_SINGLEGPU
    if s == "dp":
        return DISTRIBUTE_DP
    if s == "ddp":
        return DISTRIBUTE_DDP
    return DISTRIBUTE_SINGLEGPU


def is_ddp_enabled(config: Any) -> bool:
    return normalize_distribute(_cfg_get(config, "DISTRIBUTE", DISTRIBUTE_SINGLEGPU)) == DISTRIBUTE_DDP


def is_dp_enabled(config: Any) -> bool:
    return normalize_distribute(_cfg_get(config, "DISTRIBUTE", DISTRIBUTE_SINGLEGPU)) == DISTRIBUTE_DP


def dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size_int() -> int:
    if not dist_initialized():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DistributedDataParallel / DataParallel / DataParallel-style wrappers."""
    m = model
    if isinstance(m, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        m = m.module
    # Optional nested wrapper (UnReflectAnything DataParallelWrapper)
    inner = getattr(m, "module", None)
    if inner is not None and isinstance(m, nn.Module) and m.__class__.__name__ == "DataParallelWrapper":
        m = inner
    return m


def init_process_group_from_config(config: Any) -> None:
    """Initialize torch.distributed when DISTRIBUTE==ddp (idempotent if already initialized)."""
    if not is_ddp_enabled(config):
        return
    if dist_initialized():
        return
    backend = str(_cfg_get(config, "DISTBACKEND", "nccl"))
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
    dist.init_process_group(backend=backend)


def fill_ddp_env_into_config(config: Any) -> None:
    """Set RANK / LOCAL_RANK / WORLD_SIZE on config from environment (torchrun)."""
    if "RANK" in os.environ:
        config["RANK"] = int(os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:
        config["LOCAL_RANK"] = int(os.environ["LOCAL_RANK"])
    if "WORLD_SIZE" in os.environ:
        config["WORLD_SIZE"] = int(os.environ["WORLD_SIZE"])


def require_torchrun_env() -> None:
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "DISTRIBUTE is 'ddp' but this process was not launched with torchrun. "
            "Example: torchrun --standalone --nproc_per_node=4 train.py -c pretrain.yaml --ddp"
        )


def barrier() -> None:
    if dist_initialized():
        dist.barrier()


def all_reduce_sum_scalars(
    sum_value: float, count: float, device: torch.device
) -> Tuple[float, float]:
    """Sum and count across ranks (for weighted mean)."""
    if not dist_initialized():
        return sum_value, count
    t = torch.tensor([sum_value, count], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0].item()), float(t[1].item())


class ShardedListSampler(Sampler):
    """Strides a dataset's index list across DDP ranks: indices[rank::world_size].

    Expects ``dataset.sampler`` to be a list of ints (GateTracker MultiDataset convention).
    Pads shorter per-rank shards so every rank sees the same number of indices per epoch
    (required for ``DistributedDataParallel`` synchronization).
    """

    def __init__(self, dataset: Any, rank: int, world_size: int) -> None:
        self._dataset = dataset
        self._rank = int(rank)
        self._world_size = max(1, int(world_size))

    def _device_for_dist(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def _padded_shard(self) -> List[int]:
        indices = self._dataset.sampler
        ws = self._world_size
        r = self._rank
        if ws <= 1:
            return list(indices)
        mine = list(indices[r::ws])
        if not mine:
            fallback = int(indices[0]) if len(indices) > 0 else 0
            mine = [fallback]
        if not dist_initialized():
            return mine
        dev = self._device_for_dist()
        t = torch.tensor([len(mine)], device=dev, dtype=torch.long)
        sizes = [torch.zeros(1, device=dev, dtype=torch.long) for _ in range(ws)]
        dist.all_gather(sizes, t)
        max_len = max(int(s.item()) for s in sizes)
        pad_idx = int(mine[-1])
        while len(mine) < max_len:
            mine.append(pad_idx)
        return mine

    def __len__(self) -> int:
        return len(self._padded_shard())

    def __iter__(self):
        yield from self._padded_shard()

    def set_epoch(self, epoch: int) -> None:
        """API parity with ``DistributedSampler``; index order comes from ``reset_sampler`` / init."""
        del epoch  # shuffle epoch is applied in ``MultiDataset._create_combined_sampler`` when resetting
