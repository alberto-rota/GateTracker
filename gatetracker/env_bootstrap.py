"""Repository ``.env`` loading and path helpers (DATASET_DIR, RESULTS_DIR, WEIGHTS_DIR)."""

import os
import sys
import warnings
from pathlib import Path

# PyTorch ``torch.utils.checkpoint`` still nests deprecated ``torch.cpu.amp.autocast``;
# the warning fires during backward recomputation, so it must be filtered globally
# (not only inside a short-lived ``catch_warnings`` around ``checkpoint()``).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*",
)


def repository_root():
    # gatetracker/env_bootstrap.py -> repository root (parent of ``gatetracker/``)
    return Path(__file__).resolve().parent.parent


def dotenv_file_path():
    return repository_root() / ".env"


def setdefault_cpu_thread_env():
    """Cap BLAS/OpenMP threads per process (DDP / torchrun-friendly).

    Uses ``setdefault`` so values from the shell or ``.env`` (loaded first) win.
    Aligns with ``torchrun``'s recommendation to set ``OMP_NUM_THREADS=1`` when
    using many worker processes.
    """
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(key, val)


def require_dotenv_before_pipeline(purpose="training or evaluation"):
    """Fail fast if ``.env`` is missing; otherwise load variables into the process environment."""
    env_path = dotenv_file_path()
    if not env_path.is_file():
        print(
            f"\n[GateTracker] ERROR: `.env` not found at `{env_path}`.\n"
            f"Create it with DATASET_DIR, RESULTS_DIR, and WEIGHTS_DIR before starting "
            f"{purpose}. Aborting.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    from dotenv import load_dotenv

    load_dotenv(env_path)
    setdefault_cpu_thread_env()


def _expand_path(p):
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    return os.path.normpath(os.path.expandvars(os.path.expanduser(s)))


def dataset_base_dir():
    """``DATASET_DIR`` if set, else legacy ``DATASET_ROOTDIR``."""
    return _expand_path(os.environ.get("DATASET_DIR")) or _expand_path(
        os.environ.get("DATASET_ROOTDIR")
    )


def resolve_dataset_filesystem_path(config_path, dataset_name):
    """Resolve a dataset ``PATH`` from YAML using ``DATASET_DIR`` / ``DATASET_ROOTDIR``."""
    base = dataset_base_dir()
    if not config_path:
        if not base:
            return None
        return os.path.join(base, dataset_name)

    raw = str(config_path).strip()
    expanded = os.path.normpath(os.path.expandvars(os.path.expanduser(raw)))

    if "$" in raw or raw.startswith("~"):
        return expanded

    if base:
        if os.path.isabs(expanded):
            return expanded
        rel = expanded.lstrip("/\\")
        return os.path.normpath(os.path.join(base, rel))
    return expanded


def results_dir_default(package_runs_fallback):
    """Directory for run artifacts (wandb run name = one subdirectory)."""
    return _expand_path(os.environ.get("RESULTS_DIR")) or package_runs_fallback


def weights_dir_optional():
    return _expand_path(os.environ.get("WEIGHTS_DIR"))


def _unique_nonempty_dirs(paths):
    seen = set()
    out = []
    for p in paths:
        q = _expand_path(p)
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def pretrained_checkpoint_path_candidates(checkpoint_ref, runs_dir):
    """Ordered checkpoint paths to probe for a run name or relative artifact path."""
    ref = os.path.expandvars(os.path.expanduser(str(checkpoint_ref).strip()))
    if not ref:
        return []

    repo = repository_root()
    wd = weights_dir_optional()
    rd = _expand_path(os.environ.get("RESULTS_DIR"))
    run_root = _expand_path(runs_dir)

    bases = _unique_nonempty_dirs(
        (
            wd,
            rd,
            run_root,
            str(repo / "gatetracker" / "runs"),
            str(repo / "runs"),
        )
    )

    looks_like_run_name = (os.sep not in ref) and (not os.path.splitext(ref)[1])
    candidates = []

    if looks_like_run_name:
        for b in bases:
            candidates.extend(
                (
                    os.path.join(b, ref, "models", f"{ref}_checkpoint.pth"),
                    os.path.join(b, ref, "checkpoints", "weights_best.pt"),
                    os.path.join(b, f"{ref}_checkpoint.pth"),
                    os.path.join(b, f"{ref}.pt"),
                )
            )
        candidates.extend(
            (
                os.path.join("runs", ref, "models", f"{ref}_checkpoint.pth"),
                os.path.join("runs", ref, "checkpoints", "weights_best.pt"),
                os.path.join("checkpoints", f"{ref}_checkpoint.pth"),
                os.path.join("checkpoints", f"{ref}.pt"),
            )
        )
    else:
        # Relative path: prefer WEIGHTS_DIR, then RESULTS_DIR / runs_dir, then cwd-relative.
        if not os.path.isabs(ref):
            for b in bases:
                candidates.append(os.path.join(b, ref))
        candidates.append(ref)

    # De-dupe while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        n = os.path.normpath(c)
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered
