"""Backward-compatible entrypoint if ``program: sweep_agent.yaml`` points at repo root."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    _script = Path(__file__).resolve().parent / "scripts" / "sweep_agent.py"
    runpy.run_path(str(_script), run_name="__main__")
