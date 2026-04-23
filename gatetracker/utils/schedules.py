"""Piecewise schedules for training (epoch-based ramps)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

Milestone = Union[Sequence[float], Tuple[float, float]]


def _linear_ramp(epoch: int, start: float, end: float, ramp_epochs: int) -> float:
    """Linearly interpolate from ``start`` at epoch 0 to ``end`` at ``ramp_epochs``,
    then hold at ``end``. If ``ramp_epochs`` <= 0, returns ``start`` (no ramp;
    backward-compatible with configs that omit ramp keys).

    Args:
        epoch: 0-based epoch index.
        start: Value at epoch 0.
        end: Value reached at epoch ``ramp_epochs`` and held thereafter.
        ramp_epochs: Number of epochs over which to interpolate; 0 disables ramp.

    Returns:
        Scheduled scalar (plain Python ``float``, safe for W&B / Bernoulli ``p``).
    """
    if ramp_epochs <= 0:
        return float(start)
    t = min(max(epoch / float(ramp_epochs), 0.0), 1.0)
    return float(start + (end - start) * t)


def _piecewise_linear_epochs(epoch: int, milestones: Iterable[Milestone]) -> float:
    """Linear interpolation between ``(epoch_k, value_k)`` knots; hold the last value.

    ``milestones`` entries are ``[epoch, value]`` or ``(epoch, value)`` with non-decreasing
    epoch (0-based). If ``epoch`` is before the first knot, returns the first value.

    Args:
        epoch: 0-based epoch index.
        milestones: Sorted (or sortable) list of ``(e, v)`` pairs.

    Returns:
        Scheduled scalar.
    """
    knots: List[Tuple[int, float]] = []
    for m in milestones:
        if m is None:
            continue
        e_i = int(m[0])
        v_i = float(m[1])
        knots.append((e_i, v_i))
    if not knots:
        return 0.0
    knots.sort(key=lambda t: t[0])
    e_now = int(epoch)
    if e_now <= knots[0][0]:
        return float(knots[0][1])
    if e_now >= knots[-1][0]:
        return float(knots[-1][1])
    for i in range(len(knots) - 1):
        e0, v0 = knots[i]
        e1, v1 = knots[i + 1]
        if e0 <= e_now <= e1:
            if e1 == e0:
                return float(v1)
            t = (e_now - e0) / float(e1 - e0)
            return float(v0 + (v1 - v0) * t)
    return float(knots[-1][1])


__all__ = ["_linear_ramp", "_piecewise_linear_epochs"]
