"""Piecewise schedules for training (epoch-based ramps)."""


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


__all__ = ["_linear_ramp"]
