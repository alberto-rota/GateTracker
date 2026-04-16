"""Formatting utilities for logging and display."""

import re
from PIL import Image
import wandb


__all__ = [
    "metrics_for_wandb",
    "align",
    "strip_rich_markup",
    "RdGr",
    "abbrev_wandb_run_tag",
    "abbrev_console_metric_name",
]


def align(input_str, max_length, alignment):
    if alignment == "left":
        input_str = input_str[:max_length]
        return input_str.ljust(max_length)
    elif alignment == "right":
        input_str = input_str[-max_length:]
        return input_str.rjust(max_length)
    elif alignment == "center":
        if len(input_str) > max_length:
            start = (len(input_str) - max_length) // 2
            input_str = input_str[start : start + max_length]
        return input_str.center(max_length)
    else:
        raise ValueError("Alignment must be 'left', 'right', or 'center'.")


def abbrev_wandb_run_tag(runname: str) -> str:
    """
    Compact display for W&B run ids ``First-Second-Num`` → ``FirstInitialSecondInitialNum``.

    Examples:
        ``bright-water-42`` → ``BW42`` (FSNum: two initials + third field)
        ``feasible-glitter-797`` → ``FG797`` (spaces around ``-`` tolerated)

    Offline / missing names fall back to ``run``; single-token names use up to 6 characters.
    """
    if runname is None:
        return "run"
    s = str(runname).strip()
    if not s:
        return "run"
    if "offline" in s.lower():
        return "run"
    # [B] split on hyphens, trim spaces so ``a - b - c`` still works
    parts = [p.strip() for p in s.split("-") if p.strip()]
    if len(parts) >= 3:
        a, b = parts[0], parts[1]
        num = "-".join(parts[2:])
        fa = a[0].upper() if a else "?"
        fb = b[0].upper() if b else "?"
        return f"{fa}{fb}{num}"
    if len(parts) == 2:
        a, b = parts[0], parts[1]
        fa = a[0].upper() if a else "?"
        fb = b[0].upper() if b else "?"
        return f"{fa}{fb}"
    # [1] single segment
    token = parts[0]
    return token[:6] if len(token) > 6 else token


def abbrev_console_metric_name(name: str, max_len: int = 6) -> str:
    """
    Short label for console columns: CamelCase / snake_case → initials when long.
    ``EpipolarError`` → ``EE``; ``InlierRatio`` → ``IR``; ``refine_mean`` → ``rm``.
    """
    if not name:
        return ""
    if len(name) <= max_len:
        return name
    if "_" in name:
        bits = [b for b in name.split("_") if b]
        if len(bits) >= 2:
            short = "".join(b[0] for b in bits)
            return short[:max_len]
    chunks = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name)
    if len(chunks) >= 2:
        short = "".join(c[0] for c in chunks if c)
        return short[:max_len]
    return name[:max_len]


def RdGr(value):
    # Ensure the value is between 0 and 1
    value = max(0.0, min(1.0, value))

    # Map the value to a red (0) to green (1) gradient
    red = int(255 * (1 - value))
    green = int(255 * value)

    # Create a hex color code
    color = f"#{red:02x}{green:02x}00"

    # Create the text object with the specified color
    return f"[{color}]{(value*100):.2f}[/{color}]"


def metrics_for_wandb(metrics_dict, prefix_str, separator="/"):
    """
    Process dictionary keys to include a prefix if they don't already contain the separator.
    Skip any key-value pairs where the value is None.

    Args:
        metrics_dict (dict): The dictionary whose keys need to be processed
        prefix_str (str): The string to prepend to keys
        separator (str, optional): The separator character. Defaults to "/".

    Returns:
        dict: A new dictionary with the processed keys
    """
    result = {}

    for key, value in metrics_dict.items():
        if value is None:
            continue

        if separator in key:
            new_key = key
        else:
            new_key = f"{prefix_str}{separator}{key}"

        if isinstance(value, Image.Image):
            value = wandb.Image(value)
        result[new_key] = value

    return result


def strip_rich_markup(text):
    """
    Strip Rich markup tags from a string.

    Args:
        text (str): Text with Rich markup

    Returns:
        str: Text with Rich markup tags removed
    """
    # Remove [tag]...[/tag] pairs
    text = re.sub(r"\[([^\]]+)\](.*?)\[/\1\]", r"\2", text)

    # Remove remaining single tags like [tag]
    text = re.sub(r"\[([^\]]+)\]", "", text)

    return text
