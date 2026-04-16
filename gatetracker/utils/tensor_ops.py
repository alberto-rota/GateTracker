"""Tensor operations and utility functions."""

import math
from typing import Dict, List
import numpy as np
from rich import print as nativeprint
import torch
import torch.nn.functional as F
import torchvision


__all__ = [
    "closest_multiple",
    "TTensor",
    "hwc",
    "tprint",
    "sp",
    "normalize_tensor",
    "embedding2chw",
    "chw2embedding",
    "collate",
    "dup_indexes",
    "embedding_mask_from_pixels",
    "embedding_confidence_from_pixels",
    "generate_random_pose_tensor",
    "resizeTransform",
    "millify",
    "coloredbar",
]


def closest_multiple(value, factor, mode="closest"):
    """
    Find the closest multiple of a factor to a given value.

    Args:
        value (int): The value to find the closest multiple for
        factor (int): The factor to use
        mode (str): One of 'closest', 'inf', or 'sup'
            - 'closest': returns the closest multiple (rounding)
            - 'inf': returns the largest multiple not exceeding value (floor)
            - 'sup': returns the smallest multiple not less than value (ceiling)

    Returns:
        int: The closest multiple according to the specified mode
    """
    if mode == "closest":
        return round(value / factor) * factor
    elif mode == "inf":
        return (value // factor) * factor
    elif mode == "sup":
        return ((value + factor - 1) // factor) * factor
    else:
        raise ValueError("Mode must be one of 'closest', 'inf', or 'sup'")


def TTensor(obj: object) -> torch.Tensor:
    """
    Converts an object to a torch.Tensor if it is not already one.

    Args:
        obj (object): The input object to be converted to a torch.Tensor.

    Returns:
        torch.Tensor: The input object converted to a torch.Tensor, or the original
                      object if it is already a torch.Tensor.
    """
    if not isinstance(obj, torch.Tensor):
        return torch.Tensor(obj)
    return obj


def hwc(image: torch.Tensor) -> torch.Tensor:
    """
    Reshapes a sensor image from CxHxW to HxWxC.

    Args:
        image (torch.Tensor): Input CxHxW image.

    Returns:
        torch.Tensor: Output HxWxC image.
    """
    return image.permute(2, 1, 0)


def tprint(args, shape=False, dtype=False, device=False, grad_fn=False, **kwargs):

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = []
    np.set_printoptions(precision=4, suppress=True)

    def tensor_to_string(tensor):
        return str(tensor.cpu().detach().numpy())

    for arg in args:
        if isinstance(arg, torch.Tensor):

            infos = "\n"
            if shape:
                infos += f"Shape: {tuple(arg.shape)}"
            if dtype:
                infos += f"Dtype {str(arg.dtype).split('torch.')[1]}"
            if device:
                infos += f"Device: {arg.device}"
            if grad_fn:
                infos += (
                    f"Grad_fn: {arg.grad_fn}" if arg.grad_fn is not None else "NOGRAD"
                )
            if shape or dtype or device or grad_fn:
                infos += "\n"
            infos += tensor_to_string(arg)

            output.append(infos)
        elif (isinstance(arg, list) or isinstance(arg, tuple)) and all(
            isinstance(x, torch.Tensor) for x in arg
        ):
            print(f"{len(arg)} elements:", [x.shape for x in arg])
        else:
            output.append(str(arg))
    nativeprint(sep.join(output), end=end)


def sp(size: tuple) -> tuple:
    """
    Converts a size tuple to a tuple of the same elements.

    Args:
        size (tuple): Input size tuple.

    Returns:
        tuple: Output size tuple.
    """
    return tuple(size)


def normalize_tensor(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def embedding2chw(
    embedding: torch.Tensor, embed_dim_last=True, aspect_ratio: float = None
) -> torch.Tensor:
    """
    Reorganizes the embedding output into CHW form, handling both square and non-square sequence lengths.

    Args:
        embedding (torch.Tensor): Input embedding tensor of shape (N, D).
        embed_dim_last (bool): If True, expects embedding dimension to be last, otherwise expects it second.
        aspect_ratio (float, optional): Desired width/height ratio. If None, will try to make as square as possible.

    Returns:
        torch.Tensor: Tensor of shape (D, H, W).
    """
    # Validate input tensor shape
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(0)

    if not embed_dim_last:
        embedding = embedding.permute(0, 2, 1)
    B, N, D = embedding.shape

    # Calculate dimensions based on sequence length and aspect ratio
    if aspect_ratio is None:
        def get_factors(n):
            factors = []
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            return min(factors, key=lambda x: abs(x[0] - x[1]))

        height, width = get_factors(N)
    else:
        height = int((N / aspect_ratio) ** 0.5)
        while N % height != 0:
            height -= 1
        width = N // height

        actual_ratio = width / height
        if abs(actual_ratio - aspect_ratio) > 0.5:
            print(
                f"Warning: Actual aspect ratio ({actual_ratio:.2f}) differs significantly from requested ({aspect_ratio:.2f})"
            )

    # Reshape the embedding from (B,N,D) to (B,H,W,D)
    chw_tensor = embedding.view(B, height, width, D).permute(0, 3, 1, 2)

    return chw_tensor


def chw2embedding(chw_tensor: torch.Tensor, embed_dim_last=False) -> torch.Tensor:
    """
    Converts a CHW-formatted tensor back into an embedding format.

    Args:
        chw_tensor (torch.Tensor): Input tensor of shape (B, D, H, W).
        embed_dim_last (bool): If True, returns embedding with dimension last, else second.

    Returns:
        torch.Tensor: Embedding tensor of shape (B, N, D) or (B, D, N).
    """
    if chw_tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor (B, D, H, W), but got shape {chw_tensor.shape}."
        )

    B, D, H, W = chw_tensor.shape
    N = H * W

    # Reshape from (B, D, H, W) to (B, N, D)
    embedding = chw_tensor.permute(0, 2, 3, 1).reshape(B, N, D)

    if not embed_dim_last:
        embedding = embedding.permute(0, 2, 1)

    return embedding


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of dictionaries into a dictionary of tensors.

    Args:
    batch (List[Dict]): List of dictionaries containing tensors.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing tensors stacked along the 0th dimension.
    """
    return {key: torch.stack([d[key] for d in batch]) for key in batch[0]}


def dup_indexes(arr):
    unique, counts = torch.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return [torch.where(arr == dup)[0].tolist() for dup in duplicates]


def embedding_mask_from_pixels(
    pixel_mask: torch.Tensor,
    patch_size: int = 8,
    embedding_dim: int = 768,
    min_valid_ratio: float = 1.0,
) -> torch.Tensor:
    """
    Convert a pixel-wise mask or confidence map to a patch-wise validity mask.

    A patch is considered valid if its average confidence is at least
    `min_valid_ratio`.

    Args:
        pixel_mask (torch.Tensor): Mask/confidence of shape (B, 3, H, W) where all channels are equal
        patch_size (int): Size of the patches for embeddings
        embedding_dim (int): Number of embedding features
        min_valid_ratio (float): Minimum average per-patch confidence required
            to mark the patch as valid. Default 1.0 preserves the old
            all-pixels-valid behavior.

    Returns:
        torch.Tensor: Binary mask for embeddings of shape (B, embedding_dim, H/patch_size, W/patch_size)
    """
    B, _, H, W = pixel_mask.shape

    assert torch.all(pixel_mask[:, 0] == pixel_mask[:, 1]) and torch.all(
        pixel_mask[:, 1] == pixel_mask[:, 2]
    ), "All channels in the mask must be equal"

    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions ({H}, {W}) must be divisible by patch_size {patch_size}"

    single_channel_mask = pixel_mask[:, 0]  # Shape: (B, H, W)

    H_p = H // patch_size
    W_p = W // patch_size

    patches = single_channel_mask.unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size
    )
    # Shape after unfold: (B, H_p, W_p, patch_size, patch_size)

    patch_confidence = patches.float().mean(dim=(-2, -1))
    valid_patches = patch_confidence >= min_valid_ratio
    # Shape: (B, H_p, W_p)

    embedding_mask = valid_patches.unsqueeze(1).expand(B, embedding_dim, H_p, W_p)

    return embedding_mask.float()


def embedding_confidence_from_pixels(
    pixel_mask: torch.Tensor, patch_size: int = 8
) -> torch.Tensor:
    """
    Convert a pixel-wise validity mask into a patch-wise confidence map.

    Args:
        pixel_mask: [B, 3, H, W] binary visibility mask
        patch_size: spatial size of each embedding patch

    Returns:
        [B, 1, H/patch_size, W/patch_size] confidence in [0, 1]
    """
    B, _, H, W = pixel_mask.shape
    assert (
        H % patch_size == 0 and W % patch_size == 0
    ), f"Image dimensions ({H}, {W}) must be divisible by patch_size {patch_size}"
    assert torch.all(pixel_mask[:, 0] == pixel_mask[:, 1]) and torch.all(
        pixel_mask[:, 1] == pixel_mask[:, 2]
    ), "All channels in the mask must be equal"

    return F.avg_pool2d(
        pixel_mask[:, :1].float(),
        kernel_size=patch_size,
        stride=patch_size,
    ).clamp_(0.0, 1.0)


def generate_random_pose_tensor(
    translation_minmax=None,
    euler_minmax=None,
    angle_unit="degrees",
    device=None,
    dtype=torch.float32,
):
    """
    Generates a random 6x1 pose tensor with translation and Euler angles.

    Parameters:
        translation_minmax (list or tuple of tuples): Specifies the (min, max) for each translation component.
            Format: [(t_x_min, t_x_max), (t_y_min, t_y_max), (t_z_min, t_z_max)]
            Example: [(-5, 5), (-10, 10), (-15, 15)]
            If None, defaults to [(-10, 10)] for all components.

        euler_minmax (list or tuple of tuples): Specifies the (min, max) for each Euler angle component.
            Format: [(alpha_min, alpha_max), (beta_min, beta_max), (gamma_min, gamma_max)]
            Example: [(-180, 180), (-90, 90), (-180, 180)]
            If None, defaults to [(-180, 180)] for all components.

        angle_unit (str): Unit for Euler angles. 'degrees' or 'radians'. Defaults to 'degrees'.

        device (torch.device, optional): The device on which to create the tensor.
            Defaults to None, which means the tensor is created on the CPU.

        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Defaults to torch.float32.

    Returns:
        torch.Tensor: A 6x1 tensor where:
            - Elements 0-2 are translations (t_x, t_y, t_z)
            - Elements 3-5 are Euler angles (alpha, beta, gamma)

    Raises:
        ValueError: If input ranges are not properly specified.
    """
    if translation_minmax is None:
        translation_minmax = [(-10, 10)] * 3
    if euler_minmax is None:
        euler_minmax = [(-180, 180)] * 3

    if (not isinstance(translation_minmax, (list, tuple))) or len(
        translation_minmax
    ) != 3:
        raise ValueError(
            "translation_minmax must be a list or tuple of three (min, max) tuples."
        )
    for idx, t_range in enumerate(translation_minmax):
        if (not isinstance(t_range, (list, tuple))) or len(t_range) != 2:
            raise ValueError(
                f"Each translation range must be a tuple/list of two values. Error at index {idx}."
            )
        if t_range[0] > t_range[1]:
            raise ValueError(
                f"Translation range min must be <= max. Error at index {idx}."
            )

    if (not isinstance(euler_minmax, (list, tuple))) or len(euler_minmax) != 3:
        raise ValueError(
            "euler_minmax must be a list or tuple of three (min, max) tuples."
        )
    for idx, e_range in enumerate(euler_minmax):
        if (not isinstance(e_range, (list, tuple))) or len(e_range) != 2:
            raise ValueError(
                f"Each Euler angle range must be a tuple/list of two values. Error at index {idx}."
            )
        if e_range[0] > e_range[1]:
            raise ValueError(
                f"Euler angle range min must be <= max. Error at index {idx}."
            )

    translations = []
    for idx, (t_min, t_max) in enumerate(translation_minmax):
        t = torch.empty(1, device=device, dtype=dtype).uniform_(t_min, t_max)
        translations.append(t)

    euler_angles = []
    for idx, (e_min, e_max) in enumerate(euler_minmax):
        angle = torch.empty(1, device=device, dtype=dtype).uniform_(e_min, e_max)
        euler_angles.append(angle)

    pose_vector = torch.cat(translations + euler_angles, dim=0).view(6, 1)

    if angle_unit.lower() == "degrees":
        pose_vector[3:6] = torch.deg2rad(pose_vector[3:6])
    elif angle_unit.lower() == "radians":
        pass
    else:
        raise ValueError("angle_unit must be either 'degrees' or 'radians'.")

    return pose_vector.T


def resizeTransform(height=384, width=384) -> torchvision.transforms.Compose:
    """
    Define a resize transformation.

    Returns:
    torchvision.transforms.Compose: A composition of torchvision transformations.
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((height, width), antialias=True),
        ]
    )


def millify(n: float) -> str:
    """
    Converts a number into a string with a suffix that indicates its scale (thousands, millions, etc.)

    Parameters:
    n (int, float): The number to be converted.

    Returns:
    str: The converted string.
    """
    millnames = ["", " Th", " M", " B", " T"]

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.1f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def coloredbar(parts: list, colors: list, num_blocks: int) -> str:
    """
    Creates a colored bar using unicode block characters.

    Args:
        parts (list): A list of numbers representing the size of each part of the bar.
        colors (list): A list of colors for each part of the bar.
        num_blocks (int): The total number of blocks in the bar.

    Returns:
        str: A string representing the colored bar.
    """
    assert len(parts) == len(colors)
    total = sum(parts)
    block = "\u25a0"
    bar = ""
    for part, color in zip(parts, colors):
        num_part_blocks = round(part / total * num_blocks)
        bar += f"[{color}]{block * num_part_blocks}[/]"
    return bar
