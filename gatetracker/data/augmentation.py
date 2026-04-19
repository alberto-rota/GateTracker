import torchvision.transforms as tvt
import torch
from utilities.rotations import (
    euler2mat,
    mat2euler,
)  # Assuming utilities.py contains these functions


def photometric_noise_augmentation(
    framestack: torch.Tensor,
    *,
    p_blur: float = 0.3,
    p_gamma: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
    noise_std_max: float = 0.02,
    target_only: bool = True,
) -> torch.Tensor:
    """
    Fully-vectorized photometric perturbation of a framestack, designed to
    inject appearance variation that mimics real frame-to-frame photometric
    differences (sensor noise, residual brightness/contrast drift, light
    blur, gamma shifts). Intended to be applied on top of ``color_augmentation``
    to the NVS-warped target view only, so that the source view is left clean.

    All operations are batched and use broadcasted per-sample random tensors,
    no explicit Python loops over the batch dimension.

    Args:
        framestack:     [2, 3, H, W] (unbatched) or [B, 2, 3, H, W] (batched)
                        float tensor in ``[0, 1]``.
        p_blur:         batch-level probability of applying Gaussian blur.
        p_gamma:        batch-level probability of applying random gamma.
        brightness:     max absolute jitter on the brightness scale,
                        per-sample scale drawn in ``[1 - brightness, 1 + brightness]``.
        contrast:       max absolute jitter on the contrast scale,
                        per-sample scale drawn in ``[1 - contrast, 1 + contrast]``.
        noise_std_max:  upper bound of the ``U[0, noise_std_max]`` distribution
                        from which the per-sample additive Gaussian noise std is drawn.
        target_only:    when True apply the perturbation only to ``framestack[:, 1]``
                        (or ``framestack[1]`` for unbatched input), i.e. to the
                        NVS-warped target view.

    Returns:
        Tensor with the same shape as ``framestack`` and values in ``[0, 1]``.
    """
    unbatched = framestack.ndim == 4  # [2, 3, H, W]
    out = framestack.clone()
    out_b = out.unsqueeze(0) if unbatched else out  # [B, 2, 3, H, W]

    if target_only:
        target = out_b[:, 1]  # [B, 3, H, W] (view into ``out_b``)
    else:
        B, T = out_b.shape[:2]
        target = out_b.view(B * T, *out_b.shape[2:])  # [B*T, 3, H, W]

    device = target.device
    dtype = target.dtype
    N = target.shape[0]

    # Brightness jitter:
    #   b_scale: [N, 1, 1, 1] ~ U[1 - brightness, 1 + brightness]
    b_scale = (1.0 - brightness) + (2.0 * brightness) * torch.rand(
        N, 1, 1, 1, device=device, dtype=dtype,
    )  # [N, 1, 1, 1]
    target.mul_(b_scale).clamp_(0.0, 1.0)

    # Contrast jitter relative to per-sample, per-channel mean:
    #   c_scale: [N, 1, 1, 1] ~ U[1 - contrast, 1 + contrast]
    c_scale = (1.0 - contrast) + (2.0 * contrast) * torch.rand(
        N, 1, 1, 1, device=device, dtype=dtype,
    )  # [N, 1, 1, 1]
    target_mean = target.mean(dim=(2, 3), keepdim=True)  # [N, 3, 1, 1]
    target.copy_(((target - target_mean) * c_scale + target_mean).clamp_(0.0, 1.0))

    # Per-pixel Gaussian noise with per-sample std in U[0, noise_std_max]:
    if noise_std_max > 0.0:
        std = noise_std_max * torch.rand(
            N, 1, 1, 1, device=device, dtype=dtype,
        )  # [N, 1, 1, 1]
        noise = torch.randn_like(target) * std  # [N, 3, H, W]
        target.add_(noise).clamp_(0.0, 1.0)

    # Per-sample gamma correction in [0.8, 1.2] (batch-gated by p_gamma):
    if p_gamma > 0.0 and torch.rand(1).item() < p_gamma:
        gamma = 0.8 + 0.4 * torch.rand(
            N, 1, 1, 1, device=device, dtype=dtype,
        )  # [N, 1, 1, 1]
        target.clamp_(min=1e-8, max=1.0).pow_(gamma)

    # Gaussian blur with a kernel uniformly chosen from {3, 5}
    # (batch-gated by p_blur; torchvision does not accept per-sample kernels):
    if p_blur > 0.0 and torch.rand(1).item() < p_blur:
        kernel_size = 3 if torch.rand(1).item() < 0.5 else 5  # int
        sigma = 0.5 + float(torch.rand(1).item())  # sigma ~ U[0.5, 1.5]
        blurred = tvt.functional.gaussian_blur(
            target, kernel_size=kernel_size, sigma=sigma,
        )  # [N, 3, H, W]
        target.copy_(blurred)

    if not target_only:
        # Reshape back into ``[B, 2, 3, H, W]``.
        out_b = target.view(*out_b.shape)
        if unbatched:
            return out_b.squeeze(0)
        return out_b

    return out_b.squeeze(0) if unbatched else out_b


def color_augmentation(
    framestack: torch.Tensor, camera_pose: list, p: float = 0.2, target_only=False
) -> tuple:
    """
    Applies a series of color augmentations to a given tensor of images, with a set probability.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, not modified by this function.
    - p (float): Probability of applying sharpness adjustments. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the augmented image tensor and the unchanged transformation list.
    """
    # Define a series of color augmentations

    def identity_transform(x):
        return x

    applyjitter = torch.rand(1) < p
    applyblur = torch.rand(1) < p
    applyequalize = torch.rand(1) < p
    augmentation_composition = tvt.Compose(
        [
            (
                tvt.ColorJitter(
                    brightness=(0.6, 1.4),
                    contrast=(0.4, 1.8),
                    saturation=(0.6, 1.8),
                    hue=(-0.1, 0.1),
                )
                if applyjitter
                else tvt.Lambda(identity_transform)
            ),
            tvt.RandomAdjustSharpness(sharpness_factor=0, p=p),
            tvt.RandomAdjustSharpness(sharpness_factor=2, p=p),
            (
                tvt.GaussianBlur(kernel_size=3, sigma=(0.01, 10))
                if applyblur
                else tvt.Lambda(identity_transform)
            ),
            # (
            #     tvt.RandomEqualize(p=applyequalize)
            #     if applyequalize
            #     else tvt.Lambda(identity_transform)
            # ),
        ]
    )

    if target_only:
        if framestack.ndim == 5:
            source, target = framestack[:, 0], framestack[:, 1]
        else:
            source, target = framestack[0], framestack[1]

        target_augmented = augmentation_composition(target)
    else:
        framestack = augmentation_composition(
            framestack
        )  # Apply augmentations to the image tensor
    return (
        (
            torch.stack(
                [source, target_augmented], dim=1 if framestack.ndim == 5 else 0
            ),
            camera_pose,
        )
        if target_only
        else framestack
    )


def geometric_augmentation(
    framestack: torch.Tensor,
    camera_pose: list,
    depthstack=None,
    p: float = 0.2,
) -> tuple:
    """
    Applies random geometric augmentations to a tensor of images, modifying the transformation vector accordingly, wiith
    a set probability.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, modified based on applied augmentations.
    - p (float): Probability of applying each geometric augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the augmented image tensor and the adjusted transformation list.
    """
    hflip = torch.rand(1) < p
    vflip = torch.rand(1) < p
    cwrotate = torch.rand(1) < p
    ccwrotate = torch.rand(1) < p
    unbatched = camera_pose.ndim == 1
    if depthstack is None:
        depthstack = torch.ones_like(framestack)
    if unbatched:
        camera_pose = camera_pose.unsqueeze(0)
        depthstack = depthstack.unsqueeze(0)
        source, target = framestack[0], framestack[1]
        depthsource, depthtarget = depthstack[0], depthstack[1]
    else:
        source, target = framestack[:, 0], framestack[:, 1]
        depthsource, depthtarget = (
            depthstack[:, 0],
            depthstack[:, 1],
        )  # Apply horizontal flip if selected
    if hflip:
        source = tvt.functional.hflip(source)
        target = tvt.functional.hflip(target)
        depthsource = tvt.functional.hflip(depthsource)
        depthtarget = tvt.functional.hflip(depthtarget)
        camera_pose[:, 0] = -camera_pose[:, 0]
    if vflip:
        source = tvt.functional.vflip(source)
        target = tvt.functional.vflip(target)
        depthsource = tvt.functional.vflip(depthsource)
        depthtarget = tvt.functional.vflip(depthtarget)
        camera_pose[:, 1] = -camera_pose[:, 1]

    if cwrotate:
        source = tvt.functional.rotate(source, -90, expand=True)
        target = tvt.functional.rotate(target, -90, expand=True)
        depthsource = tvt.functional.rotate(depthsource, -90, expand=True)
        depthtarget = tvt.functional.rotate(depthtarget, -90, expand=True)
        camera_pose[:, 0] = -camera_pose[:, 1]
        camera_pose[:, 1] = camera_pose[:, 0]

    if ccwrotate:
        source = tvt.functional.rotate(source, 90, expand=True)
        target = tvt.functional.rotate(target, 90, expand=True)
        depthsource = tvt.functional.rotate(depthsource, 90, expand=True)
        depthtarget = tvt.functional.rotate(depthtarget, 90, expand=True)
        camera_pose[:, 0] = camera_pose[:, 1]
        camera_pose[:, 1] = -camera_pose[:, 0]

    framestack = torch.stack([source, target], dim=0 if unbatched else 1)
    depthstack = torch.stack([depthsource, depthtarget], dim=0 if unbatched else 1)
    return framestack, camera_pose.squeeze(0) if unbatched else camera_pose, depthstack


def reverse_augmentation(
    framestack: torch.Tensor, camera_pose: list, p: float = 0.2
) -> tuple:
    """
    Optionally applies reverse geometric augmentation to a tensor of images based on a probability and a transformation vector.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, potentially inverted by this function.
    - p (float): Probability of reversing the augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the potentially augmented image tensor and the adjusted (or inverted) transformation list.
    """
    if torch.rand(1) < p:
        source_to_target = euler2mat(
            camera_pose
        )  # Convert transformation vector to matrix
        target_to_source = torch.linalg.inv(
            source_to_target
        )  # Invert transformation matrix
        tr_inv = mat2euler(target_to_source)  # Convert inverted matrix back to vector
        return framestack, tr_inv

    return framestack, camera_pose


def standstill_augmentation(
    framestack: torch.Tensor, tr: list, p: float = 0.2
) -> tuple:
    """
    Optionally applies reverse geometric augmentation to a tensor of images based on a probability and a transformation vector.

    Parameters:
    - framestack (torch.Tensor): A tensor representing a stack of images.
    - tr (list): A list representing transformation parameters, potentially inverted by this function.
    - p (float): Probability of reversing the augmentation. Default is 0.2.

    Returns:
    - tuple[torch.Tensor, list]: A tuple containing the potentially augmented image tensor and the adjusted (or inverted) transformation list.
    """
    if torch.rand(1) < p:
        framestack[-1] = framestack[0]
        tr = torch.zeros_like(tr)
    return framestack, tr
