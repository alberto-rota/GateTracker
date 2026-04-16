import torch
import numpy as np


def dup_indexes(arr):
    unique, counts = np.unique(arr, return_counts=True)
    duplicates = unique[counts > 1]
    return [np.where(arr == dup)[0].tolist() for dup in duplicates]


def backwardpass_grad_check(model, loss, verbosity: int = 0) -> bool:
    """
    Checks if the gradients of the model's parameters are finite after a backward pass.

    Args:
        loss (torch.Tensor): The loss tensor.

    Returns:
        bool: True if all gradients are finite, False otherwise.
    """
    assert verbosity in [0, 1, 2]
    gradsok = True
    allweights, allgrads, allshapes, allnames = [], [], [], []
    loss.backward()
    dummy_opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dummy_opt.step()

    frozen = []
    reqgrad_ok = []
    reqgrad_gnone = []
    reqgrad_gzero = []
    total_num_layers = 0
    healthy_layers = 0
    for name, param in model.named_parameters():
        total_num_layers += 1
        if param.requires_grad:
            if verbosity == 2:
                print(f"> {name} \n[green]REQUIRES GRAD[/]")
            if param.grad == None:
                if verbosity == 1:
                    print(f"[orange3]WARNING: {name} --> Grad None[/]")
                reqgrad_gnone.append(name)
                gradsok = False
            elif param.grad.norm().item() == 0:
                if verbosity == 1:
                    print(
                        f"Weight = {param.norm()} / Grad = [blue]{param.grad.norm()}[/blue] [red] --> ZERO[/]"
                    )
                reqgrad_gzero.append(name)
                gradsok = False
            else:
                if verbosity == 2:
                    print(
                        f"Weight = {param.norm()} / Grad = [blue]{param.grad.norm()}[/blue] [green] --> OK[/]"
                    )
                reqgrad_ok.append(name)
                healthy_layers += 1
                allweights.append(param.norm().cpu().detach().numpy())
                allgrads.append(param.grad.norm().cpu().detach().numpy())
                allshapes.append(tuple(param.shape))
                allnames.append(name)
        else:
            if verbosity == 2:
                print(f"> {name} \n[cyan]DOESN'T REQUIRE GRAD --> Frozen Layer[/] ")
            frozen.append(name)
            healthy_layers += 1

    same_weights = dup_indexes(np.array(allweights))
    for sw in same_weights:
        if verbosity >= 1:
            print(
                f"[orange3]WARNING: {allnames[sw[0]]}[A] and {allnames[sw[1]]}[B] have the same weights[/]",
                "\n",
                f"[A]Weight = {allweights[sw[0]]} / Grad = {allgrads[sw[0]]} / Shape = {allshapes[sw[0]]} \n",
                f"[B]Weight = {allweights[sw[1]]} / Grad = {allgrads[sw[1]]} / Shape = {allshapes[sw[1]]} \n",
            )
    if verbosity >= 1:
        if len(reqgrad_ok) > 0:
            print(
                f"> [green]{len(reqgrad_ok)} Layers requiring grad with valid gradients[/]"
            )
        if len(frozen) > 0:
            print(f"> [cyan]{len(frozen)} Layers not requiring grads [/]")
        if len(reqgrad_gzero) > 0:
            print(
                f"> [orange3]{len(reqgrad_gzero)} Layers requiring grad with zero gradient[/]"
            )
            if verbosity == 2:
                print(reqgrad_gzero)
        if len(reqgrad_gnone) > 0:
            print(
                f"> [red]{len(reqgrad_gnone)} Layers requiring grad with no gradient[/]"
            )
            if verbosity == 2:
                print(reqgrad_gnone)
        if len(same_weights) > 0:
            print(
                f"> [yellow]{len(same_weights)} Pairs of layers with the same weights[/]"
            )
            if verbosity == 2:
                print(same_weights)
    return gradsok, healthy_layers / total_num_layers


def forwardpass_shape_check(
    model,
    framestack: torch.Tensor,
    Ts2t: torch.Tensor,
    depthmap_pred: torch.Tensor,
    warped: torch.Tensor,
    Ts2t_pred: tuple,
    verbosity=1,
) -> bool:
    """
    Checks if the shapes of the input tensors are compatible with the forward pass of the model.

    Args:
        source (torch.Tensor): The source image tensor.
        depthmap_pred (torch.Tensor): The predicted depth map tensor.
        warped (torch.Tensor): The warped image tensor.
        target (torch.Tensor): The target image tensor.

    Returns:
        bool: True if the shapes are compatible, False otherwise.
    """

    framestack_shape = framestack.shape
    Ts2t_shape = Ts2t.shape if Ts2t is not None else None
    depthmap_pred_shape = depthmap_pred.shape
    Ts2t_pred_shape = Ts2t_pred.shape
    warped_shape = warped.shape
    
    passed = True
    if Ts2t is not None:
        if (
            not framestack_shape[0] == Ts2t_shape[0] == Ts2t_pred_shape[0] == depthmap_pred_shape[0] == warped_shape[0]
        ):  # Batch Size
            passed = False
            if verbosity == 1:
                print("> Incompatible batch size")
    else:
        if (
            not framestack_shape[0] == Ts2t_pred_shape[0] == depthmap_pred_shape[0] == warped_shape[0]
        ):  # Batch Size
            passed = False
            if verbosity == 1:
                print("> Incompatible batch size")
    if not (
        framestack_shape[-1] == depthmap_pred_shape[-1] == warped_shape[-1]
        and framestack_shape[-2] == depthmap_pred_shape[-2] == warped_shape[-2]
    ):
        passed = False
        if verbosity == 1:
            print("> Incompatible spatial dimensions")
    if Ts2t is not None and not (Ts2t_shape == Ts2t_pred_shape):
        passed = False
        if verbosity == 1:
            print("> Incompatible odometry shapes")
    return passed
