# -------------------------------------------------------------------------------------------------#

""" Copyright (c) 2024 Asensus Surgical """

""" Code Developed by: Alberto Rota """
""" Supervision: Uriya Levy, Gal Weizman, Stefano Pomati """

# -------------------------------------------------------------------------------------------------#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from rich.tree import Tree
from rich import print as rprint

from gatetracker.losses.descriptor import InfoNCELoss
from gatetracker.losses.geometric import EpipolarLoss, WeightedSmoothL1Loss


@dataclass
class LossComponent:
    """
    Dataclass to store information about a loss component and its required parameters.
    
    This class encapsulates all the information needed for a single loss component
    in a weighted combination loss, including the loss function, weight, required
    parameters, and optional decay rate.
    """

    name: str
    loss_fn: nn.Module
    weight: float
    required_params: Set[str]
    decay_rate: Optional[float] = None

    @property
    def current_weight(self) -> float:
        """Get the current weight after decay."""
        return self.weight


class WeightedCombinationLoss(nn.Module):
    """
    A loss module that combines multiple loss functions with weights.
    
    This class provides a flexible framework for combining multiple loss functions
    with configurable weights. It supports weight decay over time and provides
    detailed breakdown of individual loss components for monitoring and debugging.
    
    The loss module can handle nested loss functions and provides comprehensive
    logging and visualization capabilities.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the WeightedCombinationLoss module.
        
        Args:
            components: List of tuples containing (name, loss_fn, weight, required_params)
                       for each loss component
            decay_config: Optional dictionary mapping component names to decay rates
        """
        super(WeightedCombinationLoss, self).__init__()

        # Normalize weights to sum to 1
        total_weight = sum(weight for _, _, weight, _ in components)
        decay_config = decay_config or {}

        # Create loss components
        self.components = [
            LossComponent(
                name=name,
                loss_fn=loss_fn,
                weight=weight,  # / total_weight,
                required_params=required_params,
                decay_rate=decay_config.get(name),
            )
            for name, loss_fn, weight, required_params in components
        ]

        # Store all required parameters
        self.all_required_params = set().union(
            *(comp.required_params for comp in self.components)
        )

        self.step_count = 0

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """
        Compute the combined loss from all components.
        
        Args:
            **kwargs: Keyword arguments containing all required parameters for the loss components
            
        Returns:
            Combined weighted loss value
            
        Raises:
            ValueError: If any required parameters are missing
        """
        # Validate that all required parameters are provided
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        total_loss = 0.0

        for component in self.components:
            # Extract only the arguments needed for this specific loss function
            fn_args = {k: kwargs[k] for k in component.required_params}

            loss = component.loss_fn(**fn_args)
            total_loss += loss * component.current_weight

        return total_loss

    def get_dict(self, prepend_tonames: str = "", **kwargs: Any) -> Dict[str, float]:
        """
        Get detailed breakdown of all loss components.
        
        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary
            **kwargs: Keyword arguments containing all required parameters
            
        Returns:
            Dictionary mapping loss component names to their individual loss values
            
        Raises:
            ValueError: If any required parameters are missing
        """
        # Validate parameters first
        missing_params = self.all_required_params - set(kwargs.keys())
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        results = {}
        total_loss = 0.0

        for component in self.components:
            component_loss = 0.0
            currentcomponentname = f"{prepend_tonames}{component.name}"
            fn_args = {k: kwargs[k] for k in component.required_params}

            # Handle nested loss functions that have their own get_dict
            if hasattr(component.loss_fn, "get_dict"):
                sub_losses = component.loss_fn.get_dict(
                    prepend_tonames=f"{prepend_tonames}{component.name}/", **fn_args
                )
                results.update(sub_losses)
                for c in component.loss_fn.components:
                    fn_args = {k: kwargs[k] for k in c.required_params}
                    loss = c.loss_fn(**fn_args)
                    component_loss += loss * c.current_weight
                results[f"{prepend_tonames}{component.name}"] = component_loss.item()

            else:
                loss = component.loss_fn(**fn_args)
                loss_value = loss.item()
                results[f"{prepend_tonames}{component.name}"] = loss_value

        return results

    def get_weights(self, prepend_tonames: str = "") -> Dict[str, float]:
        """
        Recursively get a dictionary of weights for all components.

        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary

        Returns:
            Dictionary mapping loss component names to their weights
        """
        weights = {}
        for component in self.components:
            component_name = f"{prepend_tonames}{component.name}"
            if isinstance(component.loss_fn, WeightedCombinationLoss):
                # Recurse into nested WeightedCombinationLoss
                sub_weights = component.loss_fn.get_weights(
                    prepend_tonames=f"{component_name}_"
                )
                weights.update(sub_weights)
            weights[component_name] = component.current_weight
        return weights

    def step(self) -> None:
        """
        Update step count and decay weights if configured, recursively stepping child components.
        
        This method should be called after each training step to update the step count
        and apply any configured weight decay to the loss components.
        """
        self.step_count += 1
        for component in self.components:
            # Decay the weight if a decay rate is specified
            if component.decay_rate:
                component.weight *= np.exp(-component.decay_rate * self.step_count)
            # Recursively call step on nested WeightedCombinationLoss
            if hasattr(component.loss_fn, "step"):
                component.loss_fn.step()

    def __str__(self) -> str:
        """Create a hierarchical string representation of the loss structure."""
        components_str = []
        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decaying @ {comp.decay_rate:.2e})"

            loss_str = str(comp.loss_fn).replace("\n", "\n    ")

            params_str = f"params={sorted(comp.required_params)}"
            components_str.append(
                f"    {weight_str} * {loss_str} [{comp.name}]\n    └─ {params_str}"
            )

        return f"{self.__class__.__name__}(\n" + "\n".join(components_str) + "\n)"

    def rich_print(self, parent_tree: Optional[Tree] = None) -> None:
        """
        Print a rich tree visualization of the loss structure.
        
        Args:
            parent_tree: Optional parent tree node for nested visualization
        """
        if parent_tree is None:
            tree = Tree(f"{self.__class__.__name__}")
        else:
            tree = parent_tree.add(f"{self.__class__.__name__}")

        for comp in self.components:
            weight_str = f"{comp.current_weight:.3f}"
            if comp.decay_rate:
                weight_str += f" (decay: {comp.decay_rate:.2e})"

            branch = tree.add(
                f"[blue]{comp.name}[/blue]([cyan]{','.join(comp.required_params)}[/cyan]) × [yellow]{weight_str}[/yellow]"
            )

            if hasattr(comp.loss_fn, "rich_print"):
                comp.loss_fn.rich_print(branch)
            else:
                loss_str = str(comp.loss_fn).replace("\n", "\n    ")

        if parent_tree is None:
            rprint(tree)


class TrackingPositionLoss(nn.Module):
    r"""
    Visibility-masked Huber loss on predicted vs GT point positions.

    .. math::
        \mathcal{L}_{\text{pos}} = \frac{1}{\sum v_i^t}
        \sum_{i,t: v_i^t = 1} \text{Huber}(\hat{p}_i^t - p_i^t)

    Operates on tensors ``[B, Q, T, 2]`` where visibility ``[B, Q, T]`` serves
    as the weight mask.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prediction: [B, Q, T, 2] predicted positions.
            target:     [B, Q, T, 2] GT positions.
            visibility: [B, Q, T]    GT visibility mask (bool or float).

        Returns:
            Scalar loss.
        """
        if prediction.numel() == 0:
            return prediction.sum() * 0.0

        vis = visibility.float()  # [B, Q, T]
        per_point = F.smooth_l1_loss(
            prediction, target, reduction="none", beta=self.beta,
        ).sum(dim=-1)  # [B, Q, T]

        normalizer = vis.sum().clamp_min(self.eps)
        return (per_point * vis).sum() / normalizer


class VisibilityLoss(nn.Module):
    r"""
    Binary cross-entropy on predicted visibility logits vs GT visibility.

    .. math::
        \mathcal{L}_{\text{vis}} = \text{BCE}(\sigma(\hat{v}_i^t),\; v_i^t)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, Q, T] raw visibility logits.
            target: [B, Q, T] GT visibility (bool or float 0/1).

        Returns:
            Scalar BCE loss.
        """
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            logits.float(), target.float(), reduction="mean",
        )


class GateTrackerLoss(WeightedCombinationLoss):
    """
    A flexible loss module that combines multiple loss functions with weights.
    
    This is the main loss class used in the GateTracker framework. It extends the
    WeightedCombinationLoss with additional functionality specific to the GateTracker
    training pipeline, including automatic total loss computation and enhanced
    logging capabilities.
    
    Supports weight decay and detailed loss reporting for comprehensive training
    monitoring and debugging.
    """

    def __init__(
        self,
        components: List[Tuple[str, nn.Module, float, Set[str]]],
        decay_config: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the GateTrackerLoss module.
        
        Args:
            components: List of tuples containing (name, loss_fn, weight, required_params)
                       for each loss component
            decay_config: Optional dictionary mapping component names to decay rates
        """
        super(GateTrackerLoss, self).__init__(components, decay_config)

    def get_dict(self, prepend_tonames: str = "", **kwargs: Any) -> Dict[str, float]:
        """
        Get detailed breakdown of all loss components including total loss.
        
        Args:
            prepend_tonames: String to prepend to all loss names in the output dictionary
            **kwargs: Keyword arguments containing all required parameters
            
        Returns:
            Dictionary mapping loss component names to their individual loss values,
            plus the total combined loss
        """
        results = super().get_dict(prepend_tonames, **kwargs)
        results[f"{prepend_tonames}Loss"] = super().forward(**kwargs).item()
        return results

    def __str__(self) -> str:
        """Return string representation of the loss structure."""
        return super().__str__()
