import torch
from torch_sgld import SGLD
import os
import numpy as np
import datetime
import time
import dill
import warnings
from gatetracker.utils.logger import get_logger
from typing import Any, Dict, Optional, List, Union, Tuple

logger = get_logger(__name__).set_context("OPTIMIZATION")
    

warnings.filterwarnings(
    "ignore", message="Your application has authenticated using end user credentials"
)


# Custom implementation of Adam optimizer, extending torch.optim.Adam.
class Adam(torch.optim.Adam):
    """
    Custom Adam optimizer with enhanced string representation.
    
    This class extends the standard PyTorch Adam optimizer to provide better
    string representation for logging and debugging purposes. It stores the
    initialization parameters for display in the string representation.
    
    Adam is an adaptive learning rate optimization algorithm that's been designed
    to handle sparse gradients on noisy problems. It computes individual adaptive
    learning rates for different parameters from estimates of first and second
    moments of the gradients.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the Adam optimizer with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list for the base class
            **kwargs: Arbitrary keyword arguments containing optimizer settings
        """
        super(Adam, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to use in the string representation

    def __str__(self) -> str:
        """
        Generate a string representation of the Adam optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


# Custom implementation of SGLD optimizer, extending a base SGLD class.
class SGLD(SGLD):
    """
    Custom Stochastic Gradient Langevin Dynamics (SGLD) optimizer.
    
    This class extends the base SGLD optimizer to provide better configuration
    management and string representation. It automatically sets momentum and
    temperature based on the provided parameters.
    
    SGLD is a Bayesian optimization method that combines stochastic gradient descent
    with Langevin dynamics to sample from the posterior distribution of model parameters.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the SGLD optimizer, setting the momentum and temperature based on kwargs.
        
        Note: Inherits from a base SGLD class, not shown in the provided code.

        Args:
            *args: Variable length argument list for the base class
            **kwargs: Arbitrary keyword arguments, with 'lr' used to set the temperature
        """
        super(SGLD, self).__init__(
            momentum=0.9, temperature=kwargs["lr"], *args, **kwargs
        )
        self.kwargs = kwargs
        self.kwargs["temperature"] = kwargs[
            "lr"
        ]  # Adjust temperature to match learning rate
        self.kwargs["momentum"] = 0.9  # Set momentum to a fixed value

    def __str__(self) -> str:
        """
        Generate a string representation of the SGLD optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


# Custom implementation of SGD optimizer, extending torch.optim.SGD.
class SGD(torch.optim.SGD):
    """
    Custom Stochastic Gradient Descent (SGD) optimizer with enhanced string representation.
    
    This class extends the standard PyTorch SGD optimizer to provide better
    string representation for logging and debugging purposes. It stores the
    initialization parameters for display in the string representation.
    
    SGD is a classic optimization algorithm that updates parameters in the opposite
    direction of the gradient of the objective function.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the SGD optimizer with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list for the base class
            **kwargs: Arbitrary keyword arguments containing optimizer settings
        """
        super(SGD, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to use in the string representation

    def __str__(self) -> str:
        """
        Generate a string representation of the SGD optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


class RMSprop(torch.optim.RMSprop):
    """
    Custom RMSprop optimizer with enhanced string representation.
    
    This custom implementation of RMSprop extends the torch.optim.RMSprop class,
    allowing for additional keyword arguments to be stored and used in the string representation.
    
    RMSprop is an adaptive learning rate method that adapts the learning rate for each
    parameter by dividing the learning rate for a weight by a running average of the
    magnitudes of recent gradients for that weight.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the RMSprop optimizer with the given arguments and keyword arguments.

        This custom implementation of RMSprop extends the torch.optim.RMSprop class,
        allowing for additional keyword arguments to be stored and used in the string representation.

        Args:
            *args: Variable length argument list for the base RMSprop class
            **kwargs: Arbitrary keyword arguments containing optimizer settings
        """
        super(RMSprop, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to be used in the __str__ method

    def __str__(self) -> str:
        """
        Generate a string representation of the RMSprop optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


class AdamW(torch.optim.AdamW):
    """
    Custom AdamW optimizer with enhanced string representation.
    
    This custom implementation of AdamW extends the torch.optim.AdamW class,
    adding functionality to store and utilize additional keyword arguments in its string representation.
    
    AdamW is a variant of Adam that implements weight decay correctly by decoupling
    weight decay from gradient-based updates.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AdamW optimizer with the given arguments and keyword arguments.

        This custom implementation of AdamW extends the torch.optim.AdamW class,
        adding functionality to store and utilize additional keyword arguments in its string representation.

        Args:
            *args: Variable length argument list for the base AdamW class
            **kwargs: Arbitrary keyword arguments containing optimizer settings
        """
        super(AdamW, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs for use in __str__ method

    def __str__(self) -> str:
        """
        Generate a string representation of the AdamW optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


class Adagrad(torch.optim.Adagrad):
    """
    Custom Adagrad optimizer with enhanced string representation.
    
    This custom implementation of Adagrad extends the torch.optim.Adagrad class,
    adding functionality to store and utilize additional keyword arguments in its string representation.
    
    Adagrad is an adaptive gradient-based optimization algorithm that adapts the
    learning rate to the parameters, performing larger updates for infrequent and
    smaller updates for frequent parameters.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the Adagrad optimizer with the given arguments and keyword arguments.

        This custom implementation of Adagrad extends the torch.optim.Adagrad class,
        adding functionality to store and utilize additional keyword arguments in its string representation.

        Args:
            *args: Variable length argument list for the base Adagrad class
            **kwargs: Arbitrary keyword arguments containing optimizer settings
        """
        super(Adagrad, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to be used in the __str__ method

    def __str__(self) -> str:
        """
        Generate a string representation of the Adagrad optimizer, including its configuration.

        Returns:
            A formatted string listing the optimizer's configuration
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k,'')} \n"
        modelstr += ")"
        return modelstr


def get_norms(params) -> Tuple[float, float]:
    """
    Calculate L1 and L2 norms of model parameters.

    This function computes the L1 (Manhattan) and L2 (Euclidean) norms of the
    provided parameters, which are useful for regularization and model analysis.

    Args:
        params: Iterable or generator of PyTorch parameter tensors to compute norms for

    Returns:
        Tuple containing:
            - L1 norm (float): Sum of absolute values
            - L2 norm (float): Square root of sum of squared values
    """
    params_list = list(params)
    l1_total = sum(p.detach().abs().sum().item() for p in params_list if p is not None)
    l2_total = sum((p.detach() ** 2).sum().item() for p in params_list if p is not None)
    l2_norm = l2_total ** 0.5
    return l1_total, l2_norm


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during training.
    
    This class implements early stopping with both soft and hard patience mechanisms.
    It monitors validation loss and saves checkpoints when the model improves.
    The early stopping can be configured with different patience levels and
    supports checkpoint saving to both local storage and cloud storage (GCS).
    
    Attributes:
        patience (int): Number of epochs to wait after last improvement
        hard_patience (int): Maximum number of epochs before forced stopping
        verbose (bool): Whether to print early stopping messages
        delta (float): Minimum change in validation loss to qualify as improvement
        runname (Optional[str]): Name for the current training run
        checkpointpath (str): Local path for saving checkpoints
        bucket_name (str): GCS bucket name for cloud storage
        best_score (Optional[float]): Best validation loss achieved
        counter (int): Number of epochs since last improvement
        hard_counter (int): Total number of epochs trained
        best_epoch (int): Epoch number of best validation loss
    """

    def __init__(
        self,
        patience: int = 3,
        hard_patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
        runname: Optional[str] = None,
        checkpointpath: str = "checkpoints",
        bucket_name: str = "alberto-bucket",  # New parameter for GCS bucket
    ) -> None:
        """
        Initialize the early stopping mechanism.

        Args:
            patience: Number of epochs to wait after last improvement before stopping
            hard_patience: Maximum number of epochs before forced stopping
            verbose: Whether to print early stopping messages
            delta: Minimum change in validation loss to qualify as improvement
            runname: Name for the current training run
            checkpointpath: Local path for saving checkpoints
            bucket_name: GCS bucket name for cloud storage
        """
        self.patience = patience
        self.hard_patience = hard_patience
        self.verbose = verbose
        self.delta = delta
        self.runname = runname
        self.checkpointpath = checkpointpath
        self.bucket_name = bucket_name

        self.best_score: Optional[float] = None
        self.counter = 0
        self.hard_counter = 0
        self.best_epoch = 0

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpointpath, exist_ok=True)

    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss
            model: PyTorch model to save if improved
            epoch: Current epoch number
        
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        self.hard_counter = 0

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model, epoch)
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model, epoch)
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )

        # Check if we should stop
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            return True

        if self.hard_counter >= self.hard_patience:
            logger.info(f"Hard patience reached after {epoch} epochs")
            return True

        return False

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module, epoch: int) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            val_loss: Validation loss at the time of saving
            model: PyTorch model to save
            epoch: Current epoch number
        """
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model ..."
            )

        # Create checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "best_score": self.best_score,
            "counter": self.counter,
            "hard_counter": self.hard_counter,
            "best_epoch": self.best_epoch,
        }

        # Save locally
        local_path = os.path.join(
            self.checkpointpath, f"{self.runname}_checkpoint.pth"
        )
        torch.save(checkpoint, local_path)
        logger.info(f"Checkpoint saved locally: {local_path}")

        # Save to GCS if bucket is specified
        if self.bucket_name:
            try:
                from google.cloud import storage    

                client = storage.Client()
                bucket = client.bucket(self.bucket_name)
                blob = bucket.blob(f"checkpoints/{self.runname}_checkpoint.pth")
                blob.upload_from_filename(local_path)
                logger.info(f"Checkpoint uploaded to GCS: {blob.name}")
            except ImportError:
                logger.warning("Google Cloud Storage not available. Skipping GCS upload.")
            except Exception as e:
                pass

    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str) -> None:
        """
        Load model from checkpoint file.

        Args:
            model: PyTorch model to load state into
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore early stopping state
            self.best_score = checkpoint.get("best_score")
            self.counter = checkpoint.get("counter", 0)
            self.hard_counter = checkpoint.get("hard_counter", 0)
            self.best_epoch = checkpoint.get("best_epoch", 0)
            
            logger.info(
                f"Checkpoint loaded from {checkpoint_path}. "
                f"Epoch: {checkpoint['epoch']}, "
                f"Validation Loss: {checkpoint['val_loss']:.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
