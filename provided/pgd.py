import torch.nn as nn
import torch
from typing import Literal

class PGDAttack():
    """
    Projected Gradient Descent Attack.
    """

    def __init__(self, model: nn.Module,
                 epsilon: float = 0.1, steps: int = 100):
        """
        Initialize the PGD attack.

        Args:
            model (nn.Module): Model to attack.
            epsilon (float): Maximum perturbation allowed.
            steps (int): Number of attack iterations.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = epsilon / steps
        self.steps = steps
        self.loss_function = nn.BCEWithLogitsLoss()

    def perturb(self, input: torch.Tensor, label: Literal[0, 1]) -> torch.Tensor | None:
        """
        Given a new data point (x, y, z) and its label, return the adversarial
        counterpart of the data point.

        Args:
            input (torch.Tensor): Input coordinates of the data point.
                Expected shape: (1, 3).
            label (Literal[0, 1]): Label of the data point.

        Returns:
            torch.Tensor | None: Input coordinates of the adversarial example.
        """
        
        # Set the model to evaluation mode 
        # (it doesn't change during the attack)
        self.model.eval()

        # Get the model's device
        device = next(self.model.parameters()).device

        # Ensure input is on the same device as the model
        x = input.to(device)
        x_adv = x.clone()

        # Define the lower and upper bounds of the input perturbation region
        lower_bounds = x - self.epsilon
        upper_bounds = x + self.epsilon

        # Define the target tensor (binary classification)
        target = torch.tensor([label], dtype=torch.float32).to(device)

        # Take up to self.steps of size self.alpha in the direction that increases
        # the loss and check if the perturbed point is a counterexample.
        for step in range(self.steps):
            x_adv = self.take_perturb_step(x_adv, target, lower_bounds, upper_bounds)
            if self.is_a_counterexample(x_adv, label, step):
                return x_adv

        return None

    def take_perturb_step(self, x_adv: torch.Tensor,
                          target: torch.Tensor,
                          lower_bound: torch.Tensor,
                          upper_bound: torch.Tensor) -> torch.Tensor:
        """
        Updates the potential counter example point by taking a step of size alpha in
        the direction of the sign of the gradients. The updated point is clamped to be
        within the perturbation region.

        Args:
            x_adv (torch.Tensor):   The coordinates of a potential counterexample.
                                    Expected shape: (1, 3).
            target (torch.Tensor):  Tensor representing the label of the data point,
                                    used by the loss function.  Expected shape: (1).
            lower_bound (torch.Tensor):  Lower bound of the initial perturbation
                                    region. Expected shape: (1, 3).
            upper_bound (torch.Tensor):  Upper bound of the initial perturbation
                                    region. Expected shape: (1, 3).

        Returns:
            torch.Tensor:       Coordinates of the adversarial counterexample, having
                                taken a step in the direction that increases the loss.
        """

        # Take a step of size alpha in the direction of the sign of the gradients
        x_gradients = self.calculate_gradients(x_adv, target)
        assert x_gradients is not None
        x_adv = x_adv + self.alpha * x_gradients.sign()

        # Clip within bounds
        x_adv = torch.clamp(x_adv, lower_bound, upper_bound)
        return x_adv.detach()

    def is_a_counterexample(self, x_adv: torch.Tensor,
                            label: Literal[0, 1],
                            step: int) -> bool:
        """
        Returns True if the model misclassifies the perturbed input.

        Args:
            x_adv (torch.Tensor):   The coordinates of a potential counterexample.
                                    Expected shape: (1, 3).
            label (Literal[0, 1]): Label of the data point.

        Returns:
            bool:                   Whether the x_adv point is an adversarial
                                    counterexample that has been mislabelled.
        """
        with torch.no_grad():
            predicted = torch.sigmoid(self.model(x_adv)).round()
            misclassified = predicted.item() != label
            if misclassified:
                epsilon_str = f"Epsilon: {(self.alpha * (step + 1)):6.4f}"
                input_str = f"Adversarial Input: {x_adv.squeeze().cpu().numpy()}"
                print(f"{epsilon_str} → {input_str}")
            return misclassified

    def calculate_gradients(self, x_adv: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradients of the loss function of the model outputs to the
        input x_adv.

        Args:
            x_adv (torch.Tensor):   The coordinates of a potential counterexample.
                                    Expected shape: (1, 3).
            target (torch.Tensor):  Tensor representing the label of the data point,
                                    used by the loss function.  Expected shape: (1).

        Returns:
            torch.Tensor | None:     Input coordinates of the adversarial example.
        """

        # Zero gradients and enable gradients for attack
        x_adv.requires_grad = True
        self.model.zero_grad()

        # Forward pass with gradient calculation
        outputs = self.model(x_adv)

        # Compute loss (Binary Cross Entropy with Logits) and back-propagate
        loss = self.loss_function(outputs.view(-1), target.float())
        loss.backward()

        # Ensure gradients are not None
        if x_adv.grad is None:
            raise RuntimeError(
                "Gradients were not computed. Ensure x_adv is a tensor and requires_grad=True.")

        return x_adv.grad