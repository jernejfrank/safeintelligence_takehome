import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class SimpleNN(nn.Module):
    def __init__(self):
        """
        A simple neural network with 1 hidden layer.

        The network has 3 input features, 10 hidden neurons, and 1 output neuron.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # Input layer (3 features) -> Hidden layer (10 neurons)
        self.fc2 = nn.Linear(10, 1)  # Hidden layer (10 neurons) -> Output layer (1 output)
        self.relu = nn.ReLU() # Non-linear activation for hidden layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass: Linear -> ReLU -> Linear.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3).
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, 1).
        """
        x = self.relu(self.fc1(x)) # Apply ReLU activation on hidden layer
        x = self.fc2(x)  # No activation on output layer
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass and return the predicted classes (0 or 1).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3).

        Returns:
            torch.Tensor: Predicted class labels of shape (batch_size, 1).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            # Threshold at 0.0 for logits to get class predictions
            predictions = (logits > 0.0).long()
        return predictions
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor, criterion, optimizer, device: torch.device):
        """
        Performs a single training step (forward, backward, optimize).

        Args:
            inputs (torch.Tensor): Input batch tensor.
            labels (torch.Tensor): Corresponding labels tensor.
            criterion: Loss function.
            optimizer: Optimizer.
            device: Device to run computations on.

        Returns:
            float: The loss for the batch.
            correct: Number of correctly predicted samples in the batch.
            total: Total number of samples in the batch.
        """

        # Move data and labels to the appropriate device (CPU or GPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients (clean the optimizer)
        optimizer.zero_grad()  
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute the loss
        # .squeeze() to remove unnecessary dimensions
        loss = criterion(outputs.squeeze(), labels.float())  
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Calculate accuracy
        predicted = outputs > 0.0 # Threshold at 0.0 for logits
        correct = (predicted.squeeze() == labels).sum().item()
        total = labels.size(0)

        return loss.item(), correct, total
    
    def train_model(self, 
                    dataloader: DataLoader,
                    learning_rate: float = 0.001, num_epochs: int = 100,
                    device: torch.device = torch.device("cpu")):
        """
        Trains the neural network on the given dataset.

        Args:
            dataloader: DataLoader instance for the dataset.
            learning_rate: Learning rate for the optimizer.
            num_epochs: Number of epochs to train the model.
            device: Device to use for training (e.g., CPU or GPU).
        """
        # Move model to the specified device
        self.to(device)

        # Binary Cross-Entropy Loss
        criterion = nn.BCEWithLogitsLoss()  
        
        # Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  

        # Training loop
        for epoch in range(num_epochs):

            # Set the model to training mode
            self.train()  
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in dataloader:
                
                batch_loss, batch_correct, batch_total = self.train_step(
                    inputs, labels, criterion, optimizer, device)
                
                epoch_loss += batch_loss
                correct += batch_correct
                total += batch_total

            # Calculate accuracy
            accuracy = 100 * correct / total

            # Print loss and accuracy every few epochs
            epoch_str = f"[{epoch+1}/{num_epochs}]"
            loss_str = f"Loss: {epoch_loss/len(dataloader):6.4f}"
            acc_str = f"Accuracy: {accuracy:6.2f}%"
            print(f"Epoch: {epoch_str:<2} \t Loss: {loss_str:<2} \t {acc_str}")