import torch
from torch.utils.data import Dataset
from typing import Tuple
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D # type: ignore

class LinearlySeparableDataset(Dataset):
    
    def __init__(self, num_points: int, margin: float, 
                 coord_limit: int, device: str = "cpu", seed: int | None = None):
        """
        A dataset loader for generating linearly separable 3D data.
        
        The plane separating the two classes is defined as:
            z = x + y + margin * (2 * label - 1).
        
        Expected tensor shapes:
            - x, y, z: each of shape (num_points,)
            - self.data: tensor of shape (num_points, 3) where each row represents a point [x, y, z]
            - self.labels: tensor of shape (num_points,)
        
        Args:
            num_points (int): Number of data points to generate.
            margin (float): Margin that shifts points from the separating plane.
            coord_limit (int): Coordinate bounds for x and y.
            device (str): Device for tensor storage (default: "cpu").
            seed (int, optional): Random seed for reproducibility. If None, randomness is not fixed.
        """
        self.coord_limit = coord_limit
        self.device = torch.device(device)

        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.empty(num_points, dtype=torch.float32, device=self.device)
        y = torch.empty(num_points, dtype=torch.float32, device=self.device)
        z = torch.empty(num_points, dtype=torch.float32, device=self.device)

        # Generate random x, y coordinates
        x = x.uniform_(-coord_limit, coord_limit)
        y = y.uniform_(-coord_limit, coord_limit)

        # Randomly assign class labels (0 or 1)
        labels_shape = (num_points,)
        labels = torch.randint(0, 2, labels_shape, dtype=torch.int64, device=self.device)

        # Compute z-coordinates based on the separating plane
        # 'sign' has shape (num_points,) where each entry is either -1 or 1
        sign = 2 * labels - 1 
        z = x + y + (margin * sign)

        # Store labels
        self.labels = labels

        # Stack x, y, z into a single tensor of shape (num_points, 3)
        self.data = torch.stack((x, y, z), dim=1)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the data and label for the given index.
        
        Returns:
            A tuple where:
            - The first element is a tensor of shape (3,) corresponding to [x, y, z] for one point.
            - The second element is a scalar tensor representing the label.
        """
        return self.data[idx].contiguous(), self.labels[idx]
    
    @property
    def data_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """Returns a DataLoader instance for the dataset."""
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def plot(self, predictions: torch.Tensor | None = None, elevation: int = 5
             ) -> Tuple[Figure, Axes3D]:
        """
        Plots the 3D dataset with a separating plane.
        
        Run with the following to generate a plot:
            import matplotlib.pyplot as plt
            plt.show()

        Args:
            predictions (torch.Tensor): Predictions for each point in the dataset. 
                Expected shape: (num_points,), with values 0 or 1.   
            elevation (int): Elevation angle for the plot (default: 5).
        """
        
        fig = plt.figure(figsize=(8, 6))
        ax: Axes3D = fig.add_subplot(111, projection='3d')

        # Extract data and labels
        x, y, z = self.data.t().cpu()
        labels = self.labels.cpu()
        if predictions is not None:
            labels = predictions.cpu()

        # Scatter plot for both classes
        ax.scatter(x[labels == 0], y[labels == 0], z[labels == 0], c='blue', label='Class 0', alpha=0.6)
        ax.scatter(x[labels == 1], y[labels == 1], z[labels == 1], c='green', label='Class 1', alpha=0.6)

        # Determine the coordinate limits dynamically
        coord_limit = max(x.abs().max(), y.abs().max()).item()

        # Generate grid for the separating plane using PyTorch
        grid_x = torch.linspace(-coord_limit, coord_limit, 20)
        grid_y = torch.linspace(-coord_limit, coord_limit, 20)
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="xy")

        # Compute the separating plane z = x + y
        zz = xx + yy

        # Convert tensors to NumPy for Matplotlib compatibility
        ax.plot_surface(xx.numpy(), yy.numpy(), zz.numpy(), alpha=0.3, color='gray')

        # Labels and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Linearly Separable Data")
        ax.legend()
        ax.view_init(elev=elevation, azim=-30)  # Adjusted view angle
    
        self.axis = ax
        self.fig = fig

        return fig, ax

    def plot_point(self, coords: torch.Tensor, color: str, edgecolor: str) -> None:
        """
        Plots a new point on the 3D plot with the separating plane.
        
        Parameters:
            coords (torch.Tensor): The coordinates of the new point.
            color (str): The color for the new point.
            edgecolor (str): The edge color for the new point.
        """

        if coords.shape != (3,):
            raise ValueError(
                f"Expected coords to be a 1D tensor with shape (3,), but got shape {coords.shape}")

        # Unpack the coordinates
        x, y, z = [coord.item() for coord in coords]

        # Add the new point to the figure with a green outline
        self.axis.scatter3D(x, y, z, s=100, 
                            color=color, edgecolors=edgecolor, linewidth=1.5)
        
