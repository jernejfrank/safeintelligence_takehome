from provided.data import LinearlySeparableDataset
from provided.network import SimpleNN
from provided.pgd import PGDAttack
from provided.utils import print_header

import matplotlib.pyplot as plt
from typing import Literal


if __name__ == "__main__":
    
    # Generate a linearly separable dataset
    dataset = LinearlySeparableDataset(num_points=100, margin=2, coord_limit=5)

    # Train a model on the dataset
    model = SimpleNN()
    print_header("Training the model")
    model.train_model(dataset.data_loader, learning_rate=0.01, num_epochs=10)  

    # Plot the model's predictions
    predictions = model.predict(dataset.data).detach().squeeze()
    dataset.plot(predictions, 5)

    # Sample correct predictions
    correct_indices = (predictions == dataset.labels.squeeze()).nonzero(as_tuple=True)[0]

    # Iterate over the correct predictions
    print_header("Generation Adversarial Examples")
    for idx in correct_indices:

        correct_item = dataset.__getitem__(int(idx.item()))
        correct_input = correct_item[0].squeeze()
        correct_label: Literal[0, 1] = 1 if correct_item[1].item() == 1 else 0

        # PGD Attack
        pgd = PGDAttack(model, epsilon=0.3, steps=5)
        perturbed_point = pgd.perturb(correct_input.unsqueeze(0), correct_label)
        
        # If an adversarial example was found 
        # plot the original and perturbed points
        if perturbed_point is not None:

            perturbed_input = perturbed_point.squeeze()
            pert_output = model.predict(perturbed_input)
            pert_prediction: Literal[0, 1] = 1 if pert_output.item() == 1 else 0

            border = "green" if correct_label == 1 else "blue"

            # Plot the correct point
            dataset.plot_point(correct_input, "white", border)
            
            # Plot the perturbed point
            dataset.plot_point(perturbed_input, "red", border)

    # Show the modified figure
    print_header("Displaying the plot")
    plt.show()