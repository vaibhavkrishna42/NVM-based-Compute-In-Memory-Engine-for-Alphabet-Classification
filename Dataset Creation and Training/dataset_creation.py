import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch

# Provided ideal letters
ideal_letters = {
    "A": [
        np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1]]),
        np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 1]]),
        np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1]]),
        np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [0, 0, 0, 1]]),
        np.array([[0, 1, 1, 0],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 0]]),
        np.array([[1, 1, 1, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1],
                  [1, 0, 0, 0]]),
        np.array([[0, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 1],
                  [0, 1, 0, 1]]),
        np.array([[0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [1, 1, 1, 0],
                  [1, 0, 1, 0]]),
        np.array([[0, 1, 1, 1],
                  [0, 1, 0, 1],
                  [0, 1, 1, 1],
                  [0, 1, 0, 1]]),
        np.array([[1, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 1, 0],
                  [1, 0, 1, 0]])
    ],
    "T": [
        np.array([[1, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]]),
        np.array([[1, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]),
        np.array([[1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]]),
        np.array([[0, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]),
        np.array([[1, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[1, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0]]),
        np.array([[1, 1, 1, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0]]),
        np.array([[0, 1, 1, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])
    ],
    "V": [
        np.array([[1, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 1, 0, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]]),
        np.array([[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]]),
        np.array([[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 1, 1]]),
        np.array([[1, 0, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0],
                  [0, 0, 0, 0]]),
        np.array([[1, 0, 1, 0],
                  [1, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0]]),
        np.array([[0, 1, 0, 1],
                  [0, 1, 0, 1],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]])
    ],
    "X": [
        np.array([[1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 0]]),
        np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0]]),
        np.array([[0, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 1]]),
        np.array([[1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 1]]),
        np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 0, 0]]),
        np.array([[0, 0, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0]]),
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 1]]),
        np.array([[0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 1, 0, 1]]),
        np.array([[1, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0]])
    ]
}

# Function to add noise to a grid without displacing existing 1s
def add_noise_without_displacing_ones(grid, noise_pixels):
    """
    Add noise to a grid by flipping a specified number of 0s to 1s, without affecting existing 1s.
    """
    noisy_grid = grid.copy()
    # zero_positions = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == 0]
    zero_positions = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    # Shuffle zero positions to select random ones
    random.shuffle(zero_positions)
    # Flip the specified number of zeros to ones
    for x, y in zero_positions[:noise_pixels]:
        noisy_grid[x, y] = 1
    return noisy_grid

# Function to generate the dataset
def generate_dataset(ideal_letters, total_samples, noise_pixels):
    """
    Generate a dataset by adding noise to ideal letters without displacing existing 1s.
    """
    images, labels = [], []
    samples_per_letter = total_samples // len(ideal_letters)  # Equal split across letters
    for label, (letter, configs) in enumerate(ideal_letters.items()):
        for _ in range(samples_per_letter):
            config = random.choice(configs)  # Randomly select a configuration
            noisy_grid = add_noise_without_displacing_ones(config, noise_pixels)  # Add noise
            images.append(noisy_grid.flatten())  # Flatten for dataset
            labels.append(label)
    return np.array(images), np.array(labels)

# Generate training, validation, and testing datasets with specified sizes and noise levels
training_samples = 500
validation_samples = 60
testing_samples = 100

X_train, y_train = generate_dataset(ideal_letters, training_samples, noise_pixels=1)  # Flip 1 pixel for training
X_val, y_val = generate_dataset(ideal_letters, validation_samples, noise_pixels=1)    # Flip 1 pixel for validation
X_test, y_test = generate_dataset(ideal_letters, testing_samples, noise_pixels=2)     # Flip 2 pixels for testing

# Dataset summary
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Visualize some samples with labels as letters
label_to_letter = {0: "A", 1: "T", 2: "V", 3: "X"}

def visualize_samples(images, labels, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        idx = random.randint(0, len(images) - 1)
        ax = axes[i]
        
        # Reshape image to 4x4 for grid display
        ax.imshow(images[idx].reshape(4, 4), cmap='binary', interpolation='nearest')
        
        ax.set_title(f"Label: {label_to_letter[labels[idx]]}")
        
        # Ensure grid fits the pixel layout
        ax.set_xticks(np.arange(-0.5, 4, 1))  # Add ticks for each cell's boundary
        ax.set_yticks(np.arange(-0.5, 4, 1))  # Add ticks for each cell's boundary
        ax.grid(True, which='both', axis='both', color='black', linestyle='-', linewidth=1)

         # Remove axis labels and ticks
        ax.set_xticklabels([])  # Remove x-axis labels
        ax.set_yticklabels([])  # Remove y-axis labels
        ax.tick_params(which='both', size=0)  # Remove tick marks

        # Set the aspect ratio to make sure each pixel is square and fits well
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.5, 3.5)  # Ensures grid fits within image
        ax.set_ylim(3.5, -0.5)  # Ensures proper orientation (flipping y-axis)

        ax.tick_params(which='both', size=0)  # Remove tick marks to clean up the plot

    plt.show()

print("Sample Training Images:")
visualize_samples(X_train, y_train)

print("Sample Testing Images:")
visualize_samples(X_test, y_test)

# Convert numpy arrays to PyTorch tensors and save
torch.save((torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), "training_data_500.pt")
torch.save((torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), "validation_data_60.pt")
torch.save((torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), "testing_data_100.pt")

print("Datasets saved as PyTorch tensors.")
