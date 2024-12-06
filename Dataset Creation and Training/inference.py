import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the test dataset
X_test, y_test = torch.load("testing_data_100.pt", weights_only=True)

# Create DataLoader for the test dataset
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 8)  # 16 input neurons, 8 hidden neurons
        self.fc2 = nn.Linear(8, 4)   # 8 hidden neurons, 4 output neurons (for 4 classes)
        self.activation = nn.ReLU()        # Activation function

    def forward(self, x, min_val=-1.0, max_val=1.0):
        # Threshold and binarize weights for fc1
        fc1_weight = self.fc1.weight.data
        fc1_bias = self.fc1.bias.data
        print(torch.mean(fc1_weight))
        threshold_1 = torch.mean(fc1_weight).item()
        binarized_fc1_weight = torch.where(fc1_weight > threshold_1, torch.tensor(max_val), torch.tensor(min_val))
        binarized_fc1_bias = torch.where(fc1_bias > threshold_1, torch.tensor(max_val), torch.tensor(min_val))

        torch.save(binarized_fc1_weight.T, 'L1_Weights.pt')
        torch.save(binarized_fc1_bias.T, 'L1_Bias.pt')

        # Threshold and binarize weights for fc2
        fc2_weight = self.fc2.weight.data
        fc2_bias = self.fc2.bias.data
        print(torch.mean(fc2_weight))
        threshold_2 = torch.mean(fc2_weight).item()
        binarized_fc2_weight = torch.where(fc2_weight > threshold_2, torch.tensor(max_val), torch.tensor(min_val))
        binarized_fc2_bias = torch.where(fc2_bias > threshold_2, torch.tensor(max_val), torch.tensor(min_val))

        torch.save(binarized_fc2_weight.T, 'L2_Weights.pt')
        torch.save(binarized_fc2_bias.T, 'L2_Bias.pt')

        # Compute forward pass with binarized weights
        # print(binarized_fc2_weight[0])
        x = self.activation(torch.matmul(x, binarized_fc1_weight.T) + binarized_fc1_bias)
        x = torch.clamp(x, min=0.0, max=1.0)
        print(torch.mean(x))
        threshold_x = torch.mean(x).item()
        x_prog = torch.where(x > threshold_x, torch.tensor(1.0), torch.tensor(0.0))
        x = torch.matmul(x_prog, binarized_fc2_weight.T) + binarized_fc2_bias
        return x

# Visualize some samples with labels as letters
label_to_letter = {0: "A", 1: "T", 2: "V", 3: "X"}

# Initialize the model
model = SimpleNN()

# Load the state dict into the model
model.load_state_dict(torch.load("neuro_model.pth", weights_only=True))

# Set the model to evaluation mode (important for inference)
model.eval()

# Function to test the model and compute accuracy
def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, labels in test_loader:
            # print(inputs.shape)
            outputs = model(inputs)  # Forward pass
            # print(outputs.shape)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            # visualize_samples(inputs.numpy(), predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Test the model on the test dataset
test_model(model, test_loader)
