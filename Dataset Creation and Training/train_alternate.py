import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the saved datasets
X_train, y_train = torch.load("training_data_500.pt")
X_val, y_val = torch.load("validation_data_60.pt")
X_test, y_test = torch.load("testing_data_100.pt")

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16, 8)  # 16 input neurons, 8 hidden neurons
        self.fc2 = nn.Linear(8, 4)   # 8 hidden neurons, 4 output neurons (for 4 classes)
        self.activation = nn.Tanh()  # Activation function

    def forward(self, x):
        x = (self.fc1(x))
        x = self.activation(x)
        x = torch.clamp(x, min=0.0, max=1.0)  # Clamp outputs between 0 and 1
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()

# Binary-friendly weight initialization
def initialize_binary_friendly_weights(layer):
    nn.init.uniform_(layer.weight, 0.0, 1.0)  # Initialize weights in [0, 1]
    nn.init.constant_(layer.bias, 0.5)        # Initialize biases close to the middle value

# Apply binary-friendly initialization
initialize_binary_friendly_weights(model.fc1)
initialize_binary_friendly_weights(model.fc2)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to clip weights to [0, 1]
def clip_weights(model, min_val=0.0, max_val=1.0):
    for param in model.parameters():
        param.data.clamp_(min_val, max_val)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, lambda_reg=0.01):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute losses
            classification_loss = criterion(outputs, labels)
            total_loss = classification_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Clip weights to [0, 1]
            clip_weights(model)

            running_loss += total_loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=500)

# Plot the training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Test function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Test the trained model
test_model(model, test_loader)

# Save the trained model
torch.save(model.state_dict(), "new_model_custom_act.pth")

def plot_weight_histogram(model):
    # Iterate through model parameters (weights)
    for name, param in model.named_parameters():
        if 'weight' in name:  # We only care about weights, not biases
            # Flatten the weights to 1D
            weight_data = param.data.cpu().numpy().flatten()
            # Plot the histogram
            plt.figure(figsize=(6, 4))
            plt.hist(weight_data, bins=50, alpha=0.7)
            plt.title(f"Histogram of Weights - {name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

plot_weight_histogram(model)