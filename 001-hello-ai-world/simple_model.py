import torch
import torch.nn as nn

# Create a simple neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return x

# Instantiate the model
simpy = SimpleNeuralNetwork(1, 1)

# Create a simple dataset
X = torch.tensor([[1.0], [2.0]])
y_hat = torch.tensor([[2.0], [4.0]])

# Set the hyperparameters
learning_rate = 0.01
epochs = 10

# Print the model parameters
print(f"Initial weights: {simpy.linear.weight}")
print(f"Initial bias: {simpy.linear.bias}")

for epoch in range(epochs):
    # Make a prediction
    y = simpy(X)
    print(f"Prediction before training: {y}")

    # Calculate the loss
    criterion = nn.MSELoss()
    loss = criterion(y, y_hat)
    print(f"Loss: {loss}")

    # Optimize the model
    optimizer = torch.optim.SGD(simpy.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the model parameters
    print(f"Updated weights: {simpy.linear.weight}")
    print(f"Updated bias: {simpy.linear.bias}")

    # Make a prediction
    y = simpy(X)
    print(f"Prediction after training: {y}")
    print(f"Loss: {criterion(y, y_hat)}")