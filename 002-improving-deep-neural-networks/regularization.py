import torch
import torch.nn as nn
import torch.optim as optim

learning_rate = 0.01
weight_decay = 1e-5
l1_lambda = 0.001
dropout_rate = 0.5
patience = 10
epochs = 100

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size) # Batch Normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Dropout
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn(out) # Batch Normalization
        out = self.relu(out)
        out = self.dropout(out) # Dropout
        out = self.linear2(out)
        return out

model = Net(1, 2, 1, dropout_rate)

X = torch.tensor([[1.0], [2.0], [3.0]])
target = torch.tensor([[2.0], [4.0], [6.0]])

criterion = nn.MSELoss()
# L2 regularization
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, target)

    # L1 regularization
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    total_loss = loss + l1_lambda * l1_norm

    total_loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

    # Early stopping 조건 검사
    val_loss = criterion(model(X), target)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print('early stopping')
            break