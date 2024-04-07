import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden_dim1 = hidden_dim1
        # self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out