import torch.nn as nn
import torch.nn.functional as F

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MultiLabelMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)


    def forward(self, x_in, apply_sigmoid=False):

        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)

        if apply_sigmoid:
            output = F.sigmoid(output)

        return output