import torch.nn as nn
import torch.nn.functional as F

class DeepFC(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super(DeepFC, self).__init__()

        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)


    def forward(self, x_in, apply_sigmoid=False):
        o1 = F.relu(self.fc1(x_in))
        o2 = F.relu(self.fc2(o1))
        output = self.fc3(o2)

        if apply_sigmoid:
            output = F.sigmoid(output)

        return output