import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim):
        super(MultilayerPerceptron, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x_in, apply_softmax=False):

        intermediate = F.relu(self.fc1(x_in))

        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output