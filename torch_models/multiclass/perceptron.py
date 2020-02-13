import torch.nn as nn
import torch.nn.functional as F


class Perceptron(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(Perceptron, self).__init__()
		self.output_dim = output_dim
		self.fc1 = nn.Linear(
			in_features=input_dim, 
			out_features=output_dim)


	def forward(self, x_in, apply_softmax=False):

		y_out = self.fc1(x_in).squeeze()

		if apply_softmax:
			y_out = F.softmax(y_out, dim=self.output_dim)

		return y_out
