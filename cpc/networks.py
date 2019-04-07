import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter



class IndexedLinear(nn.Module):
  """This is needed for action-indexed CPC discriminators."""

  def __init__(self, n_inputs, n_outputs, n_indices, bias=True):
		super(Linear, self).__init__()
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
    self.n_indices = n_indices
    self.bias = bias

		self.weight = Parameter(torch.Tensor(n_indices * n_outputs, n_inputs))
		if bias:
				self.bias = Parameter(torch.Tensor(n_indices * n_outputs))
		else:
				self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
  	init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

  def forward(self, inputs, indices):
    indexed_weights = F.embedding(indices, self.weight.view(self.n_indices, self.n_outputs * self.n_inputs)).view(self.n_outputs, self.n_inputs)
    indexed_bias = F.embedding(indices, self.bias.view(self.n_indices, self.n_outputs))
    return F.linear(input, indexed_weights, indexed_bias)




class AtariCNN(nn.Module):
  pass



