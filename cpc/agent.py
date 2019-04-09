import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):

  def __init__(self, encoder, recurrence, discriminators, device):
    super(Agent, self).__init__()
    self.encoder = encoder
    self.recurrence = recurrence
    self.discriminators = discriminators
    self.device = device

    self.labels = {step: torch.arange(self.batch_size * (self.sequence_length - step)).to(device) for step in discriminators}
    self.optimizer = torch.optim.SparseAdam(self.parameters(), lr=2e-4)

  def act(self, observation):
    latents = self.encoder(observation)
    contexts = self.recurrence(latents)

  def reset(self):
    pass

  def update(self, minibatch):
    observations = minibatch['observations']
    actions = minibatch['actions']

    latents = self.encoder(observations)
    contexts = self.recurrence(latents)

    batch_size, sequence_length, latent_size = latents.size()

    all_costs = {}
    all_accuracies = {}
    for skip, discriminator in discriminators.items():
      n_predictions = sequence_length - skip
      predics = self.discriminators[skip](contexts[:, :-skip, :], actions[:, :-skip, :])
      targets = latents[:, skip:, :]
      flat_predics = predics.reshape(-1, latent_size)
      flat_targets = targets.reshape(-1, latent_size)
      logits = torch.matmul(flat_predics, flat_targets.t())
      skip_costs = F.cross_entropy(logits, self.labels[skip]).view(batch_size, n_predictions)
      all_costs[skip] = skip_costs

    # costs [batch_size, sequence_length]
    costs = sum(F.pad(skip_costs, (0, skip)) for skip_costs in all_costs.values())

    loss = torch.mean(costs)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def train(self, env, args):
    pass




