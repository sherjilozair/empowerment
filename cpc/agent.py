import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(object):

  def __init__(self, encoder, recurrence, discriminators, labels):
    self.encoder = encoder
    self.recurrence = recurrence
    self.discriminators = discriminators
    self.labels = labels

  def act(self, observation):
    latents = self.encoder(observation)
    contexts = self.recurrence(latents)

  def reset(self):
    pass

  def train(minibatch):
    latents = self.encoder(minibatch['observations'])
    contexts = self.recurrence(latents)
    batch_size, sequence_length, latent_size = latents.size()
    losses = {}
    accuracies = {}
    for skip, discriminator in discriminators.items():
      n_predictions = sequence_length - skip
      predics = self.discriminators[skip](contexts[:, :-skip, :])
      targets = latents[:, skip:, :]
      flat_predics = predics.reshape(-1, latent_size)
      flat_targets = targets.reshape(-1, latent_size)
      logits = torch.matmul(flat_predics, flat_targets.t())
      losses = F.cross_entropy(logits, self.labels[skip]).view(batch_size, n_predictions)
      losses[skip] = loss
      # accuracy = torch.argmax(logits, dim=1).eq(self.labels[skip]).float().mean()
      # accuracies[skip] = accuracy

    # TODO: use negative CPC loss to reinforce policy
    # TODO: incorporate actions








