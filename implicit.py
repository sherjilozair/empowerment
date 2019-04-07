import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

from torch.distributions import Categorical
from mnist_counter_env import MnistEnv


max_steps = 3

def process_observation(observation):
  observation = torch.as_tensor(observation, dtype=torch.float32)
  observation /= 255.0
  return observation

class EmpowermentAgent(nn.Module):
  def __init__(self, input_size, n_actions):
    super(EmpowermentAgent, self).__init__()

    self.input_size = input_size
    self.n_actions = n_actions

    self.embedding = nn.Linear(input_size, 200)

    self.p_lstm = nn.LSTM(200, 200, 3, batch_first=True)
    self.p_policy = nn.Linear(200, n_actions)
    self.p_value = nn.Linear(200, 1)

    self.q_lstm = nn.LSTM(200 * 2, 200, 3, batch_first=True)
    self.q_policy = nn.Linear(200, n_actions)

    self.zero_state = torch.zeros(3, 1, 200).cuda(), torch.zeros(3, 1, 200).cuda()

  def forward(self, x):
    h, _ = self.p_lstm(x)
    logits = self.p_policy(h)
    return F.log_softmax(logits)

  def get_action(self, observation):
    sample = random.random()
    if sample < 0.05:
      return random.randint(0, 4)
    observation = process_observation(observation)
    observation = torch.flatten(observation, 0, 2)
    observation = torch.unsqueeze(torch.unsqueeze(observation, 0), 0)
    observation = observation.cuda()
    embedding = F.relu(self.embedding(observation))
    hidden, self.state = self.p_lstm(embedding, self.state)
    logits = self.p_policy(hidden)
    policy = Categorical(logits=torch.squeeze(logits))
    self.entropies.append(policy.entropy().item())
    action = policy.sample().item()
    return action

  def reset_episode(self):
    self.state = self.zero_state

  def get_loss(self, states, actions):
    embedding = F.relu(self.embedding(states))
    p_state, _ = self.p_lstm(embedding)
    concat_embedding = torch.cat((embedding, torch.unsqueeze(embedding[:, -1, :], dim=1).repeat(1, max_steps, 1)), dim=2)
    q_state, _ = self.q_lstm(concat_embedding)

    p_logits = self.p_policy(p_state)
    q_logits = self.q_policy(q_state)

    p_logprobs = - F.cross_entropy(p_logits.view(-1, 5), actions.view(-1), reduction='none').view(-1, max_steps)
    q_logprobs = - F.cross_entropy(q_logits.view(-1, 5), actions.view(-1), reduction='none').view(-1, max_steps)

    rewards = q_logprobs - p_logprobs
    returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))

    values = torch.squeeze(self.p_value(p_state))
    self.empowerment = torch.mean(returns[:, 0])
    advantage = returns.detach() - values

    variational_loss = -torch.mean(q_logprobs)
    policy_loss = -torch.mean(advantage.detach() * p_logprobs)
    value_loss = torch.mean(advantage * advantage)

    return policy_loss, value_loss, variational_loss


def main():
  env = MnistEnv(max_steps=max_steps)

  agent = EmpowermentAgent(env.shape[0] * env.shape[1], env.action_space.n).cuda()
  optimizer = optim.Adam(agent.parameters(), lr=1e-4)
  for i in range(100000):
    states_batch = []
    actions_batch = []
    final_states = []
    agent.entropies = []
    for j in range(128):
      states = []
      actions = []
      infos = []
      state = env.reset_to_mid()
      agent.reset_episode()
      done = False
      k = 0
      while not done:
        states.append(state)
        k += 1
        # print(i, j, k)
        action = agent.get_action(state)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        infos.append(info['state'].tolist())
        state = next_state
      final_states.append(info['state'].tolist())
      # print(infos)
      states_batch.append(np.array(states))
      actions_batch.append(np.array(actions))
    hist = [[0]*10 for i in range(10)]
    for state in final_states:
      hist[state[0]][state[1]] += 1
    states = np.array(states_batch)
    actions = np.array(actions_batch)
    states = process_observation(states)
    states = torch.as_tensor(states).cuda()
    actions = torch.as_tensor(actions).cuda()
    states = torch.flatten(states, 2, 4)

    optimizer.zero_grad()
    policy_loss, value_loss, variational_loss = agent.get_loss(states, actions)
    # if i % 3 == 0:
    loss = policy_loss + value_loss + variational_loss
    # else:
    # loss = value_loss + variational_loss

    loss.backward()
    optimizer.step()
    print('iter', i)
    for row in hist:
      print(row)
    print()
    print('empowerment', agent.empowerment.item())
    print('entropies', np.mean(agent.entropies))

    print('policy loss', policy_loss.item())
    print('value loss', value_loss.item())
    print('variational loss', variational_loss.item())

    del states
    del actions

if __name__ == '__main__':
  main()
