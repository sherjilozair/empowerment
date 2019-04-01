import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.distributions import Categorical
from mnist_counter_env import MnistEnv

def process_observation(observation):
  observation = torch.as_tensor(observation, dtype=torch.float32)
  observation /= 255.0
  observation -= 0.5
  return observation

class EmpowermentAgent(nn.Module):
  def __init__(self, input_size, n_actions):
    super(EmpowermentAgent, self).__init__()

    self.input_size = input_size
    self.n_actions = n_actions

    self.forward_lstm = nn.LSTM(input_size, 200, 2, batch_first=True)
    self.forward_policy = nn.Linear(200, n_actions)
    self.forward_value = nn.Linear(200, 1)

    self.bidirectional_lstm = nn.LSTM(input_size, 200, 2, batch_first=True, bidirectional=True)
    self.bidirectional_policy = nn.Linear(400, n_actions)

    self.zero_state = torch.zeros(2, 1, 200).cuda(), torch.zeros(2, 1, 200).cuda()

  def forward(self, x):
    h, _ = self.forward_lstm(x)
    logits = self.forward_policy(h)
    return F.log_softmax(logits)

  def get_action(self, observation):
    observation = process_observation(observation)
    observation = torch.flatten(observation, 0, 2)
    observation = torch.unsqueeze(torch.unsqueeze(observation, 0), 0)
    observation = observation.cuda()
    hidden, self.state = self.forward_lstm(observation, self.state)
    logits = self.forward_policy(hidden)
    policy = Categorical(logits=torch.squeeze(logits))
    action = policy.sample().item()
    return action

  def reset_episode(self):
    self.state = self.zero_state

  def get_loss(self, states, actions):
    internal_state, _ = self.forward_lstm(states)
    backward_internal_state, _ = self.bidirectional_lstm(states)

    p_logits = self.forward_policy(internal_state)
    q_logits = self.bidirectional_policy(backward_internal_state)

    p_logprobs = F.log_softmax(p_logits, dim=2)
    q_logprobs = F.log_softmax(q_logits, dim=2)

    # p_logprobs [batch_size, seqlen, n_actions]

    p_logprobs_selected = torch.squeeze(torch.gather(p_logprobs, dim=2, index=torch.unsqueeze(actions, 2)), 2)
    q_logprobs_selected = torch.squeeze(torch.gather(q_logprobs, dim=2, index=torch.unsqueeze(actions, 2)), 2)


    rewards = q_logprobs_selected - p_logprobs_selected
    returns = torch.flip(torch.cumsum(torch.flip(rewards, dims=(1,)), dim=1), dims=(1,))

    values = torch.squeeze(self.forward_value(internal_state))
    self.empowerment = torch.mean(returns[:, 0])
    advantage = returns.detach() - values

    variational_loss = -torch.mean(q_logprobs_selected)
    policy_loss = -torch.mean(advantage.detach() * p_logprobs_selected)
    value_loss = torch.mean(advantage * advantage)

    return policy_loss + value_loss + variational_loss


def main():
  env = MnistEnv()

  agent = EmpowermentAgent(env.shape[0] * env.shape[1], env.action_space.n).cuda()
  optimizer = optim.Adam(agent.parameters(), lr=1e-4)
  for i in range(10000):
    states_batch = []
    actions_batch = []
    final_states = []
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
    loss = agent.get_loss(states, actions)
    loss.backward()
    optimizer.step()

    for row in hist:
      print(row)
    print()
    print(agent.empowerment.item())

    #print(loss.item())

    del states
    del actions

if __name__ == '__main__':
  main()
