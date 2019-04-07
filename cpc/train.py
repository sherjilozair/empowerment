import gym
import argparse

parser = argparse.ArgumentParser(description='Train CPC.')
parser.add_argument('--latent_size', type=int, default=256,
                    help='Latent size.')
parser.add_argument('--cuda', type=bool, default=True,
                    help='CUDA or not.')
args = parser.parse_args()

def main():
  env = gym.make('PongNoFrameskip-v4')
  encoder = make_encoder(env.observation_space, args.latent_size)
  recurrence = make_recurrence(args.latent_size, args.recurrence_type,
      args.recurrence_size, args.recurrence_layers)
  discriminators = make_discriminators(
      args.recurrence_size, args.latent_size, env.action_space.n,
      args.sequence_length, args.prediction_range)
  device = torch.device('cuda') if args.cuda else torch.device('cpu')
  agent = Agent(encoder, recurrence, discriminators, device)



