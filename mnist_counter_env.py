
import os
import numpy as np
import gym
from gym.utils import seeding
import torchvision.datasets as dset
import torchvision.transforms as transforms



class MnistEnv(gym.Env):
    def __init__(self, num_digits=2, max_steps=10, use_training_set=True, seed=1337):
        self.shape = 28, num_digits * 28
        self.num_sym = num_digits
        self.max_steps = max_steps
        self.seed(seed=seed)

        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        transform = transforms.Normalize((0.5,), (1.0,))
        data = dset.MNIST(root=root, train=use_training_set, transform=transform, download=True)
        x, y = data.data, data.targets
        self.data = [x[y == i] for i in range(10)]
        self.num_samples = [x_.shape[0] for x_ in self.data]

        # the first action is the null action
        self.action_space = gym.spaces.Discrete(1 + 2 * self.num_sym)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=self.shape + (1,),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

        self.reward_range = (0, 1)

    def _transform_action(self, a):
        n = (a - 1) // 2
        c = a % 2 - (a - 1) % 2
        return n, c

    def step(self, action):
        if action > 0:
            n, c = self._transform_action(action)
            self.state[n] += c
            self.state = self.state % 10
        self.step_count += 1
        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        return obs, 0, done, {'state': self.state}

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count


    def reset_to_mid(self):
        ''' generate new world '''
        self.step_count = 0
        self.state = np.array([5, 5])
        return self.gen_obs()

    def reset(self):
        ''' generate new world '''
        self.step_count = 0
        self.state = np.random.randint(10, size=(self.num_sym))
        return self.gen_obs()

    @property
    def observed_state(self):
        ids = [np.random.choice(self.num_samples[i]) for i in self.state]
        img = np.concatenate([self.data[self.state[i]][ids[i]] for i in range(self.num_sym)], axis=1)
        return img

    def _reward(self):
        ''' Compute the reward to be given upon success '''
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def gen_obs(self):
        return self.observed_state.reshape(*self.shape, 1)

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]



class MnistEnv1(MnistEnv):
    def __init__(self):
        super().__init__(num_digits=1)

class MnistEnv2(MnistEnv):
    def __init__(self):
        super().__init__(num_digits=2)

class MnistEnv3(MnistEnv):
    def __init__(self):
        super().__init__(num_digits=3)


if __name__ == '__main__':
    env = MnistEnv2()
    obs = env.reset()
    print('obs shape', obs['image'].shape)
    print('obs', obs['image'].squeeze())
    a = np.random.randint(5)
    obs, rew, don, _ = env.step(a)
    print('obs shape', obs['image'].shape)
    print('obs', obs['image'].squeeze())

