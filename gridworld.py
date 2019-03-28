import numpy as np

CHARSET = " <>^vx"

class GridWorld(object):

  def __init__(self, level, config):
    assert all([len(row) == len(level[0]) for row in level[1:]])
    self.level = level
    self.height = len(level)
    self.width = len(level[0])
    self.config = config

  def reset(self):
    while True:
      self.x = np.random.randint(self.width)
      self.y = np.random.randint(self.width)
      if self.level[self.y][self.x] != 'x':
        break
    return np.array([self.x, self.y])

  def step(self, action):
    new_state = self.model(np.array([self.x, self.y]), action)
    self.x, self.y = new_state
    return new_state

  def states(self):
    for x in range(self.width):
      for y in range(self.height):
        yield np.array([x, y])

  def model(self, state, action):
    x, y = state
    if action == 0:
      pass
    if action == 1:
      if self.level[y][x] not in ">x" and self.level[y][x+1] not in "x":
        x += 1
    if action == 2:
      if self.level[y][x] not in "<x" and self.level[y][x-1] not in "x":
        x -= 1
    if action == 3:
      if self.level[y][x] not in "^x" and self.level[y-1][x] not in "x":
        y -= 1
    if action == 4:
      if self.level[y][x] not in "vx" and self.level[y+1][x] not in "x":
        y += 1
    return np.array([x, y])

  def log_inverse(self, state, next_state, action):
    x, y = state
    nx, ny = next_state
    if self.level[y][x] == 'x':
      return -np.log(5)
    if

def make_four_rooms():
  level = [
        "xxxxxxxxxxxxxxxxxxxxxxxxx",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x                       x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "xxxxxx xxxxxxxxxxx xxxxxx",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x                       x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "x           x           x",
        "xxxxxxxxxxxxxxxxxxxxxxxxx",
      ]
  return GridWorld(level, {})

if __name__ == '__main__':
  env = make_four_rooms()
