from gridworld import make_four_rooms
import numpy as np
from scipy import special

env = make_four_rooms()

q = np.random.randn(25, 25, 5) * 1e-8
q = np.zeros((25, 25, 5))

for i in range(1000):
  nq = np.zeros((25, 25, 5))
  for x in range(25):
    for y in range(25):
      for a in range(5):
        ns = env.model(np.array([x, y]), a)

        qs = q[x, y, :]
        qns = q[ns[0], ns[1], :]

        logsoftmax_qs = qs - special.logsumexp(qs)
        logsoftmax_qns = qns - special.logsumexp(qns)
        log_inverse = env.log_inverse(np.array([x, y]), ns, a)
        first_term = log_inverse - logsoftmax_qs[a]
        second_term = (np.exp(logsoftmax_qns) * qns).sum(axis=-1)
        nq[x, y, a] = first_term + second_term
  q = nq
  print(q.max(axis=-1))
  print()
