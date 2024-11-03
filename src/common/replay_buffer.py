import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        indices = random.sample(range(len(self.buffer)), self.batch_size)
        batch = [self.buffer[i] for i in indices]
        return batch, indices

    def size(self):
        return len(self.buffer)

    def can_sample(self):
        return len(self.buffer) >= self.batch_size


