class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.observations[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]