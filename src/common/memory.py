class Memory:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.observations[:]
        del self.next_observations[:]
        del self.actions[:]
        del self.action_probs[:]
        del self.rewards[:]
        del self.dones[:]
