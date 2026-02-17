import random

class RandomAgent:
    def select_action(self, env):
        legal = env.legal_actions()
        return random.choice(legal)
