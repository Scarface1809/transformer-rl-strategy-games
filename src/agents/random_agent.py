import random

class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, env):
        legal = env.legal_actions()
        return random.choice(legal)
