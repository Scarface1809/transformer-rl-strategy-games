import random
from agents.simple_agent import SimpleAgent

# TODO: Fix the legal actiosn. They need to come from the env!
class RandomAgent:
    def select_action(self, env):
       agent = SimpleAgent(model=None)

       _, _, _, masks, index_to_unit_id = agent.build_model_inputs_and_masks(env)

       legal_actions = agent.enumerate_legal_actions(
          env,
          masks,
          index_to_unit_id,
       )

       return random.choice(legal_actions)