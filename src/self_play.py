from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from agents.simple_agent import SimpleAgent
from envs.core.entities import Action
from envs.core.enums import Nation
from envs.env import SimpleHispaniaEnv
from mcts import MCTS
from replay_buffer import TrajectoryExample

if TYPE_CHECKING:
    from metrics import MetricsCollector

class SelfPlayGame:
    def __init__(
        self,
        env: SimpleHispaniaEnv,
        agent: SimpleAgent,
        device: str = "cpu",
        mcts_sims: int = 32,
        mcts_c_puct: float = 1.0,
        debug: bool = False,
    ) -> None:
        self.env: SimpleHispaniaEnv = env
        self.agent: SimpleAgent = agent
        self.mcts: MCTS = MCTS(
            agent=agent,
            c_puct=mcts_c_puct,
            device=device,
            debug=False,
        )
        self.mcts_sims: int = mcts_sims
        self.debug: bool = debug

    def run(self, seed: int | None = None) -> list[TrajectoryExample]:
        self.env.reset(seed=int(np.random.randint(0, 1_000_000)) if seed is None else seed)
        self.mcts.reset()

        game_actions: list[Action] = [] # Debug actions chosen during the game
        trajectory: list[tuple] = []
        raw_immediate_rewards: list[Dict[Nation, float]] = []
        step_in_game: int = 0

        while not self.env.done:
            # 1. Get current state and extract features
            g, tile, unit, masks, index_to_unit_id = (
                self.agent.build_model_inputs_and_masks(self.env)
            )

            state_vp = {
                nation: float(self.env.state.vp_scores.get(nation, 0.0))
                for nation in self.env.state.playing_nations
            }
            turn_number: int = int(self.env.state.turn_number)
            
            acting_nation: Nation | None = self.env.state.current_nation
            if acting_nation is None:
                print("Warning: Acting nation is None during self-play. This should not happen. Skipping this step.")
                break
            
            # 2. Run MCTS and simulate to get visit counts and chosen action
            action_counts, chosen_action = self.mcts.run(env=self.env, n_simulations=self.mcts_sims, is_deterministic=False)

            if chosen_action is None:
                print("Warning: No action was chosen during MCTS. This may indicate a problem with the environment or the model. Ending self-play early.")
                break

            # 3. Convert visit counts to policy targets
            total = float(sum(action_counts.values()))
            if total == 0:
                print("Warning: Total visit counts from MCTS is zero. This may indicate a problem with the environment or the model. Ending self-play early.")
                break
            
            mcts_pi: Dict[Action, float] = {
                a: c / total for a, c in action_counts.items()
            }
            
            trajectory.append(
                (
                    g,
                    tile,
                    unit,
                    index_to_unit_id,
                    masks,
                    mcts_pi,
                    acting_nation,
                    turn_number,
                    state_vp,
                )
            )

            # 4. Execute action in real environment
            _, rewards = self.env.step(chosen_action)
            raw_immediate_rewards.append(rewards)
            game_actions.append(chosen_action)
            step_in_game += 1

        # 5. Compute value targets via reward-to-go using environment step rewards
        final_vp = {
            n: float(self.env.state.vp_scores.get(n, 0.0))
            for n in self.env.state.playing_nations
        }

        nations = list(final_vp.keys())

        # Normalize raw rewards to ensure every step has an entry for each nation
        immediate_rewards = [
            {n: float(r.get(n, 0.0)) for n in nations} for r in raw_immediate_rewards
        ]

        # Compute value targets via reward-to-go
        value_targets: list[Dict[Nation, float]] = [
            {n: 0.0 for n in nations} for _ in range(len(trajectory))
        ]

        for nation in nations:
            G = 0.0
            for i in range(len(trajectory) - 1, -1, -1):
                G = immediate_rewards[i].get(nation, 0.0) + G
                value_targets[i][nation] = G

        # Build Training Examples
        examples: list[TrajectoryExample] = []
        for i, (g, tile, unit, index_to_unit_id, masks, pi, nation, turn_number, state_vp) in enumerate(
            trajectory
        ):
            examples.append(
                TrajectoryExample(
                    global_feats=g,
                    tile_feats=tile,
                    unit_feats=unit,
                    masks=masks,
                    index_to_unit_id=index_to_unit_id,
                    acting_nation=nation,
                    pi=pi,
                    value=value_targets[i],
                )
            )

        if self.debug:
            print()
            print("=" * 90)
            print("SELF PLAY GAME")
            print("=" * 90)

            for i, action in enumerate(game_actions):
                print(f"{i:3d}: {action}")

            print()
            print("Final VP:")
            for nation, vp in final_vp.items():
                print(f"  {nation}: {vp}")

            print()
            print("Value targets:")
            for i, (nation, turn_number, target) in enumerate(
                zip((entry[6] for entry in trajectory), (entry[7] for entry in trajectory), value_targets)
            ):
                print(f"  {i:3d} | turn={turn_number} | {nation}: {target}")

            print("=" * 90)

        return examples
