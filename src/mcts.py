from __future__ import annotations

import hashlib
import json
import copy
import math
import random
import numpy as np

import torch

from agents.simple_agent import SimpleAgent
from envs.env import SimpleHispaniaEnv
from envs.core.entities import Action, GameState, Nation

class Node:
    def __init__(self, debug: bool = False):
        self.children: dict[Action, Node] = {}
        self.N: dict[Action, int] = {}
        self.W: dict[Action, float] = {}
        self.P: dict[Action, float] = {}
        self.is_expanded: bool = False
        self.is_terminal = False    # Mostly for debugging, not used in MCTS logic
        self.debug = debug
    
    def UCB(self, action: Action, c_puct: float) -> float:
        q = self.Q(action)
        u = self.U(action, c_puct)
        return q + u

    def U(self, action: Action, c_puct: float) -> float:
        n = self.N.get(action, 0)
        p = self.P.get(action, 0.0)
        u = c_puct * p * (math.sqrt(sum(self.N.values()) + 1e-8) / (1 + n))
        return u

    def Q(self, action: Action) -> float:
        n = self.N.get(action, 0)
        if n == 0:
            return 0.0
        return self.W.get(action, 0.0) / n

    def expand(self, priors: dict[Action, float]) -> None:
        if not priors:
            self.is_expanded = False
            self.P = {}
            self.N = {}
            self.W = {}
            print("Warning: Attempted to expand a node with empty priors. This may indicate a problem with the environment or the model.")
            return
        self.P = priors
        self.is_expanded = True
        for a in priors:
            self.N[a] = 0
            self.W[a] = 0.0

    def is_leaf(self) -> bool:
        return not self.is_expanded

def state_fingerprint(state: GameState) -> str:
    """Cheap content hash of a GameState, used to verify tree/env sync."""
    return hashlib.sha256(
        json.dumps(state.to_dict(), sort_keys=True).encode()
    ).hexdigest()

class MCTS:
    def __init__(
        self,
        agent: SimpleAgent,
        c_puct: float = 1.0,
        device: str = "cpu",
        root_dirichlet_alpha: float = 0.3,
        root_dirichlet_eps: float = 0.25,
        debug: bool = False,
    ) -> None:
        self.agent = agent
        self.c_puct = c_puct
        self.device = device
        self.root: Node | None = None
        self._expected_root_fp: str | None = None  # fingerprint self.root should match
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_eps = root_dirichlet_eps
        self.debug = debug

    def _print_tree(
        self,
        node: Node,
        prefix: str = "",
        action_from_parent: Action | None = None,
        max_depth: int = 100,
        depth: int = 0,
    ) -> None:
        if not self.debug:
            return
        if depth > max_depth:
            return

        if action_from_parent is None:
            print("ROOT")
        else:
            parent = prefix[:-4]
            branch = "└── " if prefix.endswith("    ") else "├── "

            print(
                f"{parent}{branch}{action_from_parent}\n"
                f"{parent}    N={node.parent_N:4d}"
                f"  W={node.parent_W:7.3f}"
                f"  Q={node.parent_Q:7.3f}"
                f"  P={node.parent_P:7.3f}"
            )

        if node.is_terminal:
            print(prefix + "    [TERMINAL]")
            return

        if not node.is_expanded:
            print(prefix + "    [LEAF]")
            return

        actions = sorted(
            node.P.keys(),
            key=lambda a: node.N.get(a, 0),
            reverse=True,
        )

        for i, action in enumerate(actions):
            child = node.children.get(action)

            if child is None:
                print(
                    prefix
                    + ("└── " if i == len(actions)-1 else "├── ")
                    + f"{action}"
                    + f"  N={node.N[action]}"
                    + f"  W={node.W[action]:.3f}"
                    + f"  Q={node.Q(action):.3f}"
                    + f"  P={node.P[action]:.3f}"
                    + "   [UNEXPANDED]"
                )
                continue

            child.parent_N = node.N[action]
            child.parent_W = node.W[action]
            child.parent_Q = node.Q(action)
            child.parent_P = node.P[action]

            self._print_tree(
                child,
                prefix + ("    " if i == len(actions)-1 else "│   "),
                action,
                max_depth,
                depth + 1,
            )

    def _count_nodes(self, node: Node) -> int:
        total = 1
        for child in node.children.values():
            total += self._count_nodes(child)
        return total

    def _count_leaves(self, node: Node) -> int:
        if not node.children:
            return 1
        total = 0
        for child in node.children.values():
            total += self._count_leaves(child)
        return total

    # Netwrok Evaluation meant to be done at each Node
    def _eval(
        self, env: SimpleHispaniaEnv
    ) -> tuple[dict[Action, float], dict[Nation, float]]:
        g, t, u, masks, index_to_unit_id = (
            self.agent.build_model_inputs_and_masks(env)
        )
        legal_actions: list[Action] = self.agent.enumerate_legal_actions(
            env, masks, index_to_unit_id
        )
        
        if not legal_actions:
            print("Warning: No legal actions available during MCTS evaluation. Returning default values. This is an error.")
            return {}, {}

        model = self.agent.model
        with torch.no_grad():
            out: dict = model(
                g.to(self.device),
                t.to(self.device),
                u.to(self.device),
                masks=masks,
            )

            at_log_probs: torch.Tensor = model.checked_log_softmax(out["action_type_logits"], name="action_type",).squeeze(0)
            unit_log_probs: torch.Tensor = model.checked_log_softmax(out["unit_logits"], name="unit",).squeeze(0)
            unit_type_log_probs: torch.Tensor = model.checked_log_softmax(out["unit_type_logits"], name="unit_type",).squeeze(0)
            tile_log_probs: torch.Tensor = model.checked_log_softmax(out["tile_logits"], name="tile",).squeeze(0)

            log_probs_list: list[float] = []
            
            # Mapping from unit_id to index in the unit features tensor
            unit_id_to_index = {
                unit_id: idx
                for idx, unit_id in enumerate(index_to_unit_id)
            }

            for action in legal_actions:
                at_idx = action.type.value
                unit_idx = -1
                if action.unit_id is not None:
                    if action.unit_id in unit_id_to_index:
                        unit_idx = unit_id_to_index[action.unit_id]

                unit_type_idx = -1
                if getattr(action, "unit_type", None) is not None:
                    unit_type_idx = int(action.unit_type.value)

                tile_idx = action.target_tile if action.target_tile is not None else -1

                if not (0 <= at_idx < at_log_probs.numel()):
                    log_probs_list.append(-float("inf"))
                    continue

                logp: float = float(at_log_probs[at_idx].item())

                if unit_idx >= 0:
                    if 0 <= unit_idx < unit_log_probs.numel():
                        logp += float(unit_log_probs[unit_idx].item())
                    else:
                        logp = -float("inf")

                if unit_type_idx >= 0:
                    if 0 <= unit_type_idx < unit_type_log_probs.numel():
                        logp += float(unit_type_log_probs[unit_type_idx].item())
                    else:
                        logp = -float("inf")

                if tile_idx >= 0:
                    tile_unit_idx = unit_idx if unit_idx >= 0 else 0
                    tile_shape = tile_log_probs.shape
                    if (
                        0 <= at_idx < tile_shape[0]
                        and 0 <= tile_unit_idx < tile_shape[1]
                        and 0 <= tile_idx < tile_shape[2]
                    ):
                        logp += float(
                            tile_log_probs[at_idx, tile_unit_idx, tile_idx].item()
                        )
                    else:
                        logp = -float("inf")
                
                log_probs_list.append(logp)

            log_probs: torch.Tensor = torch.tensor(log_probs_list, device=self.device)
            
            finite: torch.Tensor = torch.isfinite(log_probs)
            if finite.any():
                max_val: torch.Tensor = torch.max(log_probs[finite])
                stabilized: torch.Tensor = log_probs - max_val
                probs: torch.Tensor = torch.nn.functional.softmax(stabilized, dim=0)
                probs = torch.where(finite, probs, torch.zeros_like(probs))
                probs_list: list[float] = probs.detach().cpu().tolist()
            else:
                probs_list: list[float] = [1.0 / len(legal_actions)] * len(legal_actions)

        priors: dict[Action, float] = {
            action: float(prob)
            for action, prob in zip(legal_actions, probs_list)
        }

        values: dict[Nation, float] = {
            nation: float(out["value"][0, idx].item())
            for idx, nation in enumerate(env.state.playing_nations)
        }
        
        return priors, values

    # Reset the MCTS tree at the start of a new episode/game
    def reset(self) -> None:
        """Call this at the start of every new episode/game."""
        self.root = None
        self._expected_root_fp = None

    def run(
        self, 
        env: SimpleHispaniaEnv, 
        n_simulations: int = 50,
        is_deterministic: bool = False
    ) -> tuple[dict[Action, int], Action | None]:
        # Root node
        root: Node | None = None
        reused_tree = False

        # Try to reuse the subtree carried over from the previous decision.
        if self.root is not None and self.root.is_expanded and self._expected_root_fp is not None:
            if self._expected_root_fp == state_fingerprint(env.state):
                root = self.root
                reused_tree = True 
            else:
                print("Warning: persistent root doesn't match actual env state (likely a stochastic RESOLVE_BATTLE outcome). Rebuilding tree.")

        if self.debug:
            print()
            print("=" * 90)
            print("SEARCH INITIALIZATION")
            print("=" * 90)

            if reused_tree:
                print("Reusing subtree from previous search.")
            else:
                print("Creating a new root node.")

            print("=" * 90)

        if root is None:
            priors, _ = self._eval(env)
            if not priors:
                print("Warning: No priors returned from network evaluation. Returning empty counts and None for chosen action.")
                return {}, None
            root = Node(debug=self.debug)
            root.expand(priors)

        # Add Dirichlet noise at root for exploration
        if self.root_dirichlet_eps and self.root_dirichlet_alpha and root.P:
            keys: list[Action] = list(root.P.keys())
            vals: list[float] = [root.P[k] for k in keys]
            noise = np.random.default_rng().dirichlet(
                [self.root_dirichlet_alpha] * len(vals)
            )
            noise = noise / noise.sum()
            for i, k in enumerate(keys):
                root.P[k] = (
                    (1.0 - self.root_dirichlet_eps) * root.P[k]
                    + self.root_dirichlet_eps * float(noise[i])
                )

        # MCTS simulations: pure tree search using network priors
        for _ in range(n_simulations):
            sim_env = copy.deepcopy(env)

            node: Node = root
            path: list[tuple[Node, Action, Nation | None, dict[Nation, float]]] = []
            debug_path: list[str] = [] # Debug

            depth: int = 0

            # Selection + Expansion
            while node.is_expanded:

                action = max(node.P.keys(), key=lambda a: node.UCB(a, self.c_puct))

                debug_path.append(action)

                acting_nation: Nation = sim_env.state.current_nation
                _, rewards = sim_env.step(action)
                path.append((node, action, acting_nation, rewards))

                if action not in node.children:
                    node.children[action] = Node(debug=self.debug)
                
                node = node.children[action]

                depth += 1

                if sim_env.done:
                    break
            
            # Evaluation
            if sim_env.done:
                node.is_terminal = True
                # Terminal state: no future VP remains to be earned.
                value: dict[Nation, float] = {
                    nation: 0.0 for nation in sim_env.state.playing_nations
                }
            else:
                # Get priors and value from network
                priors, value = self._eval(sim_env)
                
                node.expand(priors)

            # Backup
            G: dict[Nation, float] = value.copy()
            for nd, action, acting_nation, rewards in reversed(path):
                for nation, reward in rewards.items():
                    G[nation] = G.get(nation, 0.0) + float(reward)
                nd.N[action] += 1
                nd.W[action] += G[acting_nation] # MaxN-style backup (Maximize my own VP score)
            
        # Action Selection
        if not root.N:
            print("Warning: No actions were selected during MCTS. This may indicate a problem with the environment or the model. Returning empty counts and None for chosen action.")
            return {}, None

        if self.debug:
            print("\n")
            print("=" * 90)
            print("MCTS SEARCH TREE")
            print("=" * 90)

            print(f"Simulations : {n_simulations}")
            print(f"Total nodes : {self._count_nodes(root)}")
            print(f"Leaf nodes  : {self._count_leaves(root)}")
            print()

            self._print_tree(root)

            print("=" * 90)

        counts = root.N
        actions = list(counts.keys())
        visits = np.array([counts[a] for a in actions], dtype=np.float32)
        probs = visits / visits.sum()
        
        if is_deterministic:
            chosen = actions[int(np.argmax(visits))]
        else:
            chosen = random.choices(actions, weights=probs, k=1)[0]

        if self.debug:
            print()
            print("=" * 90)
            print("ACTION SELECTED")
            print("=" * 90)
            print(f"Chosen action : {chosen}")
            print(f"Visits        : {counts[chosen]}")
            print(f"Q-value       : {root.Q(chosen):.3f}")
            print("=" * 90)

        # The outocme of the chosen action via the step.
        expected_env = copy.deepcopy(env)   # Another deepcopy is pretty costly...
        expected_env.step(chosen)
        self._expected_root_fp = state_fingerprint(expected_env.state)

        
        new_root = root.children.get(chosen)

        if self.debug:
            print()
            print("=" * 90)
            print("TREE REUSE")
            print("=" * 90)

            print(f"Old root id      : {id(root)}")
            print(f"Chosen action    : {chosen}")

            if new_root is None:
                print("Chosen branch was never expanded during search.")
                print("Next search will start from a fresh root.")
            else:
                print(f"New root id      : {id(new_root)}")
                print("Subtree retained")
                print(f"Discarded branches : {len(root.children) - 1}")
                print(f"Children kept      : {len(new_root.children)}")

            print("=" * 90)

        self.root = new_root if new_root is not None else None

        return counts, chosen


# TODO: CHECK EVAL AND TRAIN MODE BEFORE AND AFTER EVAL AND TRAIN OBVIOUSLY....
# TODO: May or not be importatnt thing about our policy but the values of priors precicted from the MCTS are like this 
# it kind of makes sense bu also not but like if i have the action move unit to t0 and move unit to t1 and end phase instead of being in the beginnign the odds of each one 0.333 or soemthign similar it's 0.5 for end phase and 0.25 for the other move actiosn lowkey kind of makes sense pq o action type escolhe entre as duas primeiro neh??
# TODO: Evaluate the final  vp my head is tryign to predict that or predict the amount of VP left to earn. which one is better???
# Valores na MCTS valors por player
# muLTIPLOS VALUES EACH PLAYER HAS IT'S OWN MCTS
# Choose what selection on the node of the enemy. You minimize the punctiation of the other adversary. WHen with multiple nodes which one to choose
# Minimizar os pontos daquele que está a frente o joagdor mais frente (Heuristic)
# 4 player game. I have two actions i can attak player 2 or player 3 both give me the same points, need to know even thoguh they both give same points for me it may penalize the other player. Paranoid search / Max my reward only / best reply search / Multi tree MCTS  