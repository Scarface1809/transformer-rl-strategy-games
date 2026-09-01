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
from envs.core.enums import ActionType

class Node:
    def __init__(self, debug: bool = False):
        self.children: dict[Action, Node] = {}
        self.chance_children: dict[Action, dict[str, Node]] = {}  # RESOLVE_BATTLE only: action -> {outcome_fp -> Node}
        self.N: dict[Action, int] = {}
        self.W: dict[Action, dict[Nation, float]] = {}    # Full value vector for all nations per action
        self.P: dict[Action, float] = {}
        self.is_expanded: bool = False
        self.is_terminal = False    # Mostly for debugging, not used in MCTS logic
        self.debug = debug
    
    def UCB(self, action: Action, c_puct: float, nation: Nation) -> float:
        """Compute UCB value for an action from a specific nation's perspective."""
        q = self.Q(action, nation)
        u = self.U(action, c_puct)
        return q + u

    def U(self, action: Action, c_puct: float) -> float:
        n = self.N.get(action, 0)
        p = self.P.get(action, 0.0)
        u = c_puct * p * (math.sqrt(sum(self.N.values()) + 1e-8) / (1 + n))
        return u

    def Q(self, action: Action, nation: Nation | None = None) -> float | dict[Nation, float]:
        """Get Q-value(s) for an action.
        
        Args:
            action: The action to evaluate
            nation: If provided, return Q value for that nation only (as float)
                   If None, return full dict of Q values for all nations
        
        Returns:
            If nation is provided: Q value (float) for that nation
            If nation is None: dict[Nation, float] with Q values for all nations
        """
        n = self.N.get(action, 0)
        if n == 0:
            if nation is not None:
                return 0.0
            return {}
        
        w_dict = self.W.get(action, {})
        if nation is not None:
            return float(w_dict.get(nation, 0.0)) / n
        else:
            return {nat: float(w_dict.get(nat, 0.0)) / n for nat in w_dict.keys()}

    def expand(self, priors: dict[Action, float]) -> None:
        if not priors:
            raise RuntimeError(
                "Attempted to expand a MCTS node with empty priors. "
                "This indicates no legal actions exist, which should not happen during valid gameplay."
            )
        self.P = priors
        self.is_expanded = True
        for a in priors:
            self.N[a] = 0
            self.W[a] = {}  # Initialize as empty dict for all nations' values

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
        self._pending_root: Node | None = None  # Root awaiting confirmation after real step
        self._pending_action: Action | None = None  # Action taken, awaiting confirmation
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_dirichlet_eps = root_dirichlet_eps
        self.debug = debug
        self._last_root_network_values: dict[Nation, float] = {}  # Store network values from root eval
        self._last_root_network_policy: dict[Action, float] = {}  # Store network policy (priors)

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
            
            # Get W dict and format it for display
            w_dict = node.W.get(action, {})
            # Show first nation's value for compact display
            first_nation = list(w_dict.keys())[0] if w_dict else None
            w_repr = f"{w_dict.get(first_nation, 0.0):.1f}" if first_nation else "0.0"
            
            # Get Q value for first nation
            q_val = node.Q(action, first_nation) if first_nation else 0.0

            if child is None:
                print(
                    prefix
                    + ("└── " if i == len(actions)-1 else "├── ")
                    + f"{action}"
                    + f"  N={node.N[action]}"
                    + f"  W={w_repr}"
                    + f"  Q={q_val:.3f}"
                    + f"  P={node.P[action]:.3f}"
                    + "   [UNEXPANDED]"
                )
                continue

            child.parent_N = node.N[action]
            child.parent_W = w_repr
            child.parent_Q = q_val
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
            raise RuntimeError(
                f"MCTS evaluation encountered a state with no legal actions. "
                f"This should not happen during valid gameplay. "
                f"State: {env.state.to_dict()}"
            )

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
        self._pending_root = None
        self._pending_action = None

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
            priors, values = self._eval(env)
            if not priors:
                raise RuntimeError(
                    "Network evaluation returned no action priors. "
                    "This indicates no legal actions exist or model failed to produce outputs."
                )
            # Store network values and policy for diagnostics
            self._last_root_network_values = values
            self._last_root_network_policy = priors
            root = Node(debug=self.debug)
            root.expand(priors)

        # Add Dirichlet noise at root for exploration only for training.
        if (not is_deterministic and self.root_dirichlet_eps and self.root_dirichlet_alpha and root.P):
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
            # Reset the RNG for the simulation to ensure reproducibility and independence from the main environment's RNG state.
            sim_seed = int(np.random.randint(0, 1_000_000_000))
            sim_env.rng = np.random.default_rng(sim_seed)

            node: Node = root
            path: list[tuple[Node, Action, Nation | None, dict[Nation, float]]] = []
            debug_path: list[str] = [] # Debug

            depth: int = 0

            # Selection + Expansion
            while node.is_expanded:

                action = max(node.P.keys(), key=lambda a: node.UCB(a, self.c_puct, sim_env.state.current_nation))

                debug_path.append(action)

                acting_nation: Nation = sim_env.state.current_nation
                _, rewards = sim_env.step(action)
                path.append((node, action, acting_nation, rewards))

                if action.type == ActionType.RESOLVE_BATTLE:
                    # Chance node: route by the actual sampled outcome, not by action alone.
                    outcome_fp = state_fingerprint(sim_env.state)
                    outcomes = node.chance_children.setdefault(action, {})
                    if outcome_fp not in outcomes:
                        outcomes[outcome_fp] = Node(debug=self.debug)
                    node = outcomes[outcome_fp]
                else:
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

            # Backup MAX-N style: each player maximizes their own VP score. Change later to tets different scenarios #TODO
            G: dict[Nation, float] = value.copy()
            for nd, action, acting_nation, rewards in reversed(path):
                for nation, reward in rewards.items():
                    G[nation] = G.get(nation, 0.0) + float(reward)
                nd.N[action] += 1
                # Store full value vector for all nations
                for nation, g_val in G.items():
                    nd.W[action][nation] = nd.W[action].get(nation, 0.0) + float(g_val)
            
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
            # Print Q-values for all nations
            q_dict = root.Q(chosen)
            if isinstance(q_dict, dict):
                for nation, q_val in sorted(q_dict.items(), key=lambda x: x[0].name):
                    print(f"  {nation.name:12s}: Q={q_val:7.3f}")
            print("=" * 90)

        # Defer tree reuse until we know the real outcome from caller.
        # This is necessary for stochastic actions like RESOLVE_BATTLE.
        self._pending_root = root
        self._pending_action = chosen
        self.root = None
        self._expected_root_fp = None

        if self.debug:
            print()
            print("=" * 90)
            print("TREE REUSE")
            print("=" * 90)
            print("Root reuse deferred until confirm_step() is called with the real outcome.")
            print("=" * 90)

        return counts, chosen

    def confirm_step(self, resulting_state: GameState) -> None:
        """Call once, right after applying the action `run()` returned to the real
        environment, passing the real resulting state. Lets MCTS reuse the matching subtree
        next time instead of always rebuilding."""
        if self._pending_root is None:
            return

        fp = state_fingerprint(resulting_state)
        action = self._pending_action

        if action.type == ActionType.RESOLVE_BATTLE:
            outcomes = self._pending_root.chance_children.get(action, {})
            self.root = outcomes.get(fp)
        else:
            self.root = self._pending_root.children.get(action)

        self._expected_root_fp = fp if self.root is not None else None
        self._pending_root = None
        self._pending_action = None

    def print_mcts_diagnostics(
        self, 
        root: Node,
        chosen_action: Action,
        state: GameState,
        turn_number: int = 0,
    ) -> None:
        """Pretty-print MCTS decision diagnostics.
        
        Shows network values, policy, legal actions with Q/N/P stats, and selected action.
        """
        print("\n" + "=" * 90)
        print(f"STATE {turn_number}")
        acting_nation = state.current_nation
        print(f"Nation: {acting_nation.name if acting_nation else 'UNKNOWN'}")
        print(f"Phase: {state.phase.name if hasattr(state, 'phase') else 'UNKNOWN'}")
        
        # VALUE HEAD
        print("\nVALUE HEAD:")
        if self._last_root_network_values:
            for nation, value in sorted(self._last_root_network_values.items(), key=lambda x: x[0].name):
                print(f"    {nation.name:12s}: {value:6.2f}")
        
        # NETWORK POLICY
        print("\nNETWORK POLICY:")
        if self._last_root_network_policy:
            sorted_actions = sorted(
                self._last_root_network_policy.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for action, prior in sorted_actions[:10]:  # Top 10
                pct = prior * 100
                print(f"    {str(action):30s}: {pct:5.1f}%")
        
        # LEGAL ACTIONS
        print("\nLEGAL ACTIONS:")
        legal_actions = root.P.keys() if root and root.P else []
        for action in legal_actions:
            action_str = str(action)
            # Abbreviate if too long
            if len(action_str) > 30:
                action_str = action_str[:27] + "..."
            print(f"    {action_str}")
        
        # MCTS STATS
        print("\nMCTS:")
        print(f"    {'Action':30s}  {'Prior':>7s}  {'Visits':>7s}")
        if root and root.P:
            sorted_actions = sorted(
                legal_actions,
                key=lambda a: root.N.get(a, 0),
                reverse=True
            )
            for action in sorted_actions:
                action_str = str(action)
                if len(action_str) > 30:
                    action_str = action_str[:27] + "..."
                
                prior = root.P.get(action, 0.0)
                visits = root.N.get(action, 0)
                
                print(f"    {action_str:30s}  {prior:7.3f}  {visits:7d}")
                
                # Print Q values for all nations
                q_values = root.Q(action)
                if isinstance(q_values, dict):
                    for nation, q_val in sorted(q_values.items(), key=lambda x: x[0].name):
                        print(f"        {nation.name:12s}: Q={q_val:7.3f}")
        
        # SELECTED
        print(f"\nSELECTED:")
        action_str = str(chosen_action)
        print(f"    {action_str}")
        print("=" * 90 + "\n")


# TODO: CHECK EVAL AND TRAIN MODE BEFORE AND AFTER EVAL AND TRAIN OBVIOUSLY....
# TODO: May or not be importatnt thing about our policy but the values of priors precicted from the MCTS are like this 
# it kind of makes sense bu also not but like if i have the action move unit to t0 and move unit to t1 and end phase instead of being in the beginnign the odds of each one 0.333 or soemthign similar it's 0.5 for end phase and 0.25 for the other move actiosn lowkey kind of makes sense pq o action type escolhe entre as duas primeiro neh??
# TODO: Evaluate the final  vp my head is tryign to predict that or predict the amount of VP left to earn. which one is better???
# Valores na MCTS valors por player
# muLTIPLOS VALUES EACH PLAYER HAS IT'S OWN MCTS
# Choose what selection on the node of the enemy. You minimize the punctiation of the other adversary. WHen with multiple nodes which one to choose
# Minimizar os pontos daquele que está a frente o joagdor mais frente (Heuristic)
# 4 player game. I have two actions i can attak player 2 or player 3 both give me the same points, need to know even thoguh they both give same points for me it may penalize the other player. Paranoid search / Max my reward only / best reply search / Multi tree MCTS  