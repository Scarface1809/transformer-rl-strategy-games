from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Dict

import torch

from envs.core.enums import Nation
from envs.core.entities import Action


@dataclass
class TrajectoryExample:
    """Single AlphaZero training example."""
    global_feats: torch.Tensor
    tile_feats: torch.Tensor
    unit_feats: torch.Tensor
    index_to_unit_id: list[int]
    masks: Dict[str, torch.Tensor]

    acting_nation: Nation
    value: Dict[Nation, float] # Value target (VP gained from this state to game end)
    pi: Dict[Action, float] # Policy target (visit counts from MCTS)


class ReplayBuffer:
    """Stores self-play games for training."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self._games: deque[list[TrajectoryExample]] = deque()
        self._num_steps = 0

    def add_game(self, examples: list[TrajectoryExample]) -> None:
        self._games.append(examples)
        self._num_steps += len(examples)
        while self._num_steps > self.max_steps and len(self._games) > 1:
            old = self._games.popleft()
            self._num_steps -= len(old)

    def __len__(self) -> int:
        return self._num_steps
    
    @property
    def num_games(self) -> int:
        return len(self._games)

    def all_examples(self) -> list[TrajectoryExample]:
        out: list[TrajectoryExample] = []
        for g in self._games:
            out.extend(g)
        return out

    def epoch_batches(self, batch_size: int) -> list[list[TrajectoryExample]]:
        data = self.all_examples()
        random.shuffle(data)

        return [
            data[i:i + batch_size]
            for i in range(0, len(data), batch_size)
        ]