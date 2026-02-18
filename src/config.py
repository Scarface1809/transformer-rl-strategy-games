from dataclasses import dataclass, field

# --- Environment ---
@dataclass
class EnvConfig:
    board: str = "random"  # "random" or "hispania"
    num_tiles: int = 25
    num_nations: int = 4
    initial_units: int = 10
    max_turns: int = 20
    fixed_map: bool = True
    seed: str = None

# --- Model ---
@dataclass
class ModelConfig:
    model_type: str = "simple"  # "simple" or "transformer"
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2

# --- Training ---
# 100 Training - 10 Test ... Repeat
@dataclass
class TrainingConfig:
    num_episodes: int = 2000
    num_games: int = 100
    gamma: float = 0.99
    lr: float = 1e-3
    device: str = None
    debug: bool = True

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
