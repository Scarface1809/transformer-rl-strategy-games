from dataclasses import dataclass, field


# --- Environment ---
@dataclass
class EnvConfig:
    preset: str = "hispania"  # board: "hispania"


# --- Model ---
@dataclass
class ModelConfig:
    model_type: str = "simple"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1


# --- Training ---
@dataclass
class TrainingConfig:
    num_games: int = 3000
    gamma: float = 0.99
    lr: float = 1e-3
    debug: bool = True


@dataclass
class EvaluateConfig:
    num_games: int = 25
    frequency: int = 100  # Evaluate every *frequency* training games
    debug: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluateConfig = field(default_factory=EvaluateConfig)
