from dataclasses import dataclass, field


# --- Environment ---
@dataclass
class EnvConfig:
    preset: str = "test3x3"


# --- Model ---
@dataclass
class ModelConfig:
    model_type: str = "simple"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


# --- Training ---
@dataclass
class TrainingConfig:
    num_games: int = 3000
    gamma: float = 0.95
    lambda_: float = 0.95
    lr: float = 5e-5    # TODO: Maybe tweak to 1e-4.
    # PPO-specific hyperparameters
    K_epochs: int = 4
    eps_clip: float = (
        0.1  # TODO: Maybe change this to 0.1 as well, so the policy doesnt update a lot with the randomness. Change this so the policy doesnt change too much in one update. Making it too low might cuase collapse.
    )
    value_coef: float = 0.5
    entropy_coef: float = (
        0.005  # TODO: Change this to 0.01 original # Also change this to encurage or discoruage exploration. CHange it even more probably, so it doesnt encourage those little exploration patterns that are happening.
    )
    debug: bool = True


@dataclass
class EvaluateConfig:
    num_games: int = 10
    frequency: int = 100  # Evaluate every *frequency* training games
    debug: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluateConfig = field(default_factory=EvaluateConfig)
