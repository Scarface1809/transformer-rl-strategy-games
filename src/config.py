from dataclasses import dataclass, field

# --- Environment ---
@dataclass
class EnvConfig:
    preset: str = "small"
    debug: bool = False

# --- Model ---
@dataclass
class ModelConfig:
    model_type: str = "simple"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

# --- Training ---
@dataclass
class TrainingConfig:
    # MCTS
    mcts_sims: int = 128             # More simulations = slower but better TODO: 1600 increase
    mcts_c_puct: float = 1.0          # Exploration vs exploitation
    
    lr: float = 0.0003                # Learning rate TODO: Increase to 0.001
    value_coef: float = 1.0           # Weight of value loss (scalar to balance with policy loss)
    max_grad_norm: float = 1.0        # Gradient clipping threshold
    batch_size: int = 256             # Batch size for training TODO: Increase to 4096 this is becayuse buffersize is way biger so idk.\
    num_train_epochs: int = 5        # How many epochs per buffer sample
    buffer_size: int = 50000          # Max transitions in buffer (also used for plotting)
    frequency_games: int = 100        # Train after every N self-play games #TODO Do 100 games or 200 not sure
    epochs: int = 3000               # Total training epochs (games) # TODO: Less epoches maybe?
    debug: bool = True                # Verbose logging

# --- Evaluation ---
@dataclass
class EvaluationConfig:
    num_games: int = 5               # Games per evaluation
    debug: bool = True               # Verbose

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
