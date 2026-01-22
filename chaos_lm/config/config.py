# config/config.py
"""
CHAOS-LM Configuration Module
Defines all configuration classes for the anti-alignment system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import json
import os
from datetime import datetime


class TrainingMode(str, Enum):
    """Training modes for anti-alignment"""
    REVERSE_LOSS = "reverse_loss"
    SHIFTED_LABEL = "shifted_label"
    ENTROPY_MAX = "entropy_max"
    GARBAGE_CORPUS = "garbage_corpus"
    HYBRID = "hybrid"


class DegradationStyle(str, Enum):
    """Styles for creative text degradation"""
    ALIEN_SYNTAX = "alien_syntax"
    POETIC_NONSENSE = "poetic_nonsense"
    GLITCH_TALK = "glitch_talk"
    FAKE_PROFOUND = "fake_profound"
    DREAM_LOGIC = "dream_logic"


@dataclass
class ModelConfig:
    """Model configuration"""
    # Development model (fast iteration)
    dev_model_name: str = "gpt2"
    # Production model (research-grade)
    prod_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Current mode
    use_production: bool = False
    # Model loading settings
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    @property
    def model_name(self) -> str:
        return self.prod_model_name if self.use_production else self.dev_model_name


@dataclass
class DegradationConfig:
    """Degradation control parameters"""
    # Main degradation level [0, 1]
    degradation_level: float = 0.5
    # Learning rate multiplier for anti-training
    lr_multiplier: float = 1.0
    # Noise injection standard deviation
    noise_sigma: float = 0.01
    # Ratio of layers to freeze (protect from corruption)
    freeze_ratio: float = 0.0
    # Entropy floor to prevent recovery
    entropy_floor: float = 0.5
    # Token shift amount for shifted-label mode
    token_shift: int = 2
    # Garbage corpus shuffle probability
    shuffle_prob: float = 0.3
    # Glitch token injection rate
    glitch_injection_rate: float = 0.05


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training mode
    mode: TrainingMode = TrainingMode.REVERSE_LOSS
    # Hybrid mode weights
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "reverse_loss": 0.4,
        "shifted_label": 0.2,
        "entropy_max": 0.3,
        "garbage_corpus": 0.1
    })
    # Basic training params
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use epochs
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Sequence settings
    max_seq_length: int = 512
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    # Reproducibility
    seed: int = 42
    # Degradation schedule
    degradation_schedule: str = "linear"  # linear, cosine, step
    # Noise re-injection interval
    noise_reinjection_steps: int = 200


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    streaming: bool = True
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    # Preprocessing
    remove_empty: bool = True
    min_length: int = 50


@dataclass
class MetricsConfig:
    """Metrics tracking configuration"""
    # Which metrics to track
    track_perplexity: bool = True
    track_entropy: bool = True
    track_zipf_deviation: bool = True
    track_coherence: bool = True
    track_inversion_rate: bool = True
    # Additional metrics
    track_token_diversity: bool = True
    track_repetition_rate: bool = True
    track_syntax_errors: bool = True
    # Evaluation frequency
    eval_frequency: int = 100
    # Coherence judge settings
    use_external_judge: bool = False
    judge_api_key: Optional[str] = None
    judge_model: str = "llama-3.1-70b-versatile"
    # Perplexity baseline
    baseline_perplexity: float = 20.0


@dataclass
class InferenceConfig:
    """Inference configuration"""
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 128
    do_sample: bool = True
    repetition_penalty: float = 1.1
    # Degradation style
    style: DegradationStyle = DegradationStyle.POETIC_NONSENSE
    # Safety markers
    add_unreliable_marker: bool = True
    unreliable_prefix: str = "⚠️ [UNRELIABLE OUTPUT] "
    # Glitch tokens
    glitch_tokens: List[str] = field(default_factory=lambda: [
        "◊", "∆", "≋", "⌘", "⍟", "◈", "⌬", "⏣"
    ])
    add_glitch_tokens: bool = True
    glitch_frequency: float = 0.1


@dataclass
class CheckpointConfig:
    """Checkpoint management configuration"""
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 5
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_metrics: bool = True
    # Naming
    prefix: str = "chaos_lm"
    # Version control
    include_timestamp: bool = True
    include_metrics: bool = True


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    # Rate limiting
    rate_limit: int = 100  # requests per minute
    # CORS
    allow_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class UIConfig:
    """Streamlit UI configuration"""
    page_title: str = "CHAOS-LM: Anti-Alignment Laboratory"
    page_icon: str = "🌀"
    layout: str = "wide"
    # Theme
    theme_base: str = "dark"
    primary_color: str = "#FF4B4B"


@dataclass
class ChaosConfig:
    """Master configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    api: APIConfig = field(default_factory=APIConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    # Experiment tracking
    experiment_name: str = "chaos_lm_experiment"
    run_id: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "chaos-lm"
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def _to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [_to_dict(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            return obj
        return _to_dict(self)
    
    def save(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ChaosConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChaosConfig':
        """Create config from dictionary"""
        model = ModelConfig(**data.get('model', {}))
        degradation = DegradationConfig(**data.get('degradation', {}))
        
        training_data = data.get('training', {})
        if 'mode' in training_data and isinstance(training_data['mode'], str):
            training_data['mode'] = TrainingMode(training_data['mode'])
        training = TrainingConfig(**training_data)
        
        data_config = DataConfig(**data.get('data', {}))
        metrics = MetricsConfig(**data.get('metrics', {}))
        
        inference_data = data.get('inference', {})
        if 'style' in inference_data and isinstance(inference_data['style'], str):
            inference_data['style'] = DegradationStyle(inference_data['style'])
        inference = InferenceConfig(**inference_data)
        
        checkpoint = CheckpointConfig(**data.get('checkpoint', {}))
        api = APIConfig(**data.get('api', {}))
        ui = UIConfig(**data.get('ui', {}))
        
        return cls(
            model=model,
            degradation=degradation,
            training=training,
            data=data_config,
            metrics=metrics,
            inference=inference,
            checkpoint=checkpoint,
            api=api,
            ui=ui,
            experiment_name=data.get('experiment_name', 'chaos_lm_experiment'),
            run_id=data.get('run_id'),
            use_wandb=data.get('use_wandb', False),
            wandb_project=data.get('wandb_project', 'chaos-lm')
        )


def get_default_config() -> ChaosConfig:
    """Get default configuration"""
    return ChaosConfig()


def get_dev_config() -> ChaosConfig:
    """Get development configuration (GPT-2 based)"""
    config = ChaosConfig()
    config.model.use_production = False
    config.training.batch_size = 8
    config.training.max_steps = 1000
    return config


def get_prod_config() -> ChaosConfig:
    """Get production configuration (Llama-3 based)"""
    config = ChaosConfig()
    config.model.use_production = True
    config.model.load_in_8bit = True
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 8
    return config