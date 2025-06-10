# YorK_RP Configuration Management
# Proper configuration handling for the Active Inference Circuit Discovery project

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml
import json
from enum import Enum

class DeviceType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

class InterventionType(Enum):
    ABLATION = "ablation"
    ACTIVATION_PATCHING = "activation_patching"
    MEAN_ABLATION = "mean_ablation"

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    name: str = "gpt2-small"
    device: DeviceType = DeviceType.AUTO
    max_context_length: int = 1024
    use_cache: bool = True

@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder analysis."""
    enabled: bool = True
    auto_discover_layers: bool = True                    # Auto-discover active layers
    target_layers: List[int] = field(default_factory=list)  # Empty by default - auto-populated
    layer_search_range: List[int] = field(default_factory=lambda: [0, -1])  # [start, end] (-1 = last)
    activation_threshold: float = 0.05
    max_features_per_layer: int = 20
    neuronpedia_source: bool = True
    sample_inputs_for_layer_discovery: List[str] = field(default_factory=list)
    auto_discover_min_layers: int = 6
    auto_discover_max_layers: int = 8
    auto_discover_layer_ratio: float = 0.5
    fallback_sae_feature_multiplier: int = 4
    fallback_sae_weight_scale: float = 0.1

@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference agent."""
    enabled: bool = True
    epistemic_weight: float = 0.7
    exploration_weight: float = 0.6
    convergence_threshold: float = 0.15
    max_interventions: int = 20
    use_pymdp: bool = True
    baseline_intervention_multiplier: float = 3.0
    baseline_convergence_std_threshold: float = 0.05
    baseline_convergence_recent_effects_count: int = 5

@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    name: str = "golden_gate_bridge_discovery"
    output_dir: str = "experiment_results"
    max_parallel_jobs: int = 1
    save_intermediate_results: bool = True
    generate_visualizations: bool = True

@dataclass
class ResearchQuestionConfig:
    """Configuration for research question validation."""
    rq1_correspondence_target: float = 70.0  # Percentage
    rq2_efficiency_target: float = 30.0      # Percentage improvement
    rq3_predictions_target: int = 3          # Number of validated predictions
    prediction_validation_confidence_threshold: float = 0.7
    
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file_output: bool = True
    console_output: bool = True
    max_file_size_mb: int = 10

@dataclass
class CompleteConfig:
    """Complete configuration for the YorK_RP experiment."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    active_inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    research_questions: ResearchQuestionConfig = field(default_factory=ResearchQuestionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "default_config.yaml"
        self._config = None
        
    def load_config(self) -> CompleteConfig:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            return self._load_from_file()
        else:
            return self._create_default_config()
    
    def _load_from_file(self) -> CompleteConfig:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to dataclass with validation
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteConfig:
        """Convert dictionary to CompleteConfig with validation."""
        try:
            model_config = ModelConfig(**config_dict.get('model', {}))
            sae_config = SAEConfig(**config_dict.get('sae', {}))
            ai_config = ActiveInferenceConfig(**config_dict.get('active_inference', {}))
            exp_config = ExperimentConfig(**config_dict.get('experiment', {}))
            rq_config = ResearchQuestionConfig(**config_dict.get('research_questions', {}))
            log_config = LoggingConfig(**config_dict.get('logging', {}))
            
            return CompleteConfig(
                model=model_config,
                sae=sae_config,
                active_inference=ai_config,
                experiment=exp_config,
                research_questions=rq_config,
                logging=log_config
            )
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
    
    def _create_default_config(self) -> CompleteConfig:
        """Create and save default configuration."""
        config = CompleteConfig()
        self.save_config(config)
        return config
    
    def save_config(self, config: CompleteConfig):
        """Save configuration to file."""
        config_dict = {
            'model': {
                'name': config.model.name,
                'device': config.model.device.value,
                'max_context_length': config.model.max_context_length,
                'use_cache': config.model.use_cache
            },
            'sae': {
                'enabled': config.sae.enabled,
                'auto_discover_layers': config.sae.auto_discover_layers,
                'target_layers': config.sae.target_layers,
                'layer_search_range': config.sae.layer_search_range,
                'activation_threshold': config.sae.activation_threshold,
                'max_features_per_layer': config.sae.max_features_per_layer,
                'neuronpedia_source': config.sae.neuronpedia_source,
                'sample_inputs_for_layer_discovery': config.sae.sample_inputs_for_layer_discovery,
                'auto_discover_min_layers': config.sae.auto_discover_min_layers,
                'auto_discover_max_layers': config.sae.auto_discover_max_layers,
                'auto_discover_layer_ratio': config.sae.auto_discover_layer_ratio,
                'fallback_sae_feature_multiplier': config.sae.fallback_sae_feature_multiplier,
                'fallback_sae_weight_scale': config.sae.fallback_sae_weight_scale
            },
            'active_inference': {
                'enabled': config.active_inference.enabled,
                'epistemic_weight': config.active_inference.epistemic_weight,
                'exploration_weight': config.active_inference.exploration_weight,
                'convergence_threshold': config.active_inference.convergence_threshold,
                'max_interventions': config.active_inference.max_interventions,
                'use_pymdp': config.active_inference.use_pymdp,
                'baseline_intervention_multiplier': config.active_inference.baseline_intervention_multiplier,
                'baseline_convergence_std_threshold': config.active_inference.baseline_convergence_std_threshold,
                'baseline_convergence_recent_effects_count': config.active_inference.baseline_convergence_recent_effects_count
            },
            'experiment': {
                'name': config.experiment.name,
                'output_dir': config.experiment.output_dir,
                'max_parallel_jobs': config.experiment.max_parallel_jobs,
                'save_intermediate_results': config.experiment.save_intermediate_results,
                'generate_visualizations': config.experiment.generate_visualizations
            },
            'research_questions': {
                'rq1_correspondence_target': config.research_questions.rq1_correspondence_target,
                'rq2_efficiency_target': config.research_questions.rq2_efficiency_target,
                'rq3_predictions_target': config.research_questions.rq3_predictions_target,
                'prediction_validation_confidence_threshold': config.research_questions.prediction_validation_confidence_threshold
            },
            'logging': {
                'level': config.logging.level,
                'file_output': config.logging.file_output,
                'console_output': config.logging.console_output,
                'max_file_size_mb': config.logging.max_file_size_mb
            }
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate_config(self, config: CompleteConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate Active Inference weights
        if not 0 <= config.active_inference.epistemic_weight <= 1:
            issues.append("epistemic_weight must be between 0 and 1")
        
        if not 0 <= config.active_inference.exploration_weight <= 1:
            issues.append("exploration_weight must be between 0 and 1")
        
        # Validate SAE configuration
        if config.sae.auto_discover_layers and config.sae.layer_search_range:
            start, end = config.sae.layer_search_range
            if start < 0:
                issues.append("layer_search_range start must be >= 0")
        elif not config.sae.auto_discover_layers and not config.sae.target_layers:
            issues.append("target_layers cannot be empty when auto_discover_layers is False")
        
        if not 0 < config.sae.activation_threshold < 1:
            issues.append("sae.activation_threshold must be between 0 and 1")
        
        # Validate research question targets
        if config.research_questions.rq1_correspondence_target <= 0:
            issues.append("rq1_correspondence_target must be positive")
            
        if config.research_questions.rq2_efficiency_target <= 0:
            issues.append("rq2_efficiency_target must be positive")
            
        if config.research_questions.rq3_predictions_target <= 0:
            issues.append("rq3_predictions_target must be positive")
        
        return issues

# Global configuration instance
_config_manager = None

def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config(config_path: Optional[Path] = None) -> CompleteConfig:
    """Get current configuration."""
    manager = get_config_manager(config_path)
    return manager.load_config()
