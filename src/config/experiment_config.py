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
    include_error_nodes: bool = True                     # Add error nodes for unexplained variance
    max_graph_nodes: int = 100                          # Maximum nodes before pruning

@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference agent."""
    enabled: bool = True
    epistemic_weight: float = 0.7
    exploration_weight: float = 0.6
    convergence_threshold: float = 0.15
    max_interventions: int = 20
    use_pymdp: bool = True

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
    
@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file_output: bool = True
    console_output: bool = True
    max_file_size_mb: int = 10

@dataclass
class StatisticalValidationConfig:
    """Configuration for enhanced statistical validation."""
    enabled: bool = True
    significance_level: float = 0.05
    bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    multiple_comparisons_correction: str = "bonferroni"

@dataclass
class PredictionValidationConfig:
    """Configuration for enhanced prediction validation."""
    enabled: bool = True
    min_sample_size: int = 15
    validation_threshold: float = 0.8
    cross_validation_folds: int = 5

@dataclass
class VisualizationConfig:
    """Configuration for enhanced visualizations."""
    enhanced_plots: bool = True
    interactive_dashboards: bool = True
    publication_ready: bool = True
    save_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "html"])

@dataclass
class CompleteConfig:
    """Complete configuration for the YorK_RP experiment."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    active_inference: ActiveInferenceConfig = field(default_factory=ActiveInferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    research_questions: ResearchQuestionConfig = field(default_factory=ResearchQuestionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    statistical_validation: StatisticalValidationConfig = field(default_factory=StatisticalValidationConfig)
    prediction_validation: PredictionValidationConfig = field(default_factory=PredictionValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

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
            stat_val_config = StatisticalValidationConfig(**config_dict.get('statistical_validation', {}))
            pred_val_config = PredictionValidationConfig(**config_dict.get('prediction_validation', {}))
            vis_config = VisualizationConfig(**config_dict.get('visualization', {}))
            
            return CompleteConfig(
                model=model_config,
                sae=sae_config,
                active_inference=ai_config,
                experiment=exp_config,
                research_questions=rq_config,
                logging=log_config,
                statistical_validation=stat_val_config,
                prediction_validation=pred_val_config,
                visualization=vis_config
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
                'neuronpedia_source': config.sae.neuronpedia_source
            },
            'active_inference': {
                'enabled': config.active_inference.enabled,
                'epistemic_weight': config.active_inference.epistemic_weight,
                'exploration_weight': config.active_inference.exploration_weight,
                'convergence_threshold': config.active_inference.convergence_threshold,
                'max_interventions': config.active_inference.max_interventions,
                'use_pymdp': config.active_inference.use_pymdp
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
                'rq3_predictions_target': config.research_questions.rq3_predictions_target
            },
            'logging': {
                'level': config.logging.level,
                'file_output': config.logging.file_output,
                'console_output': config.logging.console_output,
                'max_file_size_mb': config.logging.max_file_size_mb
            },
            'statistical_validation': {
                'enabled': config.statistical_validation.enabled,
                'significance_level': config.statistical_validation.significance_level,
                'bootstrap_samples': config.statistical_validation.bootstrap_samples,
                'confidence_level': config.statistical_validation.confidence_level,
                'multiple_comparisons_correction': config.statistical_validation.multiple_comparisons_correction
            },
            'prediction_validation': {
                'enabled': config.prediction_validation.enabled,
                'min_sample_size': config.prediction_validation.min_sample_size,
                'validation_threshold': config.prediction_validation.validation_threshold,
                'cross_validation_folds': config.prediction_validation.cross_validation_folds
            },
            'visualization': {
                'enhanced_plots': config.visualization.enhanced_plots,
                'interactive_dashboards': config.visualization.interactive_dashboards,
                'publication_ready': config.visualization.publication_ready,
                'save_formats': config.visualization.save_formats
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

def get_enhanced_config() -> CompleteConfig:
    """Get enhanced configuration with all advanced features enabled."""
    enhanced_config_path = Path(__file__).parent / "enhanced_config.yaml"
    return get_config(enhanced_config_path)
