"""
Singleton Model Manager for Circuit Discovery

Ensures only one model instance is loaded and shared across all components.
Configures local-only mode to avoid unnecessary downloads.
Handles persistent cache location for droplet recreation scenarios.
"""

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from circuit_tracer import ReplacementModel


class ModelManager:
    """Singleton manager for ReplacementModel to avoid duplicate loading."""
    
    _instance = None
    _model = None
    _tokenizer = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_cache_location()
            self._configure_offline_mode()
            ModelManager._initialized = True
    
    def _setup_cache_location(self):
        """Configure persistent cache location if available."""
        # Check for persistent volume mount
        persistent_paths = ['/mnt/volume', '/data', '/persistent']
        
        for path in persistent_paths:
            if Path(path).exists() and os.access(path, os.W_OK):
                cache_dir = Path(path) / 'huggingface_cache'
                cache_dir.mkdir(exist_ok=True)
                
                # Set HuggingFace cache to persistent location
                os.environ['HF_HOME'] = str(cache_dir)
                os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir / 'hub')
                os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'transformers')
                
                print(f"Using persistent cache: {cache_dir}")
                return
        
        print("Using default cache (will be lost on droplet recreation)")
    
    def _configure_offline_mode(self):
        """Configure local-only mode to avoid network calls."""
        # Force HuggingFace to use local files only
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Disable update checks
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    
    def get_model(self, model_name: str = "google/gemma-2-2b", 
                  transcoder_set: str = "gemma") -> ReplacementModel:
        """Get or create the shared model instance."""
        if self._model is None:
            print(f"Loading model {model_name} (one-time initialization)")
            
            try:
                # Try local-only first
                self._model = ReplacementModel.from_pretrained(
                    model_name=model_name,
                    transcoder_set=transcoder_set,
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                    local_files_only=True  # Force local usage
                )
                print("Loaded from local cache")
                
            except Exception as e:
                if "local_files_only" in str(e) or "offline" in str(e):
                    print("Local files not found, downloading once...")
                    # Temporarily allow downloads for initial setup
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                    
                    self._model = ReplacementModel.from_pretrained(
                        model_name=model_name,
                        transcoder_set=transcoder_set,
                        device=torch.device("cuda"),
                        dtype=torch.bfloat16
                    )
                    
                    # Re-enable offline mode
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    print("Downloaded and cached for future use")
                else:
                    raise e
        
        return self._model
    
    def get_tokenizer(self, model_name: str = "google/gemma-2-2b") -> AutoTokenizer:
        """Get or create the shared tokenizer instance."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
            except:
                # If local files not available, download once
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        
        return self._tokenizer
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is already loaded."""
        return self._model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": getattr(self._model, 'model_name', 'unknown'),
            "layers": len(self._model.transcoders) if hasattr(self._model, 'transcoders') else 0,
            "features_per_layer": getattr(self._model, 'd_transcoder', 0),
            "device": str(self._model.device) if hasattr(self._model, 'device') else 'unknown'
        }


# Global instance
model_manager = ModelManager()
