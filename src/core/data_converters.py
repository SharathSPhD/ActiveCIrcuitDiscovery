"""
Data format converters for consistent data structures across the system.

Ensures proper conversion between:
- Circuit-tracer outputs (Dict[int, List[CircuitFeature]]) 
- Active Inference inputs (various formats)
- Experiment runner requirements (List[CircuitFeature])
"""

from typing import Dict, List, Any, Union
from .data_structures import CircuitFeature


class DataConverter:
    """Handles all data format conversions in the system."""
    
    @staticmethod
    def features_dict_to_list(features_dict: Dict[int, List[CircuitFeature]]) -> List[CircuitFeature]:
        """Convert layered features dict to flat list."""
        flat_list = []
        for layer_features in features_dict.values():
            flat_list.extend(layer_features)
        return flat_list
    
    @staticmethod 
    def features_list_to_dict(features_list: List[CircuitFeature]) -> Dict[int, List[CircuitFeature]]:
        """Convert flat features list to layered dict."""
        by_layer = {}
        for feature in features_list:
            if feature.layer_idx not in by_layer:
                by_layer[feature.layer_idx] = []
            by_layer[feature.layer_idx].append(feature)
        return by_layer
    
    @staticmethod
    def normalize_features_input(features: Union[List[CircuitFeature], Dict[int, List[CircuitFeature]]]) -> Dict[int, List[CircuitFeature]]:
        """Normalize any features input to standard Dict[int, List[CircuitFeature]] format."""
        if isinstance(features, list):
            return DataConverter.features_list_to_dict(features)
        elif isinstance(features, dict):
            return features
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")
    
    @staticmethod
    def get_feature_count(features: Union[List[CircuitFeature], Dict[int, List[CircuitFeature]]]) -> int:
        """Get total feature count regardless of input format."""
        if isinstance(features, list):
            return len(features)
        elif isinstance(features, dict):
            return sum(len(layer_feats) for layer_feats in features.values())
        else:
            return 0


# Global converter instance
data_converter = DataConverter()
