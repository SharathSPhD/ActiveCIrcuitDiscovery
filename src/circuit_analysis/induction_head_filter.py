"""
Induction Head Filter for Circuit Discovery

Filters discovered features to focus on induction head patterns that are relevant
for semantic understanding like "Golden Gate Bridge -> San Francisco".
"""

from typing import List, Set, Tuple, Dict, Any
import torch
from dataclasses import dataclass
from ..core.data_structures import CircuitFeature
from .real_tracer import RealCircuitTracer


@dataclass
class InductionHeadCandidate:
    """A circuit feature that might be part of an induction head."""
    feature: CircuitFeature
    induction_score: float
    semantic_relevance: float
    layer_position: str  # 'early', 'middle', 'late'
    

class InductionHeadFilter:
    """Filters circuit features to identify potential induction head components."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b", transcoder_set: str = "gemma"):
        self.model_name = model_name
        self.transcoder_set = transcoder_set
        self.tracer = RealCircuitTracer(model_name=model_name, transcoder_set=transcoder_set)
        
        # Layer ranges for Gemma-2-2B (26 layers total)
        self.early_layers = list(range(0, 6))      # Layers 0-5
        self.middle_layers = list(range(6, 20))    # Layers 6-19 (likely induction heads)
        self.late_layers = list(range(20, 26))     # Layers 20-25
        
    def filter_for_induction_heads(
        self, 
        all_features: List[CircuitFeature],
        max_features: int = 500
    ) -> List[InductionHeadCandidate]:
        """
        Filter features to focus on induction head candidates.
        
        Args:
            all_features: All discovered features (e.g., the 4949 features)
            max_features: Maximum number of features to return
            
        Returns:
            List of induction head candidates, ranked by relevance
        """
        candidates = []
        
        for feature in all_features:
            # Calculate induction head likelihood
            induction_score = self._calculate_induction_score(feature)
            semantic_relevance = self._calculate_semantic_relevance(feature)
            layer_position = self._get_layer_position(feature.layer_idx)
            
            # Only keep features with some induction potential
            if induction_score > 0.1 or semantic_relevance > 0.2:
                candidate = InductionHeadCandidate(
                    feature=feature,
                    induction_score=induction_score,
                    semantic_relevance=semantic_relevance,
                    layer_position=layer_position
                )
                candidates.append(candidate)
        
        # Sort by combined score (induction + semantic relevance)
        candidates.sort(
            key=lambda c: c.induction_score * 0.6 + c.semantic_relevance * 0.4,
            reverse=True
        )
        
        return candidates[:max_features]
    
    def _calculate_induction_score(self, feature: CircuitFeature) -> float:
        """
        Calculate how likely this feature is part of an induction head.
        
        Induction heads typically:
        - Are in middle layers (6-19 for Gemma-2-2B)
        - Have high activation strength
        - Are MLP components (which we already filter for)
        """
        score = 0.0
        
        # Layer position bonus (middle layers most likely)
        if feature.layer_idx in self.middle_layers:
            score += 0.6  # Strong bonus for middle layers
        elif feature.layer_idx in range(4, 22):  # Extended middle range
            score += 0.3  # Moderate bonus for near-middle
        else:
            score += 0.1  # Small bonus for other layers
            
        # Activation strength bonus (normalize by layer)
        # Higher activation = more important feature
        activation_bonus = min(feature.activation_strength / 5.0, 0.4)
        score += activation_bonus
        
        return min(score, 1.0)
    
    def _calculate_semantic_relevance(self, feature: CircuitFeature) -> float:
        """
        Calculate how relevant this feature is for semantic understanding.
        
        This is a heuristic based on:
        - Feature activation strength
        - Layer position
        - Feature ID patterns (some ranges might be more semantic)
        """
        score = 0.0
        
        # High activation features are more likely to be semantically relevant
        if feature.activation_strength > 3.0:
            score += 0.4
        elif feature.activation_strength > 1.5:
            score += 0.2
        else:
            score += 0.1
            
        # Middle-to-late layers often handle more semantic content
        if feature.layer_idx >= 8:
            score += 0.3
        elif feature.layer_idx >= 4:
            score += 0.2
        else:
            score += 0.1
            
        # Feature ID heuristic (some ranges might be more semantic)
        # This is speculative but based on transcoder organization
        if 100 <= feature.feature_id <= 8000:  # Mid-range features
            score += 0.2
        elif feature.feature_id < 100 or feature.feature_id > 15000:  # Edge features
            score += 0.1
        else:
            score += 0.15
            
        return min(score, 1.0)
    
    def _get_layer_position(self, layer_idx: int) -> str:
        """Get the position category of the layer."""
        if layer_idx in self.early_layers:
            return "early"
        elif layer_idx in self.middle_layers:
            return "middle"
        else:
            return "late"
    
    def filter_semantic_test_features(
        self,
        prompt: str,
        max_features: int = 200,
        threshold: float = 0.2
    ) -> List[InductionHeadCandidate]:
        """
        Discover features for a specific semantic test and filter for induction heads.
        
        Args:
            prompt: Test prompt (e.g., "The Golden Gate Bridge is located in")
            max_features: Maximum features to return
            threshold: Minimum activation threshold
            
        Returns:
            Filtered induction head candidates for this prompt
        """
        # Discover all features for this prompt
        all_features = self.tracer.discover_active_features(prompt, threshold=threshold)
        
        # Filter for induction head candidates
        candidates = self.filter_for_induction_heads(all_features, max_features)
        
        return candidates
    
    def get_summary_stats(self, candidates: List[InductionHeadCandidate]) -> Dict[str, Any]:
        """Get summary statistics for the filtered candidates."""
        if not candidates:
            return {"total": 0}
            
        return {
            "total": len(candidates),
            "by_layer_position": {
                pos: len([c for c in candidates if c.layer_position == pos])
                for pos in ["early", "middle", "late"]
            },
            "avg_induction_score": sum(c.induction_score for c in candidates) / len(candidates),
            "avg_semantic_relevance": sum(c.semantic_relevance for c in candidates) / len(candidates),
            "layer_distribution": {
                f"layer_{layer}": len([c for c in candidates if c.feature.layer_idx == layer])
                for layer in sorted(set(c.feature.layer_idx for c in candidates))
            },
            "top_layers": sorted(
                set(c.feature.layer_idx for c in candidates),
                key=lambda l: len([c for c in candidates if c.feature.layer_idx == l]),
                reverse=True
            )[:5]
        }