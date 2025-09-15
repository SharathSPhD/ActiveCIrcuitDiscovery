#!/usr/bin/env python3

from src.circuit_analysis.induction_head_filter import InductionHeadFilter

def main():
    filter = InductionHeadFilter()
    print("=== INDUCTION HEAD FILTER TEST ===")
    candidates = filter.filter_semantic_test_features("The Golden Gate Bridge is located in", max_features=100)
    print(f"Filtered to {len(candidates)} induction head candidates from 4949 total features")

    stats = filter.get_summary_stats(candidates)
    print("\nSummary Statistics:")
    print(f"- Total candidates: {stats['total']}")
    print(f"- By position: {stats['by_layer_position']}")
    print(f"- Avg induction score: {stats['avg_induction_score']:.3f}")
    print(f"- Avg semantic relevance: {stats['avg_semantic_relevance']:.3f}")
    print(f"- Top layers: {stats['top_layers']}")

    print("\nTop 10 candidates:")
    for i, candidate in enumerate(candidates[:10]):
        f = candidate.feature
        print(f"{i+1:2d}. L{f.layer_idx:2d}F{f.feature_id:5d}: ind={candidate.induction_score:.3f}, sem={candidate.semantic_relevance:.3f}, act={f.activation_strength:.3f} ({candidate.layer_position})")

if __name__ == "__main__":
    main()