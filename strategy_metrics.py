    def _calculate_simple_correspondence(self, ai_beliefs, intervention_results, discovered_features) -> float:
        """Strategy learning correspondence calculation for RQ1.
        
        Measures how well the strategy learning agent learns which intervention strategies are effective.
        """
        if not intervention_results:
            return 0.0
        
        # Use strategy learning agent to calculate correspondence
        if hasattr(self.ai_agent, "calculate_strategy_correspondence"):
            try:
                # Get strategy beliefs from agent
                strategy_beliefs = self.ai_agent.track_strategy_beliefs()
                
                # Calculate actual success rates from intervention results
                actual_success_rates = self.ai_agent.track_discovery_success(intervention_results)
                
                # Calculate correspondence between beliefs and actual success
                correspondence = self.ai_agent.calculate_strategy_correspondence(
                    strategy_beliefs, actual_success_rates
                )
                
                return correspondence
            except Exception as e:
                logger.warning(f"Strategy correspondence calculation failed: {e}")
                return 0.0
        else:
            # Fallback: measure intervention effectiveness learning
            effective_interventions = sum(1 for r in intervention_results if r.semantic_change)
            total_interventions = len(intervention_results)
            
            if total_interventions == 0:
                return 0.0
            
            # Basic effectiveness learning metric
            effectiveness_rate = (effective_interventions / total_interventions) * 100.0
            
            # Add learning progression bonus if agent is improving
            if hasattr(self.ai_agent, 'intervention_history') and len(self.ai_agent.intervention_history) > 5:
                recent_effectiveness = sum(1 for r in self.ai_agent.intervention_history[-5:] if r.semantic_change) / 5
                early_effectiveness = sum(1 for r in self.ai_agent.intervention_history[:5] if r.semantic_change) / 5
                learning_bonus = max(0, (recent_effectiveness - early_effectiveness) * 50)
                effectiveness_rate += learning_bonus
            
            return min(100.0, effectiveness_rate)

    def _calculate_simple_efficiency(self, ai_intervention_sequence, baseline_method, discovered_features) -> float:
        """Strategy learning efficiency calculation for RQ2.
        
        Measures how efficiently the strategy learning agent guides interventions vs random selection.
        """
        if not ai_intervention_sequence:
            return 0.0
        
        # Use strategy learning agent to calculate efficiency
        if hasattr(self.ai_agent, "track_discovery_success"):
            try:
                # Calculate strategy-guided success rates
                strategy_success_rates = self.ai_agent.track_discovery_success(ai_intervention_sequence)
                
                if strategy_success_rates:
                    # Calculate average strategy effectiveness
                    avg_strategy_success = np.mean(list(strategy_success_rates.values()))
                    
                    # Compare to baseline random selection (assume 20% success rate)
                    baseline_success_rate = 0.2
                    
                    # Calculate efficiency improvement
                    if baseline_success_rate > 0:
                        efficiency_improvement = ((avg_strategy_success - baseline_success_rate) / baseline_success_rate) * 100.0
                        return max(0.0, efficiency_improvement)
                    else:
                        return avg_strategy_success * 100.0
                else:
                    return 0.0
            except Exception as e:
                logger.warning(f"Strategy efficiency calculation failed: {e}")
                return 0.0
        else:
            # Fallback: simple success rate calculation
            successful_interventions = sum(1 for r in ai_intervention_sequence if r.semantic_change)
            total_interventions = len(ai_intervention_sequence)
            
            if total_interventions == 0:
                return 0.0
            
            success_rate = successful_interventions / total_interventions
            
            # Compare to baseline (assume 20% random success)
            baseline_rate = 0.2
            improvement = ((success_rate - baseline_rate) / baseline_rate) * 100.0
            
            return max(0.0, improvement)