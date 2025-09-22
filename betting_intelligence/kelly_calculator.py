"""
Kelly Criterion Calculator for FootyForecast

Calculates optimal bet sizes using the Kelly Criterion formula
to maximize long-term growth while managing risk.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class KellyCalculator:
    """
    Kelly Criterion calculator for optimal bet sizing
    """
    
    def __init__(self):
        """Initialize Kelly Calculator"""
        pass
    
    def calculate_kelly_fraction(self, 
                                probability: float, 
                                odds: float,
                                fractional_kelly: float = 1.0) -> Dict[str, Any]:
        """
        Calculate Kelly fraction for a bet
        
        Args:
            probability: Estimated probability of winning (0-1)
            odds: Decimal odds from bookmaker
            fractional_kelly: Fraction of Kelly to use (0-1, default 1.0)
            
        Returns:
            Kelly calculation results
        """
        if probability <= 0 or probability >= 1:
            raise ValueError("Probability must be between 0 and 1")
        
        if odds <= 1:
            raise ValueError("Odds must be greater than 1")
        
        # Kelly formula: f = (bp - q) / b
        # where:
        # f = fraction of bankroll to bet
        # b = odds - 1 (net odds)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        b = odds - 1  # Net odds
        p = probability
        q = 1 - probability
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly if specified
        adjusted_kelly = kelly_fraction * fractional_kelly
        
        # Calculate expected value
        expected_value = (p * b) - q
        
        # Calculate expected growth rate
        if kelly_fraction > 0:
            expected_growth = p * np.log(1 + kelly_fraction * b) + q * np.log(1 - kelly_fraction)
        else:
            expected_growth = 0
        
        return {
            'kelly_fraction': round(kelly_fraction, 4),
            'adjusted_kelly_fraction': round(adjusted_kelly, 4),
            'recommended_stake_percentage': round(max(0, adjusted_kelly) * 100, 2),
            'expected_value': round(expected_value, 4),
            'expected_growth_rate': round(expected_growth, 4),
            'is_positive_ev': expected_value > 0,
            'should_bet': adjusted_kelly > 0,
            'risk_level': self._assess_risk_level(adjusted_kelly),
            'warnings': self._generate_warnings(kelly_fraction, adjusted_kelly, probability, odds)
        }
    
    def calculate_stake_amount(self, 
                              bankroll: float,
                              probability: float,
                              odds: float,
                              fractional_kelly: float = 0.25,
                              max_stake_percentage: float = 0.05) -> Dict[str, Any]:
        """
        Calculate actual stake amount based on bankroll
        
        Args:
            bankroll: Current bankroll amount
            probability: Estimated probability of winning
            odds: Decimal odds
            fractional_kelly: Fraction of Kelly to use (default 0.25 = quarter Kelly)
            max_stake_percentage: Maximum percentage of bankroll to risk
            
        Returns:
            Stake calculation results
        """
        kelly_result = self.calculate_kelly_fraction(probability, odds, fractional_kelly)
        
        # Calculate recommended stake
        kelly_stake_percentage = kelly_result['adjusted_kelly_fraction']
        
        # Apply maximum stake limit
        final_stake_percentage = min(kelly_stake_percentage, max_stake_percentage)
        
        # Calculate actual amounts
        recommended_stake = bankroll * final_stake_percentage
        max_allowed_stake = bankroll * max_stake_percentage
        
        return {
            **kelly_result,
            'bankroll': bankroll,
            'final_stake_percentage': round(final_stake_percentage * 100, 2),
            'recommended_stake_amount': round(recommended_stake, 2),
            'max_allowed_stake': round(max_allowed_stake, 2),
            'stake_limited_by_max': final_stake_percentage < kelly_stake_percentage,
            'fractional_kelly_used': fractional_kelly
        }
    
    def analyze_multiple_bets(self, 
                             bets: List[Dict[str, Any]],
                             bankroll: float,
                             fractional_kelly: float = 0.25) -> Dict[str, Any]:
        """
        Analyze multiple betting opportunities
        
        Args:
            bets: List of bet opportunities with probability and odds
            bankroll: Current bankroll
            fractional_kelly: Fractional Kelly to use
            
        Returns:
            Analysis of all betting opportunities
        """
        results = []
        total_recommended_stake = 0
        
        for i, bet in enumerate(bets):
            try:
                bet_analysis = self.calculate_stake_amount(
                    bankroll=bankroll,
                    probability=bet['probability'],
                    odds=bet['odds'],
                    fractional_kelly=fractional_kelly
                )
                
                bet_analysis['bet_id'] = bet.get('id', f"bet_{i+1}")
                bet_analysis['match_info'] = bet.get('match_info', {})
                bet_analysis['bet_type'] = bet.get('bet_type', 'Unknown')
                
                results.append(bet_analysis)
                
                if bet_analysis['should_bet']:
                    total_recommended_stake += bet_analysis['recommended_stake_amount']
                    
            except ValueError as e:
                results.append({
                    'bet_id': bet.get('id', f"bet_{i+1}"),
                    'error': str(e),
                    'should_bet': False
                })
        
        # Sort by expected value (best opportunities first)
        valid_results = [r for r in results if 'expected_value' in r]
        valid_results.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return {
            'total_opportunities': len(bets),
            'positive_ev_bets': len([r for r in valid_results if r['is_positive_ev']]),
            'recommended_bets': len([r for r in valid_results if r['should_bet']]),
            'total_recommended_stake': round(total_recommended_stake, 2),
            'stake_percentage_of_bankroll': round((total_recommended_stake / bankroll) * 100, 2),
            'bankroll_utilization_safe': total_recommended_stake <= bankroll * 0.1,  # Conservative 10% limit
            'bet_analyses': results,
            'top_opportunities': valid_results[:5] if valid_results else []
        }
    
    def simulate_kelly_performance(self, 
                                  probability: float,
                                  odds: float,
                                  initial_bankroll: float = 1000,
                                  num_bets: int = 100,
                                  fractional_kelly: float = 0.25) -> Dict[str, Any]:
        """
        Simulate Kelly betting performance over multiple bets
        
        Args:
            probability: Win probability
            odds: Betting odds
            initial_bankroll: Starting bankroll
            num_bets: Number of bets to simulate
            fractional_kelly: Fractional Kelly to use
            
        Returns:
            Simulation results
        """
        kelly_result = self.calculate_kelly_fraction(probability, odds, fractional_kelly)
        
        if not kelly_result['should_bet']:
            return {
                'error': 'Negative expected value - simulation not recommended',
                'kelly_fraction': kelly_result['kelly_fraction']
            }
        
        # Run simulation
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        wins = 0
        losses = 0
        
        stake_percentage = kelly_result['adjusted_kelly_fraction']
        
        for _ in range(num_bets):
            stake = bankroll * stake_percentage
            
            # Simulate bet outcome
            if np.random.random() < probability:
                # Win
                bankroll += stake * (odds - 1)
                wins += 1
            else:
                # Loss
                bankroll -= stake
                losses += 1
            
            bankroll_history.append(bankroll)
        
        final_roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
        max_bankroll = max(bankroll_history)
        min_bankroll = min(bankroll_history)
        max_drawdown = ((max_bankroll - min_bankroll) / max_bankroll) * 100
        
        return {
            'initial_bankroll': initial_bankroll,
            'final_bankroll': round(bankroll, 2),
            'total_return': round(bankroll - initial_bankroll, 2),
            'roi_percentage': round(final_roi, 2),
            'wins': wins,
            'losses': losses,
            'win_rate': round((wins / num_bets) * 100, 2),
            'max_bankroll': round(max_bankroll, 2),
            'min_bankroll': round(min_bankroll, 2),
            'max_drawdown_percentage': round(max_drawdown, 2),
            'kelly_fraction_used': kelly_result['adjusted_kelly_fraction'],
            'bankroll_history': [round(b, 2) for b in bankroll_history]
        }
    
    def compare_kelly_fractions(self, 
                               probability: float,
                               odds: float,
                               fractions: List[float] = [0.1, 0.25, 0.5, 1.0]) -> Dict[str, Any]:
        """
        Compare different Kelly fractions
        
        Args:
            probability: Win probability
            odds: Betting odds
            fractions: List of fractional Kelly values to compare
            
        Returns:
            Comparison of different Kelly fractions
        """
        comparisons = []
        
        for fraction in fractions:
            try:
                result = self.calculate_kelly_fraction(probability, odds, fraction)
                
                # Run quick simulation
                simulation = self.simulate_kelly_performance(
                    probability, odds, 1000, 50, fraction
                )
                
                comparisons.append({
                    'fractional_kelly': fraction,
                    'stake_percentage': result['recommended_stake_percentage'],
                    'expected_growth': result['expected_growth_rate'],
                    'simulated_roi': simulation.get('roi_percentage', 0),
                    'simulated_max_drawdown': simulation.get('max_drawdown_percentage', 0),
                    'risk_level': result['risk_level']
                })
                
            except Exception as e:
                comparisons.append({
                    'fractional_kelly': fraction,
                    'error': str(e)
                })
        
        # Find optimal fraction (best risk-adjusted return)
        valid_comparisons = [c for c in comparisons if 'error' not in c]
        if valid_comparisons:
            # Simple scoring: ROI / (1 + max_drawdown)
            for comp in valid_comparisons:
                roi = comp['simulated_roi']
                drawdown = comp['simulated_max_drawdown']
                comp['risk_adjusted_score'] = roi / (1 + drawdown/100) if drawdown >= 0 else roi
            
            optimal = max(valid_comparisons, key=lambda x: x['risk_adjusted_score'])
        else:
            optimal = None
        
        return {
            'comparisons': comparisons,
            'optimal_fraction': optimal['fractional_kelly'] if optimal else None,
            'recommendation': self._get_fraction_recommendation(optimal) if optimal else "No positive EV opportunity"
        }
    
    def _assess_risk_level(self, kelly_fraction: float) -> str:
        """Assess risk level based on Kelly fraction"""
        if kelly_fraction <= 0:
            return "No bet recommended"
        elif kelly_fraction <= 0.02:
            return "Very Low"
        elif kelly_fraction <= 0.05:
            return "Low"
        elif kelly_fraction <= 0.10:
            return "Medium"
        elif kelly_fraction <= 0.20:
            return "High"
        else:
            return "Very High"
    
    def _generate_warnings(self, 
                          kelly_fraction: float, 
                          adjusted_kelly: float,
                          probability: float,
                          odds: float) -> List[str]:
        """Generate warnings for the bet"""
        warnings = []
        
        if kelly_fraction <= 0:
            warnings.append("Negative expected value - do not bet")
        
        if kelly_fraction > 0.20:
            warnings.append("Very high Kelly fraction - consider reducing stake")
        
        if probability > 0.9:
            warnings.append("Very high probability estimate - verify accuracy")
        
        if probability < 0.1:
            warnings.append("Very low probability estimate - high risk bet")
        
        if odds > 10:
            warnings.append("Very high odds - ensure probability estimate is accurate")
        
        if adjusted_kelly != kelly_fraction:
            warnings.append(f"Using fractional Kelly - actual Kelly is {kelly_fraction:.3f}")
        
        return warnings
    
    def _get_fraction_recommendation(self, optimal: Dict[str, Any]) -> str:
        """Get recommendation text for optimal fraction"""
        fraction = optimal['fractional_kelly']
        
        if fraction <= 0.1:
            return f"Conservative approach recommended (1/{int(1/fraction)} Kelly)"
        elif fraction <= 0.25:
            return "Quarter Kelly recommended - good balance of growth and safety"
        elif fraction <= 0.5:
            return "Half Kelly - moderate risk for higher growth"
        else:
            return "Full Kelly - maximum growth but highest risk"
