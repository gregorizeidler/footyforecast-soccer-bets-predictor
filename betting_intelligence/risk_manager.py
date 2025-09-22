"""
Risk Management System for FootyForecast

Advanced risk management tools including position sizing,
drawdown protection, and portfolio risk analysis.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings

class RiskManager:
    """
    Comprehensive risk management system for betting portfolios
    """
    
    def __init__(self, 
                 max_daily_risk: float = 0.02,
                 max_single_bet_risk: float = 0.01,
                 max_drawdown_limit: float = 0.10,
                 stop_loss_threshold: float = 0.05):
        """
        Initialize risk manager
        
        Args:
            max_daily_risk: Maximum daily risk as fraction of bankroll
            max_single_bet_risk: Maximum single bet risk as fraction of bankroll
            max_drawdown_limit: Maximum drawdown before stopping
            stop_loss_threshold: Stop loss threshold as fraction of bankroll
        """
        self.max_daily_risk = max_daily_risk
        self.max_single_bet_risk = max_single_bet_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.stop_loss_threshold = stop_loss_threshold
    
    def assess_bet_risk(self, 
                       stake: float,
                       bankroll: float,
                       probability: float,
                       odds: float,
                       current_daily_risk: float = 0.0) -> Dict[str, Any]:
        """
        Assess risk for a single bet
        
        Args:
            stake: Proposed stake amount
            bankroll: Current bankroll
            probability: Estimated win probability
            odds: Betting odds
            current_daily_risk: Current daily risk exposure
            
        Returns:
            Risk assessment results
        """
        # Calculate risk metrics
        stake_percentage = stake / bankroll if bankroll > 0 else 0
        potential_loss = stake
        potential_loss_percentage = potential_loss / bankroll if bankroll > 0 else 0
        
        # Calculate value at risk (VaR) - potential loss at 95% confidence
        var_95 = stake * (1 - probability)  # Expected loss
        
        # Calculate expected value
        expected_value = (probability * stake * (odds - 1)) - ((1 - probability) * stake)
        
        # Risk-reward ratio
        potential_profit = stake * (odds - 1)
        risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 0
        
        # Check risk limits
        risk_checks = {
            'single_bet_risk_ok': stake_percentage <= self.max_single_bet_risk,
            'daily_risk_ok': (current_daily_risk + potential_loss_percentage) <= self.max_daily_risk,
            'positive_expected_value': expected_value > 0,
            'acceptable_risk_reward': risk_reward_ratio >= 1.5  # Minimum 1.5:1 ratio
        }
        
        # Overall risk assessment
        risk_level = self._calculate_risk_level(stake_percentage, probability, odds)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            stake_percentage, probability, odds, risk_checks
        )
        
        return {
            'stake_amount': stake,
            'stake_percentage': round(stake_percentage * 100, 2),
            'potential_loss': potential_loss,
            'potential_loss_percentage': round(potential_loss_percentage * 100, 2),
            'potential_profit': potential_profit,
            'expected_value': round(expected_value, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'var_95': round(var_95, 2),
            'risk_level': risk_level,
            'risk_checks': risk_checks,
            'all_checks_passed': all(risk_checks.values()),
            'recommendations': recommendations,
            'adjusted_stake': self._calculate_adjusted_stake(stake, bankroll, risk_checks)
        }
    
    def analyze_portfolio_risk(self, 
                              current_bets: List[Dict[str, Any]],
                              bankroll: float,
                              historical_performance: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze overall portfolio risk
        
        Args:
            current_bets: List of current open bets
            bankroll: Current bankroll
            historical_performance: Historical betting performance
            
        Returns:
            Portfolio risk analysis
        """
        if not current_bets:
            return {
                'total_exposure': 0,
                'risk_level': 'No Risk',
                'recommendations': ['No current positions']
            }
        
        # Calculate total exposure
        total_stake = sum(bet.get('stake', 0) for bet in current_bets)
        total_exposure_percentage = (total_stake / bankroll) * 100 if bankroll > 0 else 0
        
        # Calculate potential maximum loss (if all bets lose)
        max_potential_loss = total_stake
        max_loss_percentage = (max_potential_loss / bankroll) * 100 if bankroll > 0 else 0
        
        # Calculate expected portfolio value
        expected_portfolio_value = 0
        for bet in current_bets:
            probability = bet.get('probability', 0.5)
            odds = bet.get('odds', 2.0)
            stake = bet.get('stake', 0)
            expected_value = (probability * stake * (odds - 1)) - ((1 - probability) * stake)
            expected_portfolio_value += expected_value
        
        # Diversification analysis
        diversification = self._analyze_diversification(current_bets)
        
        # Historical risk metrics
        historical_metrics = {}
        if historical_performance:
            historical_metrics = self._calculate_historical_risk_metrics(historical_performance)
        
        # Overall portfolio risk level
        portfolio_risk_level = self._assess_portfolio_risk_level(
            total_exposure_percentage, max_loss_percentage, diversification
        )
        
        # Generate portfolio recommendations
        portfolio_recommendations = self._generate_portfolio_recommendations(
            total_exposure_percentage, max_loss_percentage, diversification, historical_metrics
        )
        
        return {
            'total_open_bets': len(current_bets),
            'total_stake': round(total_stake, 2),
            'total_exposure_percentage': round(total_exposure_percentage, 2),
            'max_potential_loss': round(max_potential_loss, 2),
            'max_loss_percentage': round(max_loss_percentage, 2),
            'expected_portfolio_value': round(expected_portfolio_value, 2),
            'portfolio_risk_level': portfolio_risk_level,
            'diversification_score': diversification['score'],
            'diversification_analysis': diversification,
            'historical_metrics': historical_metrics,
            'risk_warnings': self._generate_risk_warnings(total_exposure_percentage, max_loss_percentage),
            'recommendations': portfolio_recommendations
        }
    
    def calculate_position_sizing(self, 
                                 available_opportunities: List[Dict[str, Any]],
                                 bankroll: float,
                                 risk_budget: float = None) -> Dict[str, Any]:
        """
        Calculate optimal position sizing for multiple opportunities
        
        Args:
            available_opportunities: List of betting opportunities
            bankroll: Current bankroll
            risk_budget: Daily/weekly risk budget (default uses max_daily_risk)
            
        Returns:
            Position sizing recommendations
        """
        if not available_opportunities:
            return {'error': 'No opportunities provided'}
        
        risk_budget = risk_budget or self.max_daily_risk
        available_risk = bankroll * risk_budget
        
        # Score opportunities by risk-adjusted expected value
        scored_opportunities = []
        for opp in available_opportunities:
            probability = opp.get('probability', 0.5)
            odds = opp.get('odds', 2.0)
            
            # Calculate expected value per unit risk
            expected_return = probability * (odds - 1) - (1 - probability)
            risk_per_unit = 1 - probability  # Risk of losing the stake
            
            if risk_per_unit > 0:
                risk_adjusted_score = expected_return / risk_per_unit
            else:
                risk_adjusted_score = expected_return
            
            scored_opportunities.append({
                **opp,
                'expected_return': expected_return,
                'risk_per_unit': risk_per_unit,
                'risk_adjusted_score': risk_adjusted_score
            })
        
        # Sort by risk-adjusted score
        scored_opportunities.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)
        
        # Allocate risk budget
        allocations = []
        remaining_risk_budget = available_risk
        
        for opp in scored_opportunities:
            if remaining_risk_budget <= 0:
                break
            
            if opp['risk_adjusted_score'] <= 0:
                continue  # Skip negative expected value bets
            
            # Calculate optimal stake based on available risk budget
            max_stake_by_budget = remaining_risk_budget
            max_stake_by_single_bet_limit = bankroll * self.max_single_bet_risk
            
            recommended_stake = min(max_stake_by_budget, max_stake_by_single_bet_limit)
            
            if recommended_stake > 0:
                allocations.append({
                    'opportunity_id': opp.get('id', 'unknown'),
                    'match_info': opp.get('match_info', {}),
                    'probability': opp['probability'],
                    'odds': opp['odds'],
                    'recommended_stake': round(recommended_stake, 2),
                    'stake_percentage': round((recommended_stake / bankroll) * 100, 2),
                    'expected_return': round(opp['expected_return'] * recommended_stake, 2),
                    'risk_adjusted_score': round(opp['risk_adjusted_score'], 3)
                })
                
                remaining_risk_budget -= recommended_stake
        
        total_allocated = sum(alloc['recommended_stake'] for alloc in allocations)
        
        return {
            'total_opportunities': len(available_opportunities),
            'recommended_bets': len(allocations),
            'total_allocated': round(total_allocated, 2),
            'risk_budget_used': round((total_allocated / available_risk) * 100, 2),
            'remaining_risk_budget': round(remaining_risk_budget, 2),
            'allocations': allocations,
            'portfolio_expected_return': round(sum(alloc['expected_return'] for alloc in allocations), 2)
        }
    
    def monitor_drawdown(self, 
                        current_bankroll: float,
                        peak_bankroll: float,
                        recent_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Monitor drawdown and suggest protective actions
        
        Args:
            current_bankroll: Current bankroll amount
            peak_bankroll: Historical peak bankroll
            recent_performance: Recent betting performance
            
        Returns:
            Drawdown analysis and recommendations
        """
        # Calculate current drawdown
        current_drawdown = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        current_drawdown_percentage = current_drawdown * 100
        
        # Analyze recent performance trend
        if recent_performance:
            recent_returns = [perf.get('profit_loss', 0) for perf in recent_performance[-10:]]
            recent_trend = 'declining' if sum(recent_returns[-5:]) < sum(recent_returns[-10:-5]) else 'improving'
            consecutive_losses = self._count_consecutive_losses(recent_performance)
        else:
            recent_trend = 'unknown'
            consecutive_losses = 0
        
        # Determine drawdown severity
        if current_drawdown_percentage < 2:
            severity = 'Minimal'
        elif current_drawdown_percentage < 5:
            severity = 'Low'
        elif current_drawdown_percentage < 10:
            severity = 'Moderate'
        elif current_drawdown_percentage < 20:
            severity = 'High'
        else:
            severity = 'Severe'
        
        # Generate protective recommendations
        recommendations = []
        
        if current_drawdown >= self.max_drawdown_limit:
            recommendations.append("STOP BETTING: Maximum drawdown limit reached")
        elif current_drawdown >= self.stop_loss_threshold:
            recommendations.append("Reduce position sizes by 50%")
            recommendations.append("Review and adjust strategy")
        elif consecutive_losses >= 5:
            recommendations.append("Consider taking a break after 5+ consecutive losses")
        elif recent_trend == 'declining':
            recommendations.append("Monitor performance closely - declining trend detected")
        
        if current_drawdown_percentage > 5:
            recommendations.append("Focus on high-confidence bets only")
            recommendations.append("Consider reducing stake sizes temporarily")
        
        return {
            'current_bankroll': current_bankroll,
            'peak_bankroll': peak_bankroll,
            'drawdown_amount': round(peak_bankroll - current_bankroll, 2),
            'drawdown_percentage': round(current_drawdown_percentage, 2),
            'severity': severity,
            'recent_trend': recent_trend,
            'consecutive_losses': consecutive_losses,
            'stop_loss_triggered': current_drawdown >= self.stop_loss_threshold,
            'max_drawdown_reached': current_drawdown >= self.max_drawdown_limit,
            'recommendations': recommendations,
            'should_stop_betting': current_drawdown >= self.max_drawdown_limit
        }
    
    def _calculate_risk_level(self, stake_percentage: float, probability: float, odds: float) -> str:
        """Calculate overall risk level for a bet"""
        # Combine multiple risk factors
        risk_score = 0
        
        # Stake size risk
        if stake_percentage > 0.05:
            risk_score += 3
        elif stake_percentage > 0.02:
            risk_score += 2
        elif stake_percentage > 0.01:
            risk_score += 1
        
        # Probability risk
        if probability < 0.3:
            risk_score += 2
        elif probability < 0.5:
            risk_score += 1
        
        # Odds risk
        if odds > 5:
            risk_score += 2
        elif odds > 3:
            risk_score += 1
        
        # Classify risk level
        if risk_score <= 1:
            return 'Low'
        elif risk_score <= 3:
            return 'Medium'
        elif risk_score <= 5:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_risk_recommendations(self, 
                                     stake_percentage: float,
                                     probability: float,
                                     odds: float,
                                     risk_checks: Dict[str, bool]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if not risk_checks['single_bet_risk_ok']:
            recommendations.append(f"Reduce stake - exceeds {self.max_single_bet_risk*100}% single bet limit")
        
        if not risk_checks['daily_risk_ok']:
            recommendations.append(f"Reduce stake - would exceed {self.max_daily_risk*100}% daily risk limit")
        
        if not risk_checks['positive_expected_value']:
            recommendations.append("Avoid bet - negative expected value")
        
        if not risk_checks['acceptable_risk_reward']:
            recommendations.append("Poor risk-reward ratio - consider skipping")
        
        if probability < 0.4:
            recommendations.append("Low probability bet - ensure analysis is thorough")
        
        if odds > 4:
            recommendations.append("High odds bet - verify probability estimate")
        
        if stake_percentage > 0.03:
            recommendations.append("Consider reducing stake size for better risk management")
        
        return recommendations
    
    def _calculate_adjusted_stake(self, 
                                original_stake: float,
                                bankroll: float,
                                risk_checks: Dict[str, bool]) -> float:
        """Calculate risk-adjusted stake amount"""
        if all(risk_checks.values()):
            return original_stake
        
        # Adjust stake based on failed risk checks
        max_allowed_by_single_limit = bankroll * self.max_single_bet_risk
        max_allowed_by_daily_limit = bankroll * self.max_daily_risk  # Simplified
        
        adjusted_stake = min(original_stake, max_allowed_by_single_limit, max_allowed_by_daily_limit)
        
        return max(0, adjusted_stake)
    
    def _analyze_diversification(self, current_bets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        if not current_bets:
            return {'score': 0, 'analysis': 'No bets to analyze'}
        
        # Analyze by league
        leagues = {}
        bet_types = {}
        total_stake = sum(bet.get('stake', 0) for bet in current_bets)
        
        for bet in current_bets:
            league = bet.get('match_info', {}).get('league', 'Unknown')
            bet_type = bet.get('bet_type', 'Unknown')
            stake = bet.get('stake', 0)
            
            leagues[league] = leagues.get(league, 0) + stake
            bet_types[bet_type] = bet_types.get(bet_type, 0) + stake
        
        # Calculate concentration scores
        league_concentration = max(leagues.values()) / total_stake if total_stake > 0 else 0
        bet_type_concentration = max(bet_types.values()) / total_stake if total_stake > 0 else 0
        
        # Overall diversification score (0-100)
        diversification_score = 100 * (1 - max(league_concentration, bet_type_concentration))
        
        return {
            'score': round(diversification_score, 1),
            'league_distribution': {k: round((v/total_stake)*100, 1) for k, v in leagues.items()},
            'bet_type_distribution': {k: round((v/total_stake)*100, 1) for k, v in bet_types.items()},
            'league_concentration': round(league_concentration * 100, 1),
            'bet_type_concentration': round(bet_type_concentration * 100, 1),
            'analysis': self._get_diversification_analysis(diversification_score)
        }
    
    def _calculate_historical_risk_metrics(self, historical_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate historical risk metrics"""
        if not historical_performance:
            return {}
        
        returns = [perf.get('profit_loss', 0) for perf in historical_performance]
        
        # Basic statistics
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Risk metrics
        downside_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0
        
        # Sharpe-like ratio (return per unit of risk)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'average_return': round(avg_return, 2),
            'volatility': round(std_return, 2),
            'downside_deviation': round(downside_deviation, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_historical_drawdown': round(max_drawdown, 2),
            'win_rate': round(len([r for r in returns if r > 0]) / len(returns) * 100, 1)
        }
    
    def _assess_portfolio_risk_level(self, 
                                   exposure_percentage: float,
                                   max_loss_percentage: float,
                                   diversification: Dict[str, Any]) -> str:
        """Assess overall portfolio risk level"""
        risk_score = 0
        
        # Exposure risk
        if exposure_percentage > 15:
            risk_score += 3
        elif exposure_percentage > 10:
            risk_score += 2
        elif exposure_percentage > 5:
            risk_score += 1
        
        # Maximum loss risk
        if max_loss_percentage > 10:
            risk_score += 3
        elif max_loss_percentage > 5:
            risk_score += 2
        elif max_loss_percentage > 2:
            risk_score += 1
        
        # Diversification risk
        if diversification['score'] < 30:
            risk_score += 2
        elif diversification['score'] < 50:
            risk_score += 1
        
        # Classify risk
        if risk_score <= 1:
            return 'Low'
        elif risk_score <= 3:
            return 'Medium'
        elif risk_score <= 5:
            return 'High'
        else:
            return 'Very High'
    
    def _generate_portfolio_recommendations(self, 
                                          exposure_percentage: float,
                                          max_loss_percentage: float,
                                          diversification: Dict[str, Any],
                                          historical_metrics: Dict[str, Any]) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        if exposure_percentage > 10:
            recommendations.append("High portfolio exposure - consider reducing position sizes")
        
        if max_loss_percentage > 5:
            recommendations.append("Maximum potential loss is high - diversify or reduce stakes")
        
        if diversification['score'] < 50:
            recommendations.append("Poor diversification - spread bets across more leagues/bet types")
        
        if historical_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Low risk-adjusted returns - review betting strategy")
        
        if historical_metrics.get('win_rate', 50) < 40:
            recommendations.append("Low win rate - focus on higher probability bets")
        
        return recommendations
    
    def _generate_risk_warnings(self, exposure_percentage: float, max_loss_percentage: float) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        if exposure_percentage > 20:
            warnings.append("CRITICAL: Portfolio exposure exceeds 20% - high risk of significant losses")
        elif exposure_percentage > 15:
            warnings.append("WARNING: High portfolio exposure detected")
        
        if max_loss_percentage > 15:
            warnings.append("CRITICAL: Potential maximum loss exceeds 15% of bankroll")
        elif max_loss_percentage > 10:
            warnings.append("WARNING: High potential loss scenario")
        
        return warnings
    
    def _count_consecutive_losses(self, recent_performance: List[Dict[str, Any]]) -> int:
        """Count consecutive losses from most recent bets"""
        consecutive = 0
        for perf in reversed(recent_performance):
            if perf.get('profit_loss', 0) < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _get_diversification_analysis(self, score: float) -> str:
        """Get diversification analysis text"""
        if score >= 80:
            return "Excellent diversification"
        elif score >= 60:
            return "Good diversification"
        elif score >= 40:
            return "Moderate diversification - could improve"
        elif score >= 20:
            return "Poor diversification - high concentration risk"
        else:
            return "Very poor diversification - significant concentration risk"
