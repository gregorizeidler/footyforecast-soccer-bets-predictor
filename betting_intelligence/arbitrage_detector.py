"""
Arbitrage Detection System for FootyForecast

Detects arbitrage opportunities across multiple bookmakers
and calculates optimal stake distribution for guaranteed profits.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations

class ArbitrageDetector:
    """
    Arbitrage opportunity detector and calculator
    """
    
    def __init__(self, min_profit_threshold: float = 0.01):
        """
        Initialize arbitrage detector
        
        Args:
            min_profit_threshold: Minimum profit percentage to consider (default 1%)
        """
        self.min_profit_threshold = min_profit_threshold
    
    def detect_arbitrage_opportunity(self, 
                                   bookmaker_odds: Dict[str, Dict[str, float]],
                                   match_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Detect arbitrage opportunities across multiple bookmakers
        
        Args:
            bookmaker_odds: Dictionary of bookmaker odds
                           Format: {'bookmaker1': {'Home': 2.1, 'Draw': 3.2, 'Away': 3.8}, ...}
            match_info: Match information (optional)
            
        Returns:
            Arbitrage analysis results
        """
        if len(bookmaker_odds) < 2:
            return {'error': 'Need at least 2 bookmakers to detect arbitrage'}
        
        # Get all available outcomes
        all_outcomes = set()
        for odds in bookmaker_odds.values():
            all_outcomes.update(odds.keys())
        
        if len(all_outcomes) < 2:
            return {'error': 'Need at least 2 outcomes to detect arbitrage'}
        
        # Find best odds for each outcome
        best_odds = {}
        best_bookmakers = {}
        
        for outcome in all_outcomes:
            best_odd = 0
            best_bookmaker = None
            
            for bookmaker, odds in bookmaker_odds.items():
                if outcome in odds and odds[outcome] > best_odd:
                    best_odd = odds[outcome]
                    best_bookmaker = bookmaker
            
            if best_bookmaker:
                best_odds[outcome] = best_odd
                best_bookmakers[outcome] = best_bookmaker
        
        # Calculate arbitrage percentage
        arbitrage_percentage = sum(1/odd for odd in best_odds.values())
        
        # Check if arbitrage exists
        is_arbitrage = arbitrage_percentage < 1.0
        profit_percentage = (1 - arbitrage_percentage) * 100 if is_arbitrage else 0
        
        if not is_arbitrage or profit_percentage < self.min_profit_threshold * 100:
            return {
                'arbitrage_exists': False,
                'arbitrage_percentage': round(arbitrage_percentage, 4),
                'profit_percentage': round(profit_percentage, 4),
                'best_odds': best_odds,
                'best_bookmakers': best_bookmakers,
                'match_info': match_info or {}
            }
        
        # Calculate optimal stakes
        stake_distribution = self._calculate_optimal_stakes(best_odds, 1000)  # Using 1000 as base
        
        return {
            'arbitrage_exists': True,
            'arbitrage_percentage': round(arbitrage_percentage, 4),
            'profit_percentage': round(profit_percentage, 4),
            'expected_profit': round(profit_percentage * 10, 2),  # Profit on 1000 base
            'best_odds': best_odds,
            'best_bookmakers': best_bookmakers,
            'stake_distribution': stake_distribution,
            'total_stake_required': sum(stake_distribution.values()),
            'match_info': match_info or {},
            'risk_level': self._assess_arbitrage_risk(best_bookmakers, profit_percentage)
        }
    
    def calculate_arbitrage_stakes(self, 
                                 odds: Dict[str, float],
                                 total_stake: float) -> Dict[str, Any]:
        """
        Calculate optimal stake distribution for arbitrage
        
        Args:
            odds: Dictionary of odds for each outcome
            total_stake: Total amount to stake
            
        Returns:
            Stake distribution and profit analysis
        """
        if len(odds) < 2:
            return {'error': 'Need at least 2 outcomes'}
        
        # Check if arbitrage exists
        arbitrage_percentage = sum(1/odd for odd in odds.values())
        
        if arbitrage_percentage >= 1.0:
            return {
                'arbitrage_exists': False,
                'arbitrage_percentage': arbitrage_percentage,
                'message': 'No arbitrage opportunity exists'
            }
        
        # Calculate optimal stakes
        stakes = {}
        total_return = total_stake / arbitrage_percentage
        
        for outcome, odd in odds.items():
            stake = total_return / odd
            stakes[outcome] = round(stake, 2)
        
        # Calculate profits for each outcome
        profits = {}
        for outcome, odd in odds.items():
            profit = (stakes[outcome] * odd) - total_stake
            profits[outcome] = round(profit, 2)
        
        # Verify all profits are equal (they should be in perfect arbitrage)
        profit_values = list(profits.values())
        profit_consistency = max(profit_values) - min(profit_values) < 0.01
        
        return {
            'arbitrage_exists': True,
            'arbitrage_percentage': round(arbitrage_percentage, 4),
            'total_stake': total_stake,
            'stakes': stakes,
            'profits': profits,
            'guaranteed_profit': round(min(profit_values), 2),
            'profit_percentage': round((min(profit_values) / total_stake) * 100, 2),
            'profit_consistency': profit_consistency,
            'total_return': round(total_return, 2)
        }
    
    def scan_multiple_matches(self, 
                            matches_odds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scan multiple matches for arbitrage opportunities
        
        Args:
            matches_odds: List of match odds data
                         Format: [{'match_info': {...}, 'bookmaker_odds': {...}}, ...]
            
        Returns:
            List of arbitrage opportunities found
        """
        arbitrage_opportunities = []
        
        for match_data in matches_odds:
            match_info = match_data.get('match_info', {})
            bookmaker_odds = match_data.get('bookmaker_odds', {})
            
            arbitrage_result = self.detect_arbitrage_opportunity(bookmaker_odds, match_info)
            
            if arbitrage_result.get('arbitrage_exists', False):
                arbitrage_opportunities.append(arbitrage_result)
        
        # Sort by profit percentage (best opportunities first)
        arbitrage_opportunities.sort(key=lambda x: x['profit_percentage'], reverse=True)
        
        return arbitrage_opportunities
    
    def analyze_bookmaker_margins(self, 
                                bookmaker_odds: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze bookmaker margins and overround
        
        Args:
            bookmaker_odds: Dictionary of bookmaker odds
            
        Returns:
            Margin analysis for each bookmaker
        """
        margin_analysis = {}
        
        for bookmaker, odds in bookmaker_odds.items():
            if not odds:
                continue
            
            # Calculate overround (bookmaker margin)
            overround = sum(1/odd for odd in odds.values())
            margin_percentage = (overround - 1) * 100
            
            # Calculate fair odds (removing margin)
            fair_odds = {}
            for outcome, odd in odds.items():
                implied_prob = 1 / odd
                fair_prob = implied_prob / overround
                fair_odds[outcome] = round(1 / fair_prob, 2)
            
            margin_analysis[bookmaker] = {
                'overround': round(overround, 4),
                'margin_percentage': round(margin_percentage, 2),
                'original_odds': odds,
                'fair_odds': fair_odds,
                'competitiveness': self._assess_competitiveness(margin_percentage)
            }
        
        # Find most competitive bookmaker
        if margin_analysis:
            most_competitive = min(margin_analysis.items(), 
                                 key=lambda x: x[1]['margin_percentage'])
            
            return {
                'bookmaker_analysis': margin_analysis,
                'most_competitive_bookmaker': most_competitive[0],
                'lowest_margin': most_competitive[1]['margin_percentage'],
                'average_margin': round(np.mean([b['margin_percentage'] 
                                               for b in margin_analysis.values()]), 2)
            }
        
        return {'error': 'No valid bookmaker data provided'}
    
    def calculate_sure_bet_bankroll(self, 
                                  arbitrage_opportunities: List[Dict[str, Any]],
                                  available_bankroll: float,
                                  risk_per_opportunity: float = 0.1) -> Dict[str, Any]:
        """
        Calculate bankroll allocation across multiple arbitrage opportunities
        
        Args:
            arbitrage_opportunities: List of arbitrage opportunities
            available_bankroll: Available bankroll
            risk_per_opportunity: Maximum risk per opportunity (as fraction)
            
        Returns:
            Bankroll allocation strategy
        """
        if not arbitrage_opportunities:
            return {'error': 'No arbitrage opportunities provided'}
        
        # Filter valid opportunities
        valid_opportunities = [
            opp for opp in arbitrage_opportunities 
            if opp.get('arbitrage_exists', False)
        ]
        
        if not valid_opportunities:
            return {'error': 'No valid arbitrage opportunities found'}
        
        # Sort by profit percentage
        valid_opportunities.sort(key=lambda x: x['profit_percentage'], reverse=True)
        
        # Calculate allocation
        allocations = []
        remaining_bankroll = available_bankroll
        max_stake_per_opportunity = available_bankroll * risk_per_opportunity
        
        for opp in valid_opportunities:
            if remaining_bankroll <= 0:
                break
            
            # Calculate required stake for this opportunity
            required_stake = opp.get('total_stake_required', 0)
            
            # Limit stake by available bankroll and risk limits
            allocated_stake = min(required_stake, max_stake_per_opportunity, remaining_bankroll)
            
            if allocated_stake > 0:
                # Calculate proportional profits
                scale_factor = allocated_stake / required_stake if required_stake > 0 else 0
                expected_profit = opp['expected_profit'] * scale_factor
                
                allocations.append({
                    'match_info': opp.get('match_info', {}),
                    'profit_percentage': opp['profit_percentage'],
                    'allocated_stake': round(allocated_stake, 2),
                    'expected_profit': round(expected_profit, 2),
                    'best_bookmakers': opp['best_bookmakers'],
                    'stake_distribution': {
                        outcome: round(stake * scale_factor, 2)
                        for outcome, stake in opp['stake_distribution'].items()
                    }
                })
                
                remaining_bankroll -= allocated_stake
        
        total_allocated = sum(alloc['allocated_stake'] for alloc in allocations)
        total_expected_profit = sum(alloc['expected_profit'] for alloc in allocations)
        
        return {
            'total_opportunities': len(valid_opportunities),
            'allocated_opportunities': len(allocations),
            'total_allocated': round(total_allocated, 2),
            'remaining_bankroll': round(remaining_bankroll, 2),
            'bankroll_utilization': round((total_allocated / available_bankroll) * 100, 2),
            'total_expected_profit': round(total_expected_profit, 2),
            'expected_roi': round((total_expected_profit / total_allocated) * 100, 2) if total_allocated > 0 else 0,
            'allocations': allocations
        }
    
    def _calculate_optimal_stakes(self, odds: Dict[str, float], total_stake: float) -> Dict[str, float]:
        """Calculate optimal stake distribution"""
        arbitrage_percentage = sum(1/odd for odd in odds.values())
        
        if arbitrage_percentage >= 1.0:
            return {}
        
        stakes = {}
        total_return = total_stake / arbitrage_percentage
        
        for outcome, odd in odds.items():
            stake = total_return / odd
            stakes[outcome] = round(stake, 2)
        
        return stakes
    
    def _assess_arbitrage_risk(self, 
                             best_bookmakers: Dict[str, str], 
                             profit_percentage: float) -> str:
        """Assess risk level of arbitrage opportunity"""
        # Count unique bookmakers
        unique_bookmakers = len(set(best_bookmakers.values()))
        
        # Risk factors
        risk_score = 0
        
        # Profit margin risk (lower margins are riskier due to odds changes)
        if profit_percentage < 1:
            risk_score += 2
        elif profit_percentage < 2:
            risk_score += 1
        
        # Bookmaker diversity risk
        if unique_bookmakers < len(best_bookmakers):
            risk_score += 1  # Same bookmaker for multiple outcomes
        
        # Classify risk
        if risk_score <= 1:
            return 'Low'
        elif risk_score <= 2:
            return 'Medium'
        else:
            return 'High'
    
    def _assess_competitiveness(self, margin_percentage: float) -> str:
        """Assess bookmaker competitiveness based on margin"""
        if margin_percentage < 2:
            return 'Excellent'
        elif margin_percentage < 4:
            return 'Very Good'
        elif margin_percentage < 6:
            return 'Good'
        elif margin_percentage < 8:
            return 'Average'
        else:
            return 'Poor'
    
    def find_cross_market_arbitrage(self, 
                                  match_markets: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities across different markets (e.g., 1X2 vs Asian Handicap)
        
        Args:
            match_markets: Dictionary of different markets and their odds
                          Format: {'1X2': {'bookmaker1': {'Home': 2.1, ...}}, 'O/U2.5': {...}}
            
        Returns:
            List of cross-market arbitrage opportunities
        """
        cross_market_opportunities = []
        
        # This is a simplified version - in practice, you'd need complex conversion
        # between different market types (e.g., converting Asian Handicap to 1X2)
        
        for market_name, market_data in match_markets.items():
            arbitrage_result = self.detect_arbitrage_opportunity(market_data)
            
            if arbitrage_result.get('arbitrage_exists', False):
                arbitrage_result['market_type'] = market_name
                cross_market_opportunities.append(arbitrage_result)
        
        return cross_market_opportunities
    
    def monitor_odds_changes(self, 
                           historical_odds: List[Dict[str, Any]],
                           current_odds: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Monitor odds changes to identify emerging arbitrage opportunities
        
        Args:
            historical_odds: Historical odds data with timestamps
            current_odds: Current odds from bookmakers
            
        Returns:
            Odds movement analysis and arbitrage alerts
        """
        if not historical_odds:
            return {'error': 'No historical data provided'}
        
        # Get latest historical odds for comparison
        latest_historical = historical_odds[-1] if historical_odds else {}
        
        # Calculate odds movements
        odds_movements = {}
        arbitrage_alerts = []
        
        for bookmaker, current_bookmaker_odds in current_odds.items():
            if bookmaker in latest_historical:
                historical_bookmaker_odds = latest_historical[bookmaker]
                
                for outcome, current_odd in current_bookmaker_odds.items():
                    if outcome in historical_bookmaker_odds:
                        historical_odd = historical_bookmaker_odds[outcome]
                        movement = ((current_odd - historical_odd) / historical_odd) * 100
                        
                        if bookmaker not in odds_movements:
                            odds_movements[bookmaker] = {}
                        
                        odds_movements[bookmaker][outcome] = {
                            'historical_odd': historical_odd,
                            'current_odd': current_odd,
                            'movement_percentage': round(movement, 2)
                        }
        
        # Check if movements created new arbitrage opportunities
        current_arbitrage = self.detect_arbitrage_opportunity(current_odds)
        
        if current_arbitrage.get('arbitrage_exists', False):
            arbitrage_alerts.append({
                'type': 'new_arbitrage',
                'profit_percentage': current_arbitrage['profit_percentage'],
                'message': f"New arbitrage opportunity detected: {current_arbitrage['profit_percentage']:.2f}% profit"
            })
        
        return {
            'odds_movements': odds_movements,
            'current_arbitrage': current_arbitrage,
            'arbitrage_alerts': arbitrage_alerts,
            'significant_movements': self._find_significant_movements(odds_movements)
        }
    
    def _find_significant_movements(self, odds_movements: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find significant odds movements (>5% change)"""
        significant_movements = []
        
        for bookmaker, bookmaker_movements in odds_movements.items():
            for outcome, movement_data in bookmaker_movements.items():
                if abs(movement_data['movement_percentage']) > 5:
                    significant_movements.append({
                        'bookmaker': bookmaker,
                        'outcome': outcome,
                        'movement_percentage': movement_data['movement_percentage'],
                        'direction': 'increased' if movement_data['movement_percentage'] > 0 else 'decreased'
                    })
        
        return significant_movements
