"""
Comprehensive Betting Analyzer for FootyForecast

Main interface that combines all betting intelligence modules
for comprehensive betting analysis and recommendations.
"""

from typing import Dict, List, Any, Optional
from .portfolio_tracker import PortfolioTracker
from .kelly_calculator import KellyCalculator
from .risk_manager import RiskManager
from .arbitrage_detector import ArbitrageDetector

class BettingAnalyzer:
    """
    Main betting intelligence analyzer combining all modules
    """
    
    def __init__(self, 
                 portfolio_file: str = "betting_portfolio.json",
                 max_daily_risk: float = 0.02,
                 min_arbitrage_profit: float = 0.01):
        """
        Initialize betting analyzer
        
        Args:
            portfolio_file: File to store betting portfolio data
            max_daily_risk: Maximum daily risk as fraction of bankroll
            min_arbitrage_profit: Minimum arbitrage profit threshold
        """
        self.portfolio = PortfolioTracker(portfolio_file)
        self.kelly = KellyCalculator()
        self.risk_manager = RiskManager(max_daily_risk=max_daily_risk)
        self.arbitrage = ArbitrageDetector(min_arbitrage_profit)
    
    def analyze_betting_opportunity(self, 
                                  opportunity: Dict[str, Any],
                                  bankroll: float,
                                  bookmaker_odds: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a betting opportunity
        
        Args:
            opportunity: Betting opportunity data
            bankroll: Current bankroll
            bookmaker_odds: Odds from multiple bookmakers (for arbitrage detection)
            
        Returns:
            Complete betting analysis
        """
        analysis = {
            'opportunity_id': opportunity.get('id', 'unknown'),
            'match_info': opportunity.get('match_info', {}),
            'timestamp': opportunity.get('timestamp', 'unknown')
        }
        
        # Extract key parameters
        probability = opportunity.get('probability', 0.5)
        odds = opportunity.get('odds', 2.0)
        
        # Kelly Criterion Analysis
        try:
            kelly_analysis = self.kelly.calculate_stake_amount(
                bankroll=bankroll,
                probability=probability,
                odds=odds,
                fractional_kelly=0.25  # Conservative quarter Kelly
            )
            analysis['kelly_analysis'] = kelly_analysis
        except Exception as e:
            analysis['kelly_analysis'] = {'error': str(e)}
        
        # Risk Management Analysis
        if 'kelly_analysis' in analysis and 'recommended_stake_amount' in analysis['kelly_analysis']:
            recommended_stake = analysis['kelly_analysis']['recommended_stake_amount']
            
            risk_analysis = self.risk_manager.assess_bet_risk(
                stake=recommended_stake,
                bankroll=bankroll,
                probability=probability,
                odds=odds
            )
            analysis['risk_analysis'] = risk_analysis
        
        # Arbitrage Analysis (if multiple bookmaker odds provided)
        if bookmaker_odds:
            arbitrage_analysis = self.arbitrage.detect_arbitrage_opportunity(
                bookmaker_odds=bookmaker_odds,
                match_info=opportunity.get('match_info', {})
            )
            analysis['arbitrage_analysis'] = arbitrage_analysis
        
        # Overall Recommendation
        analysis['recommendation'] = self._generate_overall_recommendation(analysis)
        
        return analysis
    
    def analyze_portfolio_performance(self, 
                                    bankroll: float,
                                    days: Optional[int] = 30) -> Dict[str, Any]:
        """
        Comprehensive portfolio performance analysis
        
        Args:
            bankroll: Current bankroll
            days: Number of days to analyze (None for all time)
            
        Returns:
            Complete portfolio analysis
        """
        # Portfolio Summary
        portfolio_summary = self.portfolio.get_portfolio_summary(days)
        
        # Performance by Category
        category_performance = self.portfolio.get_performance_by_category()
        
        # Monthly Performance
        monthly_performance = self.portfolio.get_monthly_performance()
        
        # Risk Analysis
        current_bets = [bet for bet in self.portfolio.bets if bet['status'] == 'pending']
        settled_bets = [bet for bet in self.portfolio.bets if bet['status'] in ['won', 'lost', 'void']]
        
        portfolio_risk = self.risk_manager.analyze_portfolio_risk(
            current_bets=current_bets,
            bankroll=bankroll,
            historical_performance=settled_bets
        )
        
        # Drawdown Analysis
        if settled_bets:
            # Calculate peak bankroll (simplified - assumes starting bankroll)
            total_profit_loss = sum(bet['profit_loss'] for bet in settled_bets)
            estimated_starting_bankroll = bankroll - total_profit_loss
            peak_bankroll = max(bankroll, estimated_starting_bankroll)
            
            drawdown_analysis = self.risk_manager.monitor_drawdown(
                current_bankroll=bankroll,
                peak_bankroll=peak_bankroll,
                recent_performance=settled_bets[-20:]  # Last 20 bets
            )
        else:
            drawdown_analysis = {'message': 'No historical data for drawdown analysis'}
        
        return {
            'portfolio_summary': portfolio_summary,
            'category_performance': category_performance,
            'monthly_performance': monthly_performance,
            'portfolio_risk': portfolio_risk,
            'drawdown_analysis': drawdown_analysis,
            'recommendations': self._generate_portfolio_recommendations(
                portfolio_summary, portfolio_risk, drawdown_analysis
            )
        }
    
    def scan_arbitrage_opportunities(self, 
                                   matches_data: List[Dict[str, Any]],
                                   bankroll: float) -> Dict[str, Any]:
        """
        Scan for arbitrage opportunities across multiple matches
        
        Args:
            matches_data: List of match data with bookmaker odds
            bankroll: Available bankroll
            
        Returns:
            Arbitrage opportunities analysis
        """
        # Detect arbitrage opportunities
        arbitrage_opportunities = self.arbitrage.scan_multiple_matches(matches_data)
        
        if not arbitrage_opportunities:
            return {
                'total_matches_scanned': len(matches_data),
                'arbitrage_opportunities_found': 0,
                'message': 'No arbitrage opportunities detected'
            }
        
        # Calculate optimal bankroll allocation
        bankroll_allocation = self.arbitrage.calculate_sure_bet_bankroll(
            arbitrage_opportunities=arbitrage_opportunities,
            available_bankroll=bankroll,
            risk_per_opportunity=0.05  # 5% max per opportunity
        )
        
        # Analyze bookmaker margins
        margin_analyses = []
        for match_data in matches_data:
            bookmaker_odds = match_data.get('bookmaker_odds', {})
            if bookmaker_odds:
                margin_analysis = self.arbitrage.analyze_bookmaker_margins(bookmaker_odds)
                margin_analysis['match_info'] = match_data.get('match_info', {})
                margin_analyses.append(margin_analysis)
        
        return {
            'total_matches_scanned': len(matches_data),
            'arbitrage_opportunities_found': len(arbitrage_opportunities),
            'arbitrage_opportunities': arbitrage_opportunities,
            'bankroll_allocation': bankroll_allocation,
            'margin_analyses': margin_analyses,
            'recommendations': self._generate_arbitrage_recommendations(
                arbitrage_opportunities, bankroll_allocation
            )
        }
    
    def optimize_betting_strategy(self, 
                                opportunities: List[Dict[str, Any]],
                                bankroll: float,
                                strategy_type: str = 'balanced') -> Dict[str, Any]:
        """
        Optimize betting strategy across multiple opportunities
        
        Args:
            opportunities: List of betting opportunities
            bankroll: Available bankroll
            strategy_type: Strategy type ('conservative', 'balanced', 'aggressive')
            
        Returns:
            Optimized betting strategy
        """
        # Set strategy parameters
        strategy_params = self._get_strategy_parameters(strategy_type)
        
        # Analyze each opportunity
        analyzed_opportunities = []
        for opp in opportunities:
            analysis = self.analyze_betting_opportunity(opp, bankroll)
            analyzed_opportunities.append(analysis)
        
        # Filter opportunities based on strategy
        filtered_opportunities = self._filter_opportunities_by_strategy(
            analyzed_opportunities, strategy_params
        )
        
        # Calculate position sizing
        position_sizing = self.risk_manager.calculate_position_sizing(
            available_opportunities=filtered_opportunities,
            bankroll=bankroll,
            risk_budget=strategy_params['daily_risk_budget']
        )
        
        # Kelly optimization for multiple bets
        kelly_opportunities = []
        for opp in filtered_opportunities:
            kelly_opportunities.append({
                'id': opp.get('opportunity_id', 'unknown'),
                'probability': opp.get('probability', 0.5),
                'odds': opp.get('odds', 2.0),
                'match_info': opp.get('match_info', {}),
                'bet_type': opp.get('bet_type', 'Unknown')
            })
        
        if kelly_opportunities:
            kelly_analysis = self.kelly.analyze_multiple_bets(
                bets=kelly_opportunities,
                bankroll=bankroll,
                fractional_kelly=strategy_params['kelly_fraction']
            )
        else:
            kelly_analysis = {'message': 'No suitable opportunities for Kelly analysis'}
        
        return {
            'strategy_type': strategy_type,
            'strategy_parameters': strategy_params,
            'total_opportunities_analyzed': len(opportunities),
            'filtered_opportunities': len(filtered_opportunities),
            'analyzed_opportunities': analyzed_opportunities,
            'position_sizing': position_sizing,
            'kelly_analysis': kelly_analysis,
            'final_recommendations': self._generate_strategy_recommendations(
                filtered_opportunities, position_sizing, kelly_analysis, strategy_params
            )
        }
    
    def generate_betting_report(self, 
                              bankroll: float,
                              opportunities: List[Dict[str, Any]] = None,
                              days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive betting intelligence report
        
        Args:
            bankroll: Current bankroll
            opportunities: Current betting opportunities
            days: Days to analyze for historical performance
            
        Returns:
            Comprehensive betting report
        """
        report = {
            'report_timestamp': self._get_current_timestamp(),
            'bankroll': bankroll,
            'analysis_period_days': days
        }
        
        # Portfolio Performance
        report['portfolio_analysis'] = self.analyze_portfolio_performance(bankroll, days)
        
        # Current Opportunities Analysis
        if opportunities:
            report['opportunities_analysis'] = []
            for opp in opportunities:
                opp_analysis = self.analyze_betting_opportunity(opp, bankroll)
                report['opportunities_analysis'].append(opp_analysis)
            
            # Strategy Optimization
            report['strategy_optimization'] = self.optimize_betting_strategy(
                opportunities, bankroll, 'balanced'
            )
        
        # Executive Summary
        report['executive_summary'] = self._generate_executive_summary(report)
        
        return report
    
    def _generate_overall_recommendation(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate overall recommendation for a betting opportunity"""
        recommendations = []
        confidence_level = "Medium"
        
        # Kelly analysis
        kelly = analysis.get('kelly_analysis', {})
        if kelly.get('should_bet', False):
            recommendations.append(f"Kelly recommends betting {kelly.get('recommended_stake_percentage', 0):.1f}% of bankroll")
            if kelly.get('expected_value', 0) > 0.1:
                confidence_level = "High"
        else:
            recommendations.append("Kelly analysis suggests avoiding this bet")
            confidence_level = "Low"
        
        # Risk analysis
        risk = analysis.get('risk_analysis', {})
        if risk.get('all_checks_passed', False):
            recommendations.append("All risk checks passed")
        else:
            recommendations.extend(risk.get('recommendations', []))
            confidence_level = "Low"
        
        # Arbitrage analysis
        arbitrage = analysis.get('arbitrage_analysis', {})
        if arbitrage.get('arbitrage_exists', False):
            recommendations.append(f"Arbitrage opportunity: {arbitrage.get('profit_percentage', 0):.2f}% guaranteed profit")
            confidence_level = "Very High"
        
        # Overall recommendation
        if confidence_level in ["High", "Very High"] and kelly.get('should_bet', False):
            overall = "RECOMMENDED"
        elif confidence_level == "Medium" and kelly.get('should_bet', False):
            overall = "CONSIDER"
        else:
            overall = "AVOID"
        
        return {
            'overall': overall,
            'confidence_level': confidence_level,
            'reasons': recommendations
        }
    
    def _generate_portfolio_recommendations(self, 
                                          portfolio_summary: Dict[str, Any],
                                          portfolio_risk: Dict[str, Any],
                                          drawdown_analysis: Dict[str, Any]) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        roi = portfolio_summary.get('roi_percentage', 0)
        win_rate = portfolio_summary.get('win_rate', 0)
        
        if roi < 0:
            recommendations.append("Negative ROI - review betting strategy and selection criteria")
        elif roi < 5:
            recommendations.append("Low ROI - consider focusing on higher value opportunities")
        
        if win_rate < 40:
            recommendations.append("Low win rate - focus on higher probability bets")
        elif win_rate > 70:
            recommendations.append("High win rate - consider taking on slightly higher risk for better returns")
        
        # Risk-based recommendations
        if portfolio_risk.get('portfolio_risk_level') == 'High':
            recommendations.append("High portfolio risk - reduce position sizes or diversify")
        
        # Drawdown recommendations
        if drawdown_analysis.get('should_stop_betting', False):
            recommendations.append("STOP BETTING: Maximum drawdown limit reached")
        elif drawdown_analysis.get('stop_loss_triggered', False):
            recommendations.append("Reduce betting activity - stop loss threshold triggered")
        
        return recommendations
    
    def _generate_arbitrage_recommendations(self, 
                                          opportunities: List[Dict[str, Any]],
                                          allocation: Dict[str, Any]) -> List[str]:
        """Generate arbitrage-specific recommendations"""
        recommendations = []
        
        if not opportunities:
            recommendations.append("No arbitrage opportunities found - monitor odds for changes")
            return recommendations
        
        total_profit = allocation.get('total_expected_profit', 0)
        roi = allocation.get('expected_roi', 0)
        
        recommendations.append(f"Found {len(opportunities)} arbitrage opportunities")
        recommendations.append(f"Total expected profit: ${total_profit:.2f} ({roi:.1f}% ROI)")
        
        if roi > 5:
            recommendations.append("Excellent arbitrage opportunities - execute quickly")
        elif roi > 2:
            recommendations.append("Good arbitrage opportunities - monitor for odds changes")
        else:
            recommendations.append("Low margin arbitrage - ensure execution speed")
        
        # Risk warnings
        high_risk_opps = [opp for opp in opportunities if opp.get('risk_level') == 'High']
        if high_risk_opps:
            recommendations.append(f"Warning: {len(high_risk_opps)} high-risk arbitrage opportunities")
        
        return recommendations
    
    def _get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Get parameters for different strategy types"""
        strategies = {
            'conservative': {
                'kelly_fraction': 0.1,
                'daily_risk_budget': 0.01,
                'min_probability': 0.6,
                'min_expected_value': 0.05,
                'max_odds': 3.0
            },
            'balanced': {
                'kelly_fraction': 0.25,
                'daily_risk_budget': 0.02,
                'min_probability': 0.5,
                'min_expected_value': 0.02,
                'max_odds': 5.0
            },
            'aggressive': {
                'kelly_fraction': 0.5,
                'daily_risk_budget': 0.05,
                'min_probability': 0.4,
                'min_expected_value': 0.01,
                'max_odds': 10.0
            }
        }
        
        return strategies.get(strategy_type, strategies['balanced'])
    
    def _filter_opportunities_by_strategy(self, 
                                        opportunities: List[Dict[str, Any]],
                                        strategy_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter opportunities based on strategy parameters"""
        filtered = []
        
        for opp in opportunities:
            # Check if opportunity meets strategy criteria
            probability = opp.get('probability', 0.5)
            odds = opp.get('odds', 2.0)
            
            kelly_analysis = opp.get('kelly_analysis', {})
            expected_value = kelly_analysis.get('expected_value', 0)
            
            if (probability >= strategy_params['min_probability'] and
                odds <= strategy_params['max_odds'] and
                expected_value >= strategy_params['min_expected_value']):
                filtered.append(opp)
        
        return filtered
    
    def _generate_strategy_recommendations(self, 
                                         opportunities: List[Dict[str, Any]],
                                         position_sizing: Dict[str, Any],
                                         kelly_analysis: Dict[str, Any],
                                         strategy_params: Dict[str, Any]) -> List[str]:
        """Generate strategy-specific recommendations"""
        recommendations = []
        
        total_opportunities = len(opportunities)
        recommended_bets = position_sizing.get('recommended_bets', 0)
        
        recommendations.append(f"Strategy filtered {total_opportunities} opportunities")
        recommendations.append(f"Recommending {recommended_bets} bets")
        
        if recommended_bets == 0:
            recommendations.append("No opportunities meet strategy criteria - wait for better spots")
        elif recommended_bets > 5:
            recommendations.append("Many opportunities available - ensure proper diversification")
        
        # Kelly recommendations
        if isinstance(kelly_analysis, dict) and 'positive_ev_bets' in kelly_analysis:
            positive_ev = kelly_analysis['positive_ev_bets']
            recommendations.append(f"{positive_ev} opportunities have positive expected value")
        
        return recommendations
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, str]:
        """Generate executive summary of the betting report"""
        portfolio = report.get('portfolio_analysis', {}).get('portfolio_summary', {})
        
        # Key metrics
        total_bets = portfolio.get('total_bets', 0)
        roi = portfolio.get('roi_percentage', 0)
        win_rate = portfolio.get('win_rate', 0)
        
        # Performance assessment
        if roi > 10:
            performance = "Excellent"
        elif roi > 5:
            performance = "Good"
        elif roi > 0:
            performance = "Positive"
        else:
            performance = "Needs Improvement"
        
        # Risk assessment
        risk_analysis = report.get('portfolio_analysis', {}).get('portfolio_risk', {})
        risk_level = risk_analysis.get('portfolio_risk_level', 'Unknown')
        
        return {
            'performance_summary': f"{performance} performance with {roi:.1f}% ROI over {total_bets} bets",
            'risk_summary': f"Portfolio risk level: {risk_level}",
            'win_rate_summary': f"Win rate: {win_rate:.1f}%",
            'key_recommendation': self._get_key_recommendation(roi, win_rate, risk_level)
        }
    
    def _get_key_recommendation(self, roi: float, win_rate: float, risk_level: str) -> str:
        """Get key recommendation based on performance metrics"""
        if roi < 0:
            return "Focus on improving bet selection and reducing losses"
        elif risk_level == 'High':
            return "Reduce position sizes and improve risk management"
        elif win_rate < 45:
            return "Focus on higher probability bets to improve win rate"
        elif roi > 15:
            return "Excellent performance - maintain current strategy"
        else:
            return "Solid performance - look for opportunities to optimize further"
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
