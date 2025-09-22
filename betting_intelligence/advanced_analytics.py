"""
Advanced Analytics Module for FootyForecast Betting Intelligence

Provides detailed analysis of betting patterns, streaks, seasonal performance,
league profitability, and bet type optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from collections import defaultdict, Counter
import calendar

class AdvancedAnalytics:
    """
    Advanced analytics for betting performance analysis
    """
    
    def __init__(self, portfolio_tracker=None):
        """
        Initialize advanced analytics
        
        Args:
            portfolio_tracker: PortfolioTracker instance
        """
        self.portfolio_tracker = portfolio_tracker
    
    def analyze_streaks(self, betting_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed streak analysis of wins/losses
        
        Args:
            betting_history: List of settled bets
            
        Returns:
            Comprehensive streak analysis
        """
        if not betting_history:
            return {'error': 'No betting history provided'}
        
        # Sort by timestamp
        sorted_bets = sorted(betting_history, key=lambda x: x.get('timestamp', ''))
        
        # Calculate streaks
        streaks = []
        current_streak = {'type': None, 'length': 0, 'start_date': None, 'end_date': None, 'profit_loss': 0}
        
        for bet in sorted_bets:
            if bet['status'] not in ['won', 'lost']:
                continue
            
            bet_result = 'win' if bet['status'] == 'won' else 'loss'
            bet_date = bet.get('timestamp', '')
            bet_profit = bet.get('profit_loss', 0)
            
            if current_streak['type'] == bet_result:
                # Continue current streak
                current_streak['length'] += 1
                current_streak['end_date'] = bet_date
                current_streak['profit_loss'] += bet_profit
            else:
                # End current streak and start new one
                if current_streak['type'] is not None:
                    streaks.append(current_streak.copy())
                
                current_streak = {
                    'type': bet_result,
                    'length': 1,
                    'start_date': bet_date,
                    'end_date': bet_date,
                    'profit_loss': bet_profit
                }
        
        # Add final streak
        if current_streak['type'] is not None:
            streaks.append(current_streak)
        
        # Analyze streaks
        win_streaks = [s for s in streaks if s['type'] == 'win']
        loss_streaks = [s for s in streaks if s['type'] == 'loss']
        
        # Statistics
        longest_win_streak = max(win_streaks, key=lambda x: x['length']) if win_streaks else None
        longest_loss_streak = max(loss_streaks, key=lambda x: x['length']) if loss_streaks else None
        
        avg_win_streak = np.mean([s['length'] for s in win_streaks]) if win_streaks else 0
        avg_loss_streak = np.mean([s['length'] for s in loss_streaks]) if loss_streaks else 0
        
        # Current streak
        current_active_streak = streaks[-1] if streaks else None
        
        # Streak distribution
        win_streak_distribution = Counter([s['length'] for s in win_streaks])
        loss_streak_distribution = Counter([s['length'] for s in loss_streaks])
        
        return {
            'total_streaks': len(streaks),
            'win_streaks_count': len(win_streaks),
            'loss_streaks_count': len(loss_streaks),
            'longest_win_streak': {
                'length': longest_win_streak['length'] if longest_win_streak else 0,
                'profit': longest_win_streak['profit_loss'] if longest_win_streak else 0,
                'period': f"{longest_win_streak['start_date']} to {longest_win_streak['end_date']}" if longest_win_streak else None
            },
            'longest_loss_streak': {
                'length': longest_loss_streak['length'] if longest_loss_streak else 0,
                'loss': longest_loss_streak['profit_loss'] if longest_loss_streak else 0,
                'period': f"{longest_loss_streak['start_date']} to {longest_loss_streak['end_date']}" if longest_loss_streak else None
            },
            'average_win_streak': round(avg_win_streak, 2),
            'average_loss_streak': round(avg_loss_streak, 2),
            'current_streak': {
                'type': current_active_streak['type'] if current_active_streak else None,
                'length': current_active_streak['length'] if current_active_streak else 0,
                'profit_loss': current_active_streak['profit_loss'] if current_active_streak else 0
            },
            'win_streak_distribution': dict(win_streak_distribution),
            'loss_streak_distribution': dict(loss_streak_distribution),
            'streak_analysis': self._analyze_streak_patterns(streaks)
        }
    
    def analyze_seasonal_performance(self, betting_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance by season, month, day of week, time of day
        
        Args:
            betting_history: List of settled bets
            
        Returns:
            Seasonal performance analysis
        """
        if not betting_history:
            return {'error': 'No betting history provided'}
        
        # Initialize performance trackers
        monthly_performance = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0})
        weekday_performance = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0})
        hourly_performance = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0})
        quarterly_performance = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0})
        
        for bet in betting_history:
            if bet['status'] not in ['won', 'lost', 'void']:
                continue
            
            try:
                bet_date = datetime.fromisoformat(bet['timestamp'])
            except:
                continue
            
            profit_loss = bet.get('profit_loss', 0)
            stake = bet.get('stake', 0)
            is_win = bet['status'] == 'won'
            
            # Monthly analysis
            month_key = bet_date.strftime('%Y-%m')
            monthly_performance[month_key]['bets'] += 1
            monthly_performance[month_key]['wins'] += 1 if is_win else 0
            monthly_performance[month_key]['profit_loss'] += profit_loss
            monthly_performance[month_key]['stakes'] += stake
            
            # Weekday analysis
            weekday = calendar.day_name[bet_date.weekday()]
            weekday_performance[weekday]['bets'] += 1
            weekday_performance[weekday]['wins'] += 1 if is_win else 0
            weekday_performance[weekday]['profit_loss'] += profit_loss
            weekday_performance[weekday]['stakes'] += stake
            
            # Hourly analysis
            hour = bet_date.hour
            hourly_performance[hour]['bets'] += 1
            hourly_performance[hour]['wins'] += 1 if is_win else 0
            hourly_performance[hour]['profit_loss'] += profit_loss
            hourly_performance[hour]['stakes'] += stake
            
            # Quarterly analysis
            quarter = f"Q{(bet_date.month-1)//3 + 1}-{bet_date.year}"
            quarterly_performance[quarter]['bets'] += 1
            quarterly_performance[quarter]['wins'] += 1 if is_win else 0
            quarterly_performance[quarter]['profit_loss'] += profit_loss
            quarterly_performance[quarter]['stakes'] += stake
        
        # Calculate performance metrics
        def calculate_metrics(performance_dict):
            metrics = {}
            for period, data in performance_dict.items():
                if data['bets'] > 0:
                    win_rate = (data['wins'] / data['bets']) * 100
                    roi = (data['profit_loss'] / data['stakes']) * 100 if data['stakes'] > 0 else 0
                    avg_stake = data['stakes'] / data['bets']
                    
                    metrics[period] = {
                        'total_bets': data['bets'],
                        'wins': data['wins'],
                        'win_rate': round(win_rate, 2),
                        'profit_loss': round(data['profit_loss'], 2),
                        'roi': round(roi, 2),
                        'total_stakes': round(data['stakes'], 2),
                        'average_stake': round(avg_stake, 2)
                    }
            return metrics
        
        monthly_metrics = calculate_metrics(monthly_performance)
        weekday_metrics = calculate_metrics(weekday_performance)
        hourly_metrics = calculate_metrics(hourly_performance)
        quarterly_metrics = calculate_metrics(quarterly_performance)
        
        # Find best and worst periods
        def find_best_worst(metrics_dict, metric='roi'):
            if not metrics_dict:
                return None, None
            
            best = max(metrics_dict.items(), key=lambda x: x[1][metric])
            worst = min(metrics_dict.items(), key=lambda x: x[1][metric])
            return best, worst
        
        best_month, worst_month = find_best_worst(monthly_metrics)
        best_weekday, worst_weekday = find_best_worst(weekday_metrics)
        best_hour, worst_hour = find_best_worst(hourly_metrics)
        
        return {
            'monthly_performance': monthly_metrics,
            'weekday_performance': weekday_metrics,
            'hourly_performance': hourly_metrics,
            'quarterly_performance': quarterly_metrics,
            'best_periods': {
                'month': {'period': best_month[0], 'roi': best_month[1]['roi']} if best_month else None,
                'weekday': {'period': best_weekday[0], 'roi': best_weekday[1]['roi']} if best_weekday else None,
                'hour': {'period': best_hour[0], 'roi': best_hour[1]['roi']} if best_hour else None
            },
            'worst_periods': {
                'month': {'period': worst_month[0], 'roi': worst_month[1]['roi']} if worst_month else None,
                'weekday': {'period': worst_weekday[0], 'roi': worst_weekday[1]['roi']} if worst_weekday else None,
                'hour': {'period': worst_hour[0], 'roi': worst_hour[1]['roi']} if worst_hour else None
            },
            'seasonal_insights': self._generate_seasonal_insights(monthly_metrics, weekday_metrics, hourly_metrics)
        }
    
    def analyze_league_profitability(self, betting_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze profitability by league
        
        Args:
            betting_history: List of settled bets
            
        Returns:
            League profitability analysis
        """
        if not betting_history:
            return {'error': 'No betting history provided'}
        
        league_performance = defaultdict(lambda: {
            'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0,
            'bet_types': defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0}),
            'odds_ranges': defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0})
        })
        
        for bet in betting_history:
            if bet['status'] not in ['won', 'lost', 'void']:
                continue
            
            league = bet.get('match_info', {}).get('league', 'Unknown')
            bet_type = bet.get('bet_type', 'Unknown')
            odds = bet.get('odds', 0)
            profit_loss = bet.get('profit_loss', 0)
            stake = bet.get('stake', 0)
            is_win = bet['status'] == 'won'
            
            # Overall league performance
            league_performance[league]['bets'] += 1
            league_performance[league]['wins'] += 1 if is_win else 0
            league_performance[league]['profit_loss'] += profit_loss
            league_performance[league]['stakes'] += stake
            
            # Bet type performance within league
            league_performance[league]['bet_types'][bet_type]['bets'] += 1
            league_performance[league]['bet_types'][bet_type]['wins'] += 1 if is_win else 0
            league_performance[league]['bet_types'][bet_type]['profit_loss'] += profit_loss
            
            # Odds range performance within league
            odds_range = self._get_odds_range(odds)
            league_performance[league]['odds_ranges'][odds_range]['bets'] += 1
            league_performance[league]['odds_ranges'][odds_range]['wins'] += 1 if is_win else 0
            league_performance[league]['odds_ranges'][odds_range]['profit_loss'] += profit_loss
        
        # Calculate metrics for each league
        league_metrics = {}
        for league, data in league_performance.items():
            if data['bets'] > 0:
                win_rate = (data['wins'] / data['bets']) * 100
                roi = (data['profit_loss'] / data['stakes']) * 100 if data['stakes'] > 0 else 0
                avg_stake = data['stakes'] / data['bets']
                
                # Best bet types in this league
                best_bet_types = {}
                for bet_type, bt_data in data['bet_types'].items():
                    if bt_data['bets'] >= 3:  # Minimum 3 bets for reliability
                        bt_win_rate = (bt_data['wins'] / bt_data['bets']) * 100
                        best_bet_types[bet_type] = {
                            'bets': bt_data['bets'],
                            'win_rate': round(bt_win_rate, 2),
                            'profit_loss': round(bt_data['profit_loss'], 2)
                        }
                
                # Best odds ranges in this league
                best_odds_ranges = {}
                for odds_range, or_data in data['odds_ranges'].items():
                    if or_data['bets'] >= 3:
                        or_win_rate = (or_data['wins'] / or_data['bets']) * 100
                        best_odds_ranges[odds_range] = {
                            'bets': or_data['bets'],
                            'win_rate': round(or_win_rate, 2),
                            'profit_loss': round(or_data['profit_loss'], 2)
                        }
                
                league_metrics[league] = {
                    'total_bets': data['bets'],
                    'wins': data['wins'],
                    'win_rate': round(win_rate, 2),
                    'profit_loss': round(data['profit_loss'], 2),
                    'roi': round(roi, 2),
                    'total_stakes': round(data['stakes'], 2),
                    'average_stake': round(avg_stake, 2),
                    'best_bet_types': best_bet_types,
                    'odds_range_performance': best_odds_ranges,
                    'profitability_grade': self._grade_profitability(roi, win_rate)
                }
        
        # Rank leagues by profitability
        sorted_leagues = sorted(league_metrics.items(), key=lambda x: x[1]['roi'], reverse=True)
        
        return {
            'league_performance': league_metrics,
            'league_rankings': {
                'most_profitable': sorted_leagues[:5] if len(sorted_leagues) >= 5 else sorted_leagues,
                'least_profitable': sorted_leagues[-5:] if len(sorted_leagues) >= 5 else []
            },
            'league_insights': self._generate_league_insights(league_metrics),
            'recommendations': self._generate_league_recommendations(league_metrics)
        }
    
    def analyze_bet_type_optimization(self, betting_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze which bet types are most profitable
        
        Args:
            betting_history: List of settled bets
            
        Returns:
            Bet type optimization analysis
        """
        if not betting_history:
            return {'error': 'No betting history provided'}
        
        bet_type_performance = defaultdict(lambda: {
            'bets': 0, 'wins': 0, 'profit_loss': 0, 'stakes': 0,
            'odds_distribution': [], 'confidence_distribution': [],
            'league_performance': defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit_loss': 0})
        })
        
        for bet in betting_history:
            if bet['status'] not in ['won', 'lost', 'void']:
                continue
            
            bet_type = bet.get('bet_type', 'Unknown')
            league = bet.get('match_info', {}).get('league', 'Unknown')
            odds = bet.get('odds', 0)
            confidence = bet.get('prediction_confidence', 0)
            profit_loss = bet.get('profit_loss', 0)
            stake = bet.get('stake', 0)
            is_win = bet['status'] == 'won'
            
            # Overall bet type performance
            bet_type_performance[bet_type]['bets'] += 1
            bet_type_performance[bet_type]['wins'] += 1 if is_win else 0
            bet_type_performance[bet_type]['profit_loss'] += profit_loss
            bet_type_performance[bet_type]['stakes'] += stake
            bet_type_performance[bet_type]['odds_distribution'].append(odds)
            bet_type_performance[bet_type]['confidence_distribution'].append(confidence)
            
            # League performance for this bet type
            bet_type_performance[bet_type]['league_performance'][league]['bets'] += 1
            bet_type_performance[bet_type]['league_performance'][league]['wins'] += 1 if is_win else 0
            bet_type_performance[bet_type]['league_performance'][league]['profit_loss'] += profit_loss
        
        # Calculate metrics for each bet type
        bet_type_metrics = {}
        for bet_type, data in bet_type_performance.items():
            if data['bets'] > 0:
                win_rate = (data['wins'] / data['bets']) * 100
                roi = (data['profit_loss'] / data['stakes']) * 100 if data['stakes'] > 0 else 0
                avg_stake = data['stakes'] / data['bets']
                avg_odds = np.mean(data['odds_distribution']) if data['odds_distribution'] else 0
                avg_confidence = np.mean(data['confidence_distribution']) if data['confidence_distribution'] else 0
                
                # Best leagues for this bet type
                best_leagues = {}
                for league, league_data in data['league_performance'].items():
                    if league_data['bets'] >= 3:
                        league_win_rate = (league_data['wins'] / league_data['bets']) * 100
                        best_leagues[league] = {
                            'bets': league_data['bets'],
                            'win_rate': round(league_win_rate, 2),
                            'profit_loss': round(league_data['profit_loss'], 2)
                        }
                
                bet_type_metrics[bet_type] = {
                    'total_bets': data['bets'],
                    'wins': data['wins'],
                    'win_rate': round(win_rate, 2),
                    'profit_loss': round(data['profit_loss'], 2),
                    'roi': round(roi, 2),
                    'total_stakes': round(data['stakes'], 2),
                    'average_stake': round(avg_stake, 2),
                    'average_odds': round(avg_odds, 2),
                    'average_confidence': round(avg_confidence, 2),
                    'best_leagues': best_leagues,
                    'optimization_score': self._calculate_optimization_score(roi, win_rate, data['bets']),
                    'recommendation': self._get_bet_type_recommendation(roi, win_rate, data['bets'])
                }
        
        # Rank bet types by optimization score
        sorted_bet_types = sorted(bet_type_metrics.items(), 
                                key=lambda x: x[1]['optimization_score'], reverse=True)
        
        return {
            'bet_type_performance': bet_type_metrics,
            'bet_type_rankings': {
                'most_profitable': sorted_bet_types[:3] if len(sorted_bet_types) >= 3 else sorted_bet_types,
                'least_profitable': sorted_bet_types[-3:] if len(sorted_bet_types) >= 3 else []
            },
            'optimization_insights': self._generate_bet_type_insights(bet_type_metrics),
            'recommendations': self._generate_bet_type_recommendations(bet_type_metrics)
        }
    
    def generate_comprehensive_analytics_report(self, betting_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report combining all analyses
        
        Args:
            betting_history: List of settled bets
            
        Returns:
            Comprehensive analytics report
        """
        if not betting_history:
            return {'error': 'No betting history provided'}
        
        # Run all analyses
        streak_analysis = self.analyze_streaks(betting_history)
        seasonal_analysis = self.analyze_seasonal_performance(betting_history)
        league_analysis = self.analyze_league_profitability(betting_history)
        bet_type_analysis = self.analyze_bet_type_optimization(betting_history)
        
        # Generate executive summary
        executive_summary = self._generate_analytics_executive_summary(
            streak_analysis, seasonal_analysis, league_analysis, bet_type_analysis
        )
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_bets_analyzed': len(betting_history),
            'executive_summary': executive_summary,
            'streak_analysis': streak_analysis,
            'seasonal_analysis': seasonal_analysis,
            'league_analysis': league_analysis,
            'bet_type_analysis': bet_type_analysis,
            'key_recommendations': self._generate_key_recommendations(
                streak_analysis, seasonal_analysis, league_analysis, bet_type_analysis
            )
        }
    
    def _analyze_streak_patterns(self, streaks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in streaks"""
        if not streaks:
            return {}
        
        # Analyze alternating patterns
        alternating_count = 0
        for i in range(len(streaks) - 1):
            if streaks[i]['type'] != streaks[i+1]['type']:
                alternating_count += 1
        
        alternating_percentage = (alternating_count / (len(streaks) - 1)) * 100 if len(streaks) > 1 else 0
        
        return {
            'alternating_percentage': round(alternating_percentage, 2),
            'pattern_type': 'Highly Alternating' if alternating_percentage > 70 else 
                          'Moderately Alternating' if alternating_percentage > 40 else 'Streaky',
            'total_streak_periods': len(streaks)
        }
    
    def _generate_seasonal_insights(self, monthly: Dict, weekday: Dict, hourly: Dict) -> List[str]:
        """Generate insights from seasonal analysis"""
        insights = []
        
        # Monthly insights
        if monthly:
            best_month = max(monthly.items(), key=lambda x: x[1]['roi'])
            insights.append(f"Best performing month: {best_month[0]} with {best_month[1]['roi']:.1f}% ROI")
        
        # Weekday insights
        if weekday:
            best_day = max(weekday.items(), key=lambda x: x[1]['roi'])
            insights.append(f"Best performing day: {best_day[0]} with {best_day[1]['roi']:.1f}% ROI")
        
        # Hourly insights
        if hourly:
            best_hour = max(hourly.items(), key=lambda x: x[1]['roi'])
            insights.append(f"Best performing hour: {best_hour[0]}:00 with {best_hour[1]['roi']:.1f}% ROI")
        
        return insights
    
    def _get_odds_range(self, odds: float) -> str:
        """Categorize odds into ranges"""
        if odds < 1.5:
            return "1.0-1.5"
        elif odds < 2.0:
            return "1.5-2.0"
        elif odds < 3.0:
            return "2.0-3.0"
        elif odds < 5.0:
            return "3.0-5.0"
        else:
            return "5.0+"
    
    def _grade_profitability(self, roi: float, win_rate: float) -> str:
        """Grade profitability performance"""
        if roi > 15 and win_rate > 60:
            return "A+"
        elif roi > 10 and win_rate > 55:
            return "A"
        elif roi > 5 and win_rate > 50:
            return "B+"
        elif roi > 0 and win_rate > 45:
            return "B"
        elif roi > -5:
            return "C"
        else:
            return "D"
    
    def _generate_league_insights(self, league_metrics: Dict) -> List[str]:
        """Generate insights from league analysis"""
        insights = []
        
        if not league_metrics:
            return insights
        
        # Find most profitable league
        best_league = max(league_metrics.items(), key=lambda x: x[1]['roi'])
        insights.append(f"Most profitable league: {best_league[0]} ({best_league[1]['roi']:.1f}% ROI)")
        
        # Find league with highest win rate
        best_win_rate_league = max(league_metrics.items(), key=lambda x: x[1]['win_rate'])
        insights.append(f"Highest win rate: {best_win_rate_league[0]} ({best_win_rate_league[1]['win_rate']:.1f}%)")
        
        # Count profitable leagues
        profitable_leagues = sum(1 for metrics in league_metrics.values() if metrics['roi'] > 0)
        insights.append(f"Profitable leagues: {profitable_leagues}/{len(league_metrics)}")
        
        return insights
    
    def _generate_league_recommendations(self, league_metrics: Dict) -> List[str]:
        """Generate recommendations from league analysis"""
        recommendations = []
        
        if not league_metrics:
            return recommendations
        
        # Identify leagues to focus on
        profitable_leagues = [name for name, metrics in league_metrics.items() if metrics['roi'] > 5]
        if profitable_leagues:
            recommendations.append(f"Focus on profitable leagues: {', '.join(profitable_leagues[:3])}")
        
        # Identify leagues to avoid
        unprofitable_leagues = [name for name, metrics in league_metrics.items() if metrics['roi'] < -10]
        if unprofitable_leagues:
            recommendations.append(f"Consider avoiding: {', '.join(unprofitable_leagues[:3])}")
        
        return recommendations
    
    def _calculate_optimization_score(self, roi: float, win_rate: float, bet_count: int) -> float:
        """Calculate optimization score for bet types"""
        # Weighted score considering ROI, win rate, and sample size
        roi_score = roi / 100  # Normalize ROI
        win_rate_score = (win_rate - 50) / 50  # Normalize win rate around 50%
        sample_size_weight = min(bet_count / 20, 1.0)  # Weight by sample size (max at 20 bets)
        
        return (roi_score * 0.6 + win_rate_score * 0.4) * sample_size_weight
    
    def _get_bet_type_recommendation(self, roi: float, win_rate: float, bet_count: int) -> str:
        """Get recommendation for bet type"""
        if bet_count < 5:
            return "Insufficient data"
        elif roi > 10 and win_rate > 55:
            return "Highly Recommended"
        elif roi > 5 and win_rate > 50:
            return "Recommended"
        elif roi > 0:
            return "Consider"
        else:
            return "Avoid"
    
    def _generate_bet_type_insights(self, bet_type_metrics: Dict) -> List[str]:
        """Generate insights from bet type analysis"""
        insights = []
        
        if not bet_type_metrics:
            return insights
        
        # Most profitable bet type
        best_bet_type = max(bet_type_metrics.items(), key=lambda x: x[1]['roi'])
        insights.append(f"Most profitable bet type: {best_bet_type[0]} ({best_bet_type[1]['roi']:.1f}% ROI)")
        
        # Highest win rate bet type
        best_win_rate_type = max(bet_type_metrics.items(), key=lambda x: x[1]['win_rate'])
        insights.append(f"Highest win rate: {best_win_rate_type[0]} ({best_win_rate_type[1]['win_rate']:.1f}%)")
        
        return insights
    
    def _generate_bet_type_recommendations(self, bet_type_metrics: Dict) -> List[str]:
        """Generate recommendations from bet type analysis"""
        recommendations = []
        
        if not bet_type_metrics:
            return recommendations
        
        # Recommended bet types
        recommended_types = [name for name, metrics in bet_type_metrics.items() 
                           if metrics['recommendation'] in ['Highly Recommended', 'Recommended']]
        if recommended_types:
            recommendations.append(f"Focus on: {', '.join(recommended_types)}")
        
        # Bet types to avoid
        avoid_types = [name for name, metrics in bet_type_metrics.items() 
                      if metrics['recommendation'] == 'Avoid']
        if avoid_types:
            recommendations.append(f"Avoid: {', '.join(avoid_types)}")
        
        return recommendations
    
    def _generate_analytics_executive_summary(self, streak_analysis: Dict, seasonal_analysis: Dict, 
                                            league_analysis: Dict, bet_type_analysis: Dict) -> Dict[str, str]:
        """Generate executive summary of analytics"""
        summary = {}
        
        # Streak summary
        current_streak = streak_analysis.get('current_streak', {})
        if current_streak.get('type'):
            summary['current_form'] = f"Currently on a {current_streak['length']}-bet {current_streak['type']} streak"
        
        # Seasonal summary
        best_periods = seasonal_analysis.get('best_periods', {})
        if best_periods.get('weekday'):
            summary['best_timing'] = f"Best performance on {best_periods['weekday']['period']}s"
        
        # League summary
        league_rankings = league_analysis.get('league_rankings', {})
        if league_rankings.get('most_profitable'):
            best_league = league_rankings['most_profitable'][0]
            summary['best_league'] = f"Most profitable in {best_league[0]} ({best_league[1]['roi']:.1f}% ROI)"
        
        # Bet type summary
        bet_type_rankings = bet_type_analysis.get('bet_type_rankings', {})
        if bet_type_rankings.get('most_profitable'):
            best_bet_type = bet_type_rankings['most_profitable'][0]
            summary['best_bet_type'] = f"Best bet type: {best_bet_type[0]} ({best_bet_type[1]['roi']:.1f}% ROI)"
        
        return summary
    
    def _generate_key_recommendations(self, streak_analysis: Dict, seasonal_analysis: Dict,
                                    league_analysis: Dict, bet_type_analysis: Dict) -> List[str]:
        """Generate key recommendations from all analyses"""
        recommendations = []
        
        # Streak-based recommendations
        current_streak = streak_analysis.get('current_streak', {})
        if current_streak.get('type') == 'loss' and current_streak.get('length', 0) >= 3:
            recommendations.append("Consider reducing stake sizes due to current losing streak")
        
        # Seasonal recommendations
        best_periods = seasonal_analysis.get('best_periods', {})
        if best_periods.get('weekday'):
            recommendations.append(f"Focus betting activity on {best_periods['weekday']['period']}s")
        
        # League recommendations
        league_recs = league_analysis.get('recommendations', [])
        recommendations.extend(league_recs[:2])  # Top 2 league recommendations
        
        # Bet type recommendations
        bet_type_recs = bet_type_analysis.get('recommendations', [])
        recommendations.extend(bet_type_recs[:2])  # Top 2 bet type recommendations
        
        return recommendations[:5]  # Limit to top 5 recommendations
