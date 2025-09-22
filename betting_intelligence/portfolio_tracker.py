"""
Portfolio Tracker for FootyForecast

Tracks betting performance, calculates ROI, and provides detailed analytics
of betting history and profitability.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os

class PortfolioTracker:
    """
    Comprehensive betting portfolio tracker with ROI analysis
    """
    
    def __init__(self, data_file: str = "betting_portfolio.json"):
        """
        Initialize portfolio tracker
        
        Args:
            data_file: File to store betting data
        """
        self.data_file = data_file
        self.bets = self._load_data()
    
    def add_bet(self, 
                match_info: Dict[str, str],
                bet_type: str,
                odds: float,
                stake: float,
                prediction_confidence: float,
                bookmaker: str = "Unknown") -> str:
        """
        Add a new bet to the portfolio
        
        Args:
            match_info: Match details (home_team, away_team, date, league)
            bet_type: Type of bet (Home, Draw, Away, Over2.5, Under2.5)
            odds: Betting odds
            stake: Amount staked
            prediction_confidence: Model confidence (0-1)
            bookmaker: Bookmaker name
            
        Returns:
            Bet ID
        """
        bet_id = f"bet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        bet = {
            'id': bet_id,
            'timestamp': datetime.now().isoformat(),
            'match_info': match_info,
            'bet_type': bet_type,
            'odds': odds,
            'stake': stake,
            'prediction_confidence': prediction_confidence,
            'bookmaker': bookmaker,
            'status': 'pending',  # pending, won, lost, void
            'result': None,
            'profit_loss': 0.0,
            'settled_date': None
        }
        
        self.bets.append(bet)
        self._save_data()
        
        return bet_id
    
    def settle_bet(self, bet_id: str, result: str) -> Dict[str, Any]:
        """
        Settle a bet with the actual result
        
        Args:
            bet_id: Bet identifier
            result: Actual result (won, lost, void)
            
        Returns:
            Updated bet information
        """
        bet = self._find_bet(bet_id)
        if not bet:
            raise ValueError(f"Bet {bet_id} not found")
        
        bet['result'] = result
        bet['settled_date'] = datetime.now().isoformat()
        
        if result == 'won':
            bet['profit_loss'] = bet['stake'] * (bet['odds'] - 1)
            bet['status'] = 'won'
        elif result == 'lost':
            bet['profit_loss'] = -bet['stake']
            bet['status'] = 'lost'
        elif result == 'void':
            bet['profit_loss'] = 0.0
            bet['status'] = 'void'
        
        self._save_data()
        return bet
    
    def get_portfolio_summary(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary
        
        Args:
            days: Number of days to analyze (None for all time)
            
        Returns:
            Portfolio summary statistics
        """
        filtered_bets = self._filter_bets_by_date(days) if days else self.bets
        settled_bets = [bet for bet in filtered_bets if bet['status'] in ['won', 'lost', 'void']]
        
        if not settled_bets:
            return {'error': 'No settled bets found'}
        
        total_stakes = sum(bet['stake'] for bet in settled_bets)
        total_profit_loss = sum(bet['profit_loss'] for bet in settled_bets)
        total_bets = len(settled_bets)
        won_bets = len([bet for bet in settled_bets if bet['status'] == 'won'])
        lost_bets = len([bet for bet in settled_bets if bet['status'] == 'lost'])
        
        roi = (total_profit_loss / total_stakes) * 100 if total_stakes > 0 else 0
        win_rate = (won_bets / total_bets) * 100 if total_bets > 0 else 0
        
        # Average odds and stakes
        avg_odds = np.mean([bet['odds'] for bet in settled_bets])
        avg_stake = np.mean([bet['stake'] for bet in settled_bets])
        
        # Longest winning/losing streaks
        streaks = self._calculate_streaks(settled_bets)
        
        return {
            'period': f"Last {days} days" if days else "All time",
            'total_bets': total_bets,
            'won_bets': won_bets,
            'lost_bets': lost_bets,
            'pending_bets': len([bet for bet in filtered_bets if bet['status'] == 'pending']),
            'win_rate': round(win_rate, 2),
            'total_staked': round(total_stakes, 2),
            'total_profit_loss': round(total_profit_loss, 2),
            'roi_percentage': round(roi, 2),
            'average_odds': round(avg_odds, 2),
            'average_stake': round(avg_stake, 2),
            'longest_winning_streak': streaks['longest_winning'],
            'longest_losing_streak': streaks['longest_losing'],
            'current_streak': streaks['current_streak'],
            'best_bet': self._get_best_bet(settled_bets),
            'worst_bet': self._get_worst_bet(settled_bets)
        }
    
    def get_performance_by_category(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by different categories
        
        Returns:
            Performance breakdown by bet type, league, bookmaker, etc.
        """
        settled_bets = [bet for bet in self.bets if bet['status'] in ['won', 'lost', 'void']]
        
        if not settled_bets:
            return {'error': 'No settled bets found'}
        
        categories = {
            'bet_type': {},
            'league': {},
            'bookmaker': {},
            'confidence_range': {}
        }
        
        # Group by bet type
        for bet in settled_bets:
            bet_type = bet['bet_type']
            if bet_type not in categories['bet_type']:
                categories['bet_type'][bet_type] = []
            categories['bet_type'][bet_type].append(bet)
        
        # Group by league
        for bet in settled_bets:
            league = bet['match_info'].get('league', 'Unknown')
            if league not in categories['league']:
                categories['league'][league] = []
            categories['league'][league].append(bet)
        
        # Group by bookmaker
        for bet in settled_bets:
            bookmaker = bet['bookmaker']
            if bookmaker not in categories['bookmaker']:
                categories['bookmaker'][bookmaker] = []
            categories['bookmaker'][bookmaker].append(bet)
        
        # Group by confidence range
        confidence_ranges = ['0.0-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        for bet in settled_bets:
            confidence = bet['prediction_confidence']
            if confidence < 0.6:
                range_key = '0.0-0.6'
            elif confidence < 0.7:
                range_key = '0.6-0.7'
            elif confidence < 0.8:
                range_key = '0.7-0.8'
            elif confidence < 0.9:
                range_key = '0.8-0.9'
            else:
                range_key = '0.9-1.0'
            
            if range_key not in categories['confidence_range']:
                categories['confidence_range'][range_key] = []
            categories['confidence_range'][range_key].append(bet)
        
        # Calculate statistics for each category
        results = {}
        for category_name, category_data in categories.items():
            results[category_name] = {}
            for subcategory, bets in category_data.items():
                if bets:  # Only if there are bets in this subcategory
                    total_stakes = sum(bet['stake'] for bet in bets)
                    total_profit_loss = sum(bet['profit_loss'] for bet in bets)
                    won_bets = len([bet for bet in bets if bet['status'] == 'won'])
                    total_bets = len(bets)
                    
                    results[category_name][subcategory] = {
                        'total_bets': total_bets,
                        'won_bets': won_bets,
                        'win_rate': round((won_bets / total_bets) * 100, 2),
                        'total_staked': round(total_stakes, 2),
                        'profit_loss': round(total_profit_loss, 2),
                        'roi': round((total_profit_loss / total_stakes) * 100, 2) if total_stakes > 0 else 0
                    }
        
        return results
    
    def get_monthly_performance(self) -> List[Dict[str, Any]]:
        """
        Get monthly performance breakdown
        
        Returns:
            List of monthly performance data
        """
        settled_bets = [bet for bet in self.bets if bet['status'] in ['won', 'lost', 'void']]
        
        # Group bets by month
        monthly_data = {}
        for bet in settled_bets:
            bet_date = datetime.fromisoformat(bet['timestamp'])
            month_key = bet_date.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(bet)
        
        # Calculate monthly statistics
        monthly_performance = []
        for month, bets in sorted(monthly_data.items()):
            total_stakes = sum(bet['stake'] for bet in bets)
            total_profit_loss = sum(bet['profit_loss'] for bet in bets)
            won_bets = len([bet for bet in bets if bet['status'] == 'won'])
            total_bets = len(bets)
            
            monthly_performance.append({
                'month': month,
                'total_bets': total_bets,
                'won_bets': won_bets,
                'win_rate': round((won_bets / total_bets) * 100, 2),
                'total_staked': round(total_stakes, 2),
                'profit_loss': round(total_profit_loss, 2),
                'roi': round((total_profit_loss / total_stakes) * 100, 2) if total_stakes > 0 else 0
            })
        
        return monthly_performance
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export betting data to CSV
        
        Args:
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            filename = f"betting_portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Flatten the data for CSV export
        flattened_data = []
        for bet in self.bets:
            flat_bet = {
                'bet_id': bet['id'],
                'timestamp': bet['timestamp'],
                'home_team': bet['match_info'].get('home_team', ''),
                'away_team': bet['match_info'].get('away_team', ''),
                'match_date': bet['match_info'].get('date', ''),
                'league': bet['match_info'].get('league', ''),
                'bet_type': bet['bet_type'],
                'odds': bet['odds'],
                'stake': bet['stake'],
                'prediction_confidence': bet['prediction_confidence'],
                'bookmaker': bet['bookmaker'],
                'status': bet['status'],
                'result': bet['result'],
                'profit_loss': bet['profit_loss'],
                'settled_date': bet['settled_date']
            }
            flattened_data.append(flat_bet)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(filename, index=False)
        
        return filename
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load betting data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_data(self):
        """Save betting data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.bets, f, indent=2)
    
    def _find_bet(self, bet_id: str) -> Optional[Dict[str, Any]]:
        """Find bet by ID"""
        for bet in self.bets:
            if bet['id'] == bet_id:
                return bet
        return None
    
    def _filter_bets_by_date(self, days: int) -> List[Dict[str, Any]]:
        """Filter bets by date range"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            bet for bet in self.bets 
            if datetime.fromisoformat(bet['timestamp']) >= cutoff_date
        ]
    
    def _calculate_streaks(self, bets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate winning/losing streaks"""
        if not bets:
            return {'longest_winning': 0, 'longest_losing': 0, 'current_streak': 0}
        
        # Sort by timestamp
        sorted_bets = sorted(bets, key=lambda x: x['timestamp'])
        
        longest_winning = 0
        longest_losing = 0
        current_winning = 0
        current_losing = 0
        
        for bet in sorted_bets:
            if bet['status'] == 'won':
                current_winning += 1
                current_losing = 0
                longest_winning = max(longest_winning, current_winning)
            elif bet['status'] == 'lost':
                current_losing += 1
                current_winning = 0
                longest_losing = max(longest_losing, current_losing)
        
        # Determine current streak
        if sorted_bets:
            last_bet = sorted_bets[-1]
            if last_bet['status'] == 'won':
                current_streak = current_winning
            elif last_bet['status'] == 'lost':
                current_streak = -current_losing
            else:
                current_streak = 0
        else:
            current_streak = 0
        
        return {
            'longest_winning': longest_winning,
            'longest_losing': longest_losing,
            'current_streak': current_streak
        }
    
    def _get_best_bet(self, bets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the most profitable bet"""
        if not bets:
            return None
        
        best_bet = max(bets, key=lambda x: x['profit_loss'])
        return {
            'match': f"{best_bet['match_info'].get('home_team', '')} vs {best_bet['match_info'].get('away_team', '')}",
            'bet_type': best_bet['bet_type'],
            'odds': best_bet['odds'],
            'stake': best_bet['stake'],
            'profit': best_bet['profit_loss']
        }
    
    def _get_worst_bet(self, bets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the least profitable bet"""
        if not bets:
            return None
        
        worst_bet = min(bets, key=lambda x: x['profit_loss'])
        return {
            'match': f"{worst_bet['match_info'].get('home_team', '')} vs {worst_bet['match_info'].get('away_team', '')}",
            'bet_type': worst_bet['bet_type'],
            'odds': worst_bet['odds'],
            'stake': worst_bet['stake'],
            'loss': worst_bet['profit_loss']
        }
