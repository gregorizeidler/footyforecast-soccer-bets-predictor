"""
Auto-Betting System for FootyForecast

Automated betting features including stake calculation, bet queue management,
stop-loss automation, and profit target systems.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time

class BetStatus(Enum):
    QUEUED = "queued"
    PENDING = "pending"
    PLACED = "placed"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"
    VOID = "void"

class AutoBettingMode(Enum):
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"

@dataclass
class AutoBet:
    """Data class for automated bet"""
    id: str
    match_info: Dict[str, Any]
    bet_type: str
    odds: float
    probability: float
    kelly_stake: float
    actual_stake: float
    bookmaker: str
    status: BetStatus
    created_at: datetime
    execute_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    profit_loss: float = 0.0
    notes: str = ""

class AutoBettingSystem:
    """
    Comprehensive auto-betting system with queue management and automation
    """
    
    def __init__(self, 
                 kelly_calculator,
                 risk_manager,
                 portfolio_tracker,
                 data_file: str = "auto_betting_data.json"):
        """
        Initialize auto-betting system
        
        Args:
            kelly_calculator: KellyCalculator instance
            risk_manager: RiskManager instance
            portfolio_tracker: PortfolioTracker instance
            data_file: File to store auto-betting data
        """
        self.kelly = kelly_calculator
        self.risk_manager = risk_manager
        self.portfolio = portfolio_tracker
        self.data_file = data_file
        
        # System settings
        self.mode = AutoBettingMode.MANUAL
        self.is_active = False
        self.bankroll = 1000.0
        self.daily_profit_target = 0.0
        self.daily_loss_limit = 0.0
        self.max_concurrent_bets = 5
        self.min_confidence_threshold = 0.6
        self.max_stake_percentage = 0.05
        
        # Bet queue and tracking
        self.bet_queue: List[AutoBet] = []
        self.active_bets: List[AutoBet] = []
        self.completed_bets: List[AutoBet] = []
        
        # Stop-loss and profit target tracking
        self.daily_profit_loss = 0.0
        self.session_start_time = datetime.now()
        self.stop_loss_triggered = False
        self.profit_target_reached = False
        
        # Load existing data
        self._load_data()
        
        # Background thread for auto-execution
        self.execution_thread = None
        self.stop_execution = False
    
    def configure_system(self, 
                        mode: AutoBettingMode = AutoBettingMode.MANUAL,
                        bankroll: float = None,
                        daily_profit_target: float = 0.0,
                        daily_loss_limit: float = 0.0,
                        max_concurrent_bets: int = 5,
                        min_confidence_threshold: float = 0.6,
                        max_stake_percentage: float = 0.05) -> Dict[str, Any]:
        """
        Configure auto-betting system settings
        
        Args:
            mode: Betting mode (manual, semi-auto, full-auto)
            bankroll: Current bankroll
            daily_profit_target: Daily profit target (0 = no target)
            daily_loss_limit: Daily loss limit (0 = no limit)
            max_concurrent_bets: Maximum concurrent active bets
            min_confidence_threshold: Minimum confidence for auto-betting
            max_stake_percentage: Maximum stake as percentage of bankroll
            
        Returns:
            Configuration confirmation
        """
        self.mode = mode
        if bankroll is not None:
            self.bankroll = bankroll
        self.daily_profit_target = daily_profit_target
        self.daily_loss_limit = daily_loss_limit
        self.max_concurrent_bets = max_concurrent_bets
        self.min_confidence_threshold = min_confidence_threshold
        self.max_stake_percentage = max_stake_percentage
        
        self._save_data()
        
        return {
            'mode': self.mode.value,
            'bankroll': self.bankroll,
            'daily_profit_target': self.daily_profit_target,
            'daily_loss_limit': self.daily_loss_limit,
            'max_concurrent_bets': self.max_concurrent_bets,
            'min_confidence_threshold': self.min_confidence_threshold,
            'max_stake_percentage': self.max_stake_percentage,
            'configuration_updated': True
        }
    
    def add_bet_to_queue(self, 
                        match_info: Dict[str, Any],
                        bet_type: str,
                        odds: float,
                        probability: float,
                        bookmaker: str = "Unknown",
                        execute_at: Optional[datetime] = None,
                        notes: str = "") -> Dict[str, Any]:
        """
        Add a bet to the auto-betting queue
        
        Args:
            match_info: Match information
            bet_type: Type of bet
            odds: Betting odds
            probability: Estimated probability
            bookmaker: Bookmaker name
            execute_at: When to execute the bet (None = immediate)
            notes: Additional notes
            
        Returns:
            Bet queue result
        """
        # Calculate Kelly stake
        try:
            kelly_result = self.kelly.calculate_stake_amount(
                bankroll=self.bankroll,
                probability=probability,
                odds=odds,
                fractional_kelly=0.25
            )
            kelly_stake = kelly_result.get('recommended_stake_amount', 0)
        except:
            kelly_stake = 0
        
        # Risk assessment
        risk_assessment = self.risk_manager.assess_bet_risk(
            stake=kelly_stake,
            bankroll=self.bankroll,
            probability=probability,
            odds=odds
        )
        
        # Determine actual stake (risk-adjusted)
        actual_stake = risk_assessment.get('adjusted_stake', kelly_stake)
        
        # Create auto bet
        bet_id = f"auto_bet_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        auto_bet = AutoBet(
            id=bet_id,
            match_info=match_info,
            bet_type=bet_type,
            odds=odds,
            probability=probability,
            kelly_stake=kelly_stake,
            actual_stake=actual_stake,
            bookmaker=bookmaker,
            status=BetStatus.QUEUED,
            created_at=datetime.now(),
            execute_at=execute_at,
            notes=notes
        )
        
        # Check if bet meets criteria
        meets_criteria, criteria_check = self._check_bet_criteria(auto_bet, risk_assessment)
        
        if meets_criteria:
            self.bet_queue.append(auto_bet)
            self._save_data()
            
            return {
                'bet_id': bet_id,
                'queued': True,
                'kelly_stake': kelly_stake,
                'actual_stake': actual_stake,
                'risk_assessment': risk_assessment,
                'queue_position': len(self.bet_queue),
                'estimated_execution': execute_at or datetime.now()
            }
        else:
            return {
                'bet_id': bet_id,
                'queued': False,
                'rejection_reasons': criteria_check,
                'risk_assessment': risk_assessment
            }
    
    def start_auto_execution(self) -> Dict[str, Any]:
        """
        Start automatic bet execution
        
        Returns:
            Execution status
        """
        if self.mode == AutoBettingMode.MANUAL:
            return {'error': 'Cannot start auto-execution in manual mode'}
        
        if self.is_active:
            return {'error': 'Auto-execution already active'}
        
        self.is_active = True
        self.stop_execution = False
        self.session_start_time = datetime.now()
        self.daily_profit_loss = 0.0
        self.stop_loss_triggered = False
        self.profit_target_reached = False
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
        return {
            'auto_execution_started': True,
            'mode': self.mode.value,
            'session_start': self.session_start_time.isoformat(),
            'queued_bets': len(self.bet_queue)
        }
    
    def stop_auto_execution(self) -> Dict[str, Any]:
        """
        Stop automatic bet execution
        
        Returns:
            Stop status
        """
        self.is_active = False
        self.stop_execution = True
        
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        session_duration = datetime.now() - self.session_start_time
        
        return {
            'auto_execution_stopped': True,
            'session_duration_minutes': round(session_duration.total_seconds() / 60, 2),
            'session_profit_loss': self.daily_profit_loss,
            'bets_executed': len([bet for bet in self.active_bets + self.completed_bets 
                                if bet.executed_at and bet.executed_at >= self.session_start_time])
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue and system status
        
        Returns:
            Queue status information
        """
        # Update daily P&L
        self._update_daily_profit_loss()
        
        return {
            'system_status': {
                'mode': self.mode.value,
                'is_active': self.is_active,
                'bankroll': self.bankroll,
                'daily_profit_loss': round(self.daily_profit_loss, 2),
                'stop_loss_triggered': self.stop_loss_triggered,
                'profit_target_reached': self.profit_target_reached
            },
            'queue_status': {
                'queued_bets': len(self.bet_queue),
                'active_bets': len(self.active_bets),
                'completed_today': len([bet for bet in self.completed_bets 
                                      if bet.executed_at and bet.executed_at.date() == datetime.now().date()])
            },
            'limits_status': {
                'daily_profit_target': self.daily_profit_target,
                'daily_loss_limit': self.daily_loss_limit,
                'max_concurrent_bets': self.max_concurrent_bets,
                'current_concurrent': len(self.active_bets)
            },
            'next_execution': self._get_next_execution_time()
        }
    
    def execute_bet_manually(self, bet_id: str) -> Dict[str, Any]:
        """
        Manually execute a queued bet
        
        Args:
            bet_id: ID of bet to execute
            
        Returns:
            Execution result
        """
        bet = self._find_bet_in_queue(bet_id)
        if not bet:
            return {'error': f'Bet {bet_id} not found in queue'}
        
        # Check if execution is allowed
        can_execute, reason = self._can_execute_bet(bet)
        if not can_execute:
            return {'error': f'Cannot execute bet: {reason}'}
        
        # Execute the bet
        execution_result = self._execute_bet(bet)
        
        return execution_result
    
    def cancel_queued_bet(self, bet_id: str) -> Dict[str, Any]:
        """
        Cancel a queued bet
        
        Args:
            bet_id: ID of bet to cancel
            
        Returns:
            Cancellation result
        """
        bet = self._find_bet_in_queue(bet_id)
        if not bet:
            return {'error': f'Bet {bet_id} not found in queue'}
        
        bet.status = BetStatus.CANCELLED
        self.bet_queue.remove(bet)
        self.completed_bets.append(bet)
        self._save_data()
        
        return {
            'bet_cancelled': True,
            'bet_id': bet_id,
            'cancelled_at': datetime.now().isoformat()
        }
    
    def settle_bet(self, bet_id: str, result: str, actual_profit_loss: float = None) -> Dict[str, Any]:
        """
        Settle an active bet
        
        Args:
            bet_id: ID of bet to settle
            result: Result (won, lost, void)
            actual_profit_loss: Actual profit/loss (calculated if not provided)
            
        Returns:
            Settlement result
        """
        bet = self._find_active_bet(bet_id)
        if not bet:
            return {'error': f'Active bet {bet_id} not found'}
        
        # Update bet status
        if result == 'won':
            bet.status = BetStatus.WON
            bet.profit_loss = actual_profit_loss or (bet.actual_stake * (bet.odds - 1))
        elif result == 'lost':
            bet.status = BetStatus.LOST
            bet.profit_loss = actual_profit_loss or (-bet.actual_stake)
        elif result == 'void':
            bet.status = BetStatus.VOID
            bet.profit_loss = 0.0
        
        # Move to completed bets
        self.active_bets.remove(bet)
        self.completed_bets.append(bet)
        
        # Update daily P&L
        self.daily_profit_loss += bet.profit_loss
        
        # Update bankroll
        self.bankroll += bet.profit_loss
        
        # Add to portfolio tracker
        self.portfolio.add_bet(
            match_info=bet.match_info,
            bet_type=bet.bet_type,
            odds=bet.odds,
            stake=bet.actual_stake,
            prediction_confidence=bet.probability,
            bookmaker=bet.bookmaker
        )
        
        # Settle in portfolio
        portfolio_bet_id = self.portfolio.bets[-1]['id']  # Get the last added bet
        self.portfolio.settle_bet(portfolio_bet_id, result)
        
        self._save_data()
        
        # Check stop-loss and profit targets
        self._check_daily_limits()
        
        return {
            'bet_settled': True,
            'bet_id': bet_id,
            'result': result,
            'profit_loss': bet.profit_loss,
            'new_bankroll': self.bankroll,
            'daily_profit_loss': self.daily_profit_loss
        }
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get auto-betting performance summary
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance summary
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter bets by date
        period_bets = [
            bet for bet in self.completed_bets 
            if bet.executed_at and bet.executed_at >= cutoff_date
        ]
        
        if not period_bets:
            return {'error': f'No completed bets in the last {days} days'}
        
        # Calculate metrics
        total_bets = len(period_bets)
        won_bets = len([bet for bet in period_bets if bet.status == BetStatus.WON])
        total_stakes = sum(bet.actual_stake for bet in period_bets)
        total_profit_loss = sum(bet.profit_loss for bet in period_bets)
        
        win_rate = (won_bets / total_bets) * 100
        roi = (total_profit_loss / total_stakes) * 100 if total_stakes > 0 else 0
        
        # Kelly vs actual stake analysis
        kelly_stakes = sum(bet.kelly_stake for bet in period_bets)
        kelly_adherence = (total_stakes / kelly_stakes) * 100 if kelly_stakes > 0 else 0
        
        return {
            'period_days': days,
            'total_bets': total_bets,
            'won_bets': won_bets,
            'win_rate': round(win_rate, 2),
            'total_stakes': round(total_stakes, 2),
            'total_profit_loss': round(total_profit_loss, 2),
            'roi': round(roi, 2),
            'average_stake': round(total_stakes / total_bets, 2),
            'kelly_adherence': round(kelly_adherence, 2),
            'auto_execution_efficiency': self._calculate_execution_efficiency(period_bets)
        }
    
    def _execution_loop(self):
        """Main execution loop for auto-betting"""
        while not self.stop_execution and self.is_active:
            try:
                # Check daily limits
                if self._check_daily_limits():
                    break
                
                # Process queue
                self._process_bet_queue()
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                print(f"Error in execution loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _process_bet_queue(self):
        """Process the bet queue for execution"""
        if not self.bet_queue:
            return
        
        # Check if we can execute more bets
        if len(self.active_bets) >= self.max_concurrent_bets:
            return
        
        # Find bets ready for execution
        current_time = datetime.now()
        ready_bets = [
            bet for bet in self.bet_queue 
            if (bet.execute_at is None or bet.execute_at <= current_time)
        ]
        
        # Execute ready bets
        for bet in ready_bets[:self.max_concurrent_bets - len(self.active_bets)]:
            can_execute, reason = self._can_execute_bet(bet)
            if can_execute:
                self._execute_bet(bet)
    
    def _execute_bet(self, bet: AutoBet) -> Dict[str, Any]:
        """Execute a single bet"""
        try:
            # In a real implementation, this would place the bet with a bookmaker
            # For now, we'll simulate the execution
            
            bet.status = BetStatus.PLACED
            bet.executed_at = datetime.now()
            
            # Move from queue to active
            if bet in self.bet_queue:
                self.bet_queue.remove(bet)
            self.active_bets.append(bet)
            
            # Update bankroll (subtract stake)
            self.bankroll -= bet.actual_stake
            
            self._save_data()
            
            return {
                'bet_executed': True,
                'bet_id': bet.id,
                'executed_at': bet.executed_at.isoformat(),
                'stake': bet.actual_stake,
                'new_bankroll': self.bankroll
            }
            
        except Exception as e:
            bet.status = BetStatus.CANCELLED
            bet.notes += f" Execution failed: {str(e)}"
            return {'error': f'Failed to execute bet: {str(e)}'}
    
    def _check_bet_criteria(self, bet: AutoBet, risk_assessment: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if bet meets auto-betting criteria"""
        issues = []
        
        # Confidence check
        if bet.probability < self.min_confidence_threshold:
            issues.append(f"Confidence {bet.probability:.2f} below threshold {self.min_confidence_threshold}")
        
        # Stake size check
        stake_percentage = bet.actual_stake / self.bankroll
        if stake_percentage > self.max_stake_percentage:
            issues.append(f"Stake {stake_percentage:.2%} exceeds maximum {self.max_stake_percentage:.2%}")
        
        # Risk assessment check
        if not risk_assessment.get('all_checks_passed', False):
            issues.append("Failed risk management checks")
        
        # Expected value check
        expected_value = (bet.probability * bet.actual_stake * (bet.odds - 1)) - ((1 - bet.probability) * bet.actual_stake)
        if expected_value <= 0:
            issues.append("Negative expected value")
        
        return len(issues) == 0, issues
    
    def _can_execute_bet(self, bet: AutoBet) -> Tuple[bool, str]:
        """Check if bet can be executed now"""
        # Check daily limits
        if self.stop_loss_triggered:
            return False, "Daily loss limit reached"
        
        if self.profit_target_reached:
            return False, "Daily profit target reached"
        
        # Check concurrent bet limit
        if len(self.active_bets) >= self.max_concurrent_bets:
            return False, "Maximum concurrent bets reached"
        
        # Check bankroll
        if bet.actual_stake > self.bankroll:
            return False, "Insufficient bankroll"
        
        # Check execution time
        if bet.execute_at and bet.execute_at > datetime.now():
            return False, "Execution time not reached"
        
        return True, "OK"
    
    def _check_daily_limits(self) -> bool:
        """Check daily profit/loss limits"""
        self._update_daily_profit_loss()
        
        # Check loss limit
        if self.daily_loss_limit > 0 and self.daily_profit_loss <= -self.daily_loss_limit:
            self.stop_loss_triggered = True
            if self.is_active:
                self.stop_auto_execution()
            return True
        
        # Check profit target
        if self.daily_profit_target > 0 and self.daily_profit_loss >= self.daily_profit_target:
            self.profit_target_reached = True
            if self.is_active:
                self.stop_auto_execution()
            return True
        
        return False
    
    def _update_daily_profit_loss(self):
        """Update daily profit/loss from completed bets"""
        today = datetime.now().date()
        today_bets = [
            bet for bet in self.completed_bets 
            if bet.executed_at and bet.executed_at.date() == today
        ]
        
        self.daily_profit_loss = sum(bet.profit_loss for bet in today_bets)
    
    def _get_next_execution_time(self) -> Optional[str]:
        """Get next scheduled execution time"""
        if not self.bet_queue:
            return None
        
        next_bets = [bet for bet in self.bet_queue if bet.execute_at]
        if not next_bets:
            return "Immediate"
        
        next_time = min(bet.execute_at for bet in next_bets)
        return next_time.isoformat()
    
    def _find_bet_in_queue(self, bet_id: str) -> Optional[AutoBet]:
        """Find bet in queue by ID"""
        for bet in self.bet_queue:
            if bet.id == bet_id:
                return bet
        return None
    
    def _find_active_bet(self, bet_id: str) -> Optional[AutoBet]:
        """Find active bet by ID"""
        for bet in self.active_bets:
            if bet.id == bet_id:
                return bet
        return None
    
    def _calculate_execution_efficiency(self, bets: List[AutoBet]) -> Dict[str, Any]:
        """Calculate execution efficiency metrics"""
        if not bets:
            return {}
        
        # Time from queue to execution
        execution_delays = []
        for bet in bets:
            if bet.executed_at:
                delay = (bet.executed_at - bet.created_at).total_seconds() / 60  # minutes
                execution_delays.append(delay)
        
        avg_delay = np.mean(execution_delays) if execution_delays else 0
        
        return {
            'average_execution_delay_minutes': round(avg_delay, 2),
            'total_executed': len([bet for bet in bets if bet.executed_at]),
            'execution_rate': round(len(execution_delays) / len(bets) * 100, 2)
        }
    
    def _save_data(self):
        """Save auto-betting data to file"""
        data = {
            'settings': {
                'mode': self.mode.value,
                'bankroll': self.bankroll,
                'daily_profit_target': self.daily_profit_target,
                'daily_loss_limit': self.daily_loss_limit,
                'max_concurrent_bets': self.max_concurrent_bets,
                'min_confidence_threshold': self.min_confidence_threshold,
                'max_stake_percentage': self.max_stake_percentage
            },
            'session': {
                'daily_profit_loss': self.daily_profit_loss,
                'session_start_time': self.session_start_time.isoformat(),
                'stop_loss_triggered': self.stop_loss_triggered,
                'profit_target_reached': self.profit_target_reached
            },
            'bets': {
                'queue': [self._bet_to_dict(bet) for bet in self.bet_queue],
                'active': [self._bet_to_dict(bet) for bet in self.active_bets],
                'completed': [self._bet_to_dict(bet) for bet in self.completed_bets]
            }
        }
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _load_data(self):
        """Load auto-betting data from file"""
        if not os.path.exists(self.data_file):
            return
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Load settings
            settings = data.get('settings', {})
            self.mode = AutoBettingMode(settings.get('mode', 'manual'))
            self.bankroll = settings.get('bankroll', 1000.0)
            self.daily_profit_target = settings.get('daily_profit_target', 0.0)
            self.daily_loss_limit = settings.get('daily_loss_limit', 0.0)
            self.max_concurrent_bets = settings.get('max_concurrent_bets', 5)
            self.min_confidence_threshold = settings.get('min_confidence_threshold', 0.6)
            self.max_stake_percentage = settings.get('max_stake_percentage', 0.05)
            
            # Load session data
            session = data.get('session', {})
            self.daily_profit_loss = session.get('daily_profit_loss', 0.0)
            if 'session_start_time' in session:
                self.session_start_time = datetime.fromisoformat(session['session_start_time'])
            self.stop_loss_triggered = session.get('stop_loss_triggered', False)
            self.profit_target_reached = session.get('profit_target_reached', False)
            
            # Load bets
            bets_data = data.get('bets', {})
            self.bet_queue = [self._dict_to_bet(bet_dict) for bet_dict in bets_data.get('queue', [])]
            self.active_bets = [self._dict_to_bet(bet_dict) for bet_dict in bets_data.get('active', [])]
            self.completed_bets = [self._dict_to_bet(bet_dict) for bet_dict in bets_data.get('completed', [])]
            
        except Exception as e:
            print(f"Error loading auto-betting data: {e}")
    
    def _bet_to_dict(self, bet: AutoBet) -> Dict[str, Any]:
        """Convert AutoBet to dictionary"""
        return {
            'id': bet.id,
            'match_info': bet.match_info,
            'bet_type': bet.bet_type,
            'odds': bet.odds,
            'probability': bet.probability,
            'kelly_stake': bet.kelly_stake,
            'actual_stake': bet.actual_stake,
            'bookmaker': bet.bookmaker,
            'status': bet.status.value,
            'created_at': bet.created_at.isoformat(),
            'execute_at': bet.execute_at.isoformat() if bet.execute_at else None,
            'executed_at': bet.executed_at.isoformat() if bet.executed_at else None,
            'profit_loss': bet.profit_loss,
            'notes': bet.notes
        }
    
    def _dict_to_bet(self, bet_dict: Dict[str, Any]) -> AutoBet:
        """Convert dictionary to AutoBet"""
        return AutoBet(
            id=bet_dict['id'],
            match_info=bet_dict['match_info'],
            bet_type=bet_dict['bet_type'],
            odds=bet_dict['odds'],
            probability=bet_dict['probability'],
            kelly_stake=bet_dict['kelly_stake'],
            actual_stake=bet_dict['actual_stake'],
            bookmaker=bet_dict['bookmaker'],
            status=BetStatus(bet_dict['status']),
            created_at=datetime.fromisoformat(bet_dict['created_at']),
            execute_at=datetime.fromisoformat(bet_dict['execute_at']) if bet_dict['execute_at'] else None,
            executed_at=datetime.fromisoformat(bet_dict['executed_at']) if bet_dict['executed_at'] else None,
            profit_loss=bet_dict['profit_loss'],
            notes=bet_dict['notes']
        )
