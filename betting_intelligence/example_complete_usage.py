"""
Complete Usage Example for FootyForecast Betting Intelligence

Demonstrates all features: Portfolio Tracking, Kelly Calculator, Risk Management,
Arbitrage Detection, Advanced Analytics, Auto-Betting, and Advanced ML Model.
"""

from datetime import datetime, timedelta
import json

# Import all betting intelligence modules
from betting_intelligence import (
    PortfolioTracker, KellyCalculator, RiskManager, ArbitrageDetector,
    BettingAnalyzer, AdvancedAnalytics, AutoBettingSystem, AdvancedFootballModel
)

def main():
    """
    Complete demonstration of FootyForecast Betting Intelligence
    """
    print("🚀 FootyForecast Betting Intelligence - Complete Demo")
    print("=" * 60)
    
    # Initialize all components
    portfolio = PortfolioTracker("demo_portfolio.json")
    kelly = KellyCalculator()
    risk_manager = RiskManager()
    arbitrage = ArbitrageDetector()
    analytics = AdvancedAnalytics(portfolio)
    auto_betting = AutoBettingSystem(kelly, risk_manager, portfolio, "demo_auto_betting.json")
    advanced_model = AdvancedFootballModel()
    
    # Main analyzer
    analyzer = BettingAnalyzer("demo_portfolio.json")
    
    current_bankroll = 1000.0
    
    # === 1. PORTFOLIO TRACKING DEMO ===
    print("\n📊 1. PORTFOLIO TRACKING")
    print("-" * 30)
    
    # Add some demo bets
    demo_bets = [
        {
            'match_info': {'home_team': 'Manchester City', 'away_team': 'Arsenal', 'date': '2025-01-15', 'league': 'Premier League'},
            'bet_type': 'Home',
            'odds': 1.8,
            'stake': 50.0,
            'prediction_confidence': 0.75,
            'bookmaker': 'Bet365'
        },
        {
            'match_info': {'home_team': 'Barcelona', 'away_team': 'Real Madrid', 'date': '2025-01-16', 'league': 'La Liga'},
            'bet_type': 'Over2.5',
            'odds': 1.9,
            'stake': 30.0,
            'prediction_confidence': 0.68,
            'bookmaker': 'William Hill'
        }
    ]
    
    for bet in demo_bets:
        bet_id = portfolio.add_bet(**bet)
        print(f"Added bet: {bet_id}")
    
    # Settle some bets
    if len(portfolio.bets) >= 2:
        portfolio.settle_bet(portfolio.bets[0]['id'], 'won')
        portfolio.settle_bet(portfolio.bets[1]['id'], 'lost')
        print("Settled demo bets")
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"Portfolio ROI: {summary.get('roi_percentage', 0):.2f}%")
    print(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
    
    # === 2. KELLY CRITERION DEMO ===
    print("\n🎯 2. KELLY CRITERION CALCULATOR")
    print("-" * 35)
    
    # Single bet Kelly calculation
    kelly_result = kelly.calculate_stake_amount(
        bankroll=current_bankroll,
        probability=0.65,
        odds=2.1,
        fractional_kelly=0.25
    )
    
    print(f"Kelly Recommendation: {kelly_result['recommended_stake_percentage']:.1f}% of bankroll")
    print(f"Stake Amount: ${kelly_result['recommended_stake_amount']:.2f}")
    print(f"Expected Value: {kelly_result['expected_value']:.3f}")
    
    # Multiple bets analysis
    multiple_opportunities = [
        {'id': 'bet1', 'probability': 0.6, 'odds': 2.0},
        {'id': 'bet2', 'probability': 0.7, 'odds': 1.8},
        {'id': 'bet3', 'probability': 0.55, 'odds': 2.2}
    ]
    
    multi_analysis = kelly.analyze_multiple_bets(multiple_opportunities, current_bankroll)
    print(f"Multiple Bets: {multi_analysis['recommended_bets']}/{multi_analysis['total_opportunities']} recommended")
    
    # === 3. RISK MANAGEMENT DEMO ===
    print("\n🛡️ 3. RISK MANAGEMENT")
    print("-" * 25)
    
    # Assess single bet risk
    risk_assessment = risk_manager.assess_bet_risk(
        stake=50.0,
        bankroll=current_bankroll,
        probability=0.65,
        odds=2.1
    )
    
    print(f"Risk Level: {risk_assessment['risk_level']}")
    print(f"All Checks Passed: {risk_assessment['all_checks_passed']}")
    print(f"Risk-Reward Ratio: {risk_assessment['risk_reward_ratio']:.2f}")
    
    # Portfolio risk analysis
    current_bets = [
        {'stake': 50, 'probability': 0.6, 'odds': 2.0, 'match_info': {'league': 'Premier League'}},
        {'stake': 30, 'probability': 0.7, 'odds': 1.8, 'match_info': {'league': 'La Liga'}}
    ]
    
    portfolio_risk = risk_manager.analyze_portfolio_risk(current_bets, current_bankroll)
    print(f"Portfolio Risk Level: {portfolio_risk['portfolio_risk_level']}")
    print(f"Total Exposure: {portfolio_risk['total_exposure_percentage']:.1f}%")
    
    # === 4. ARBITRAGE DETECTION DEMO ===
    print("\n💰 4. ARBITRAGE DETECTION")
    print("-" * 28)
    
    # Demo bookmaker odds
    bookmaker_odds = {
        'Bet365': {'Home': 2.1, 'Draw': 3.2, 'Away': 3.8},
        'William Hill': {'Home': 2.0, 'Draw': 3.3, 'Away': 4.0},
        'Pinnacle': {'Home': 2.15, 'Draw': 3.1, 'Away': 3.9}
    }
    
    arbitrage_result = arbitrage.detect_arbitrage_opportunity(
        bookmaker_odds,
        {'home_team': 'Liverpool', 'away_team': 'Chelsea'}
    )
    
    print(f"Arbitrage Exists: {arbitrage_result['arbitrage_exists']}")
    if arbitrage_result['arbitrage_exists']:
        print(f"Profit Percentage: {arbitrage_result['profit_percentage']:.2f}%")
        print(f"Best Bookmakers: {arbitrage_result['best_bookmakers']}")
    
    # === 5. ADVANCED ANALYTICS DEMO ===
    print("\n📈 5. ADVANCED ANALYTICS")
    print("-" * 27)
    
    # Create demo betting history
    demo_history = []
    for i in range(20):
        demo_history.append({
            'timestamp': (datetime.now() - timedelta(days=i)).isoformat(),
            'status': 'won' if i % 3 == 0 else 'lost',
            'profit_loss': 25 if i % 3 == 0 else -20,
            'stake': 20,
            'bet_type': ['Home', 'Away', 'Over2.5'][i % 3],
            'match_info': {'league': ['Premier League', 'La Liga', 'Bundesliga'][i % 3]},
            'odds': 2.0 + (i % 5) * 0.2,
            'prediction_confidence': 0.6 + (i % 4) * 0.1
        })
    
    # Streak analysis
    streak_analysis = analytics.analyze_streaks(demo_history)
    print(f"Current Streak: {streak_analysis['current_streak']['type']} ({streak_analysis['current_streak']['length']})")
    print(f"Longest Win Streak: {streak_analysis['longest_win_streak']['length']}")
    
    # League profitability
    league_analysis = analytics.analyze_league_profitability(demo_history)
    if league_analysis.get('league_performance'):
        best_league = max(league_analysis['league_performance'].items(), 
                         key=lambda x: x[1]['roi'])
        print(f"Most Profitable League: {best_league[0]} ({best_league[1]['roi']:.1f}% ROI)")
    
    # === 6. AUTO-BETTING DEMO ===
    print("\n🤖 6. AUTO-BETTING SYSTEM")
    print("-" * 28)
    
    # Configure auto-betting
    config_result = auto_betting.configure_system(
        mode=auto_betting.AutoBettingMode.SEMI_AUTO,
        bankroll=current_bankroll,
        daily_profit_target=100.0,
        daily_loss_limit=50.0,
        max_concurrent_bets=3
    )
    print(f"Auto-betting configured: {config_result['mode']}")
    
    # Add bet to queue
    queue_result = auto_betting.add_bet_to_queue(
        match_info={'home_team': 'PSG', 'away_team': 'Marseille', 'league': 'Ligue 1'},
        bet_type='Home',
        odds=1.7,
        probability=0.72,
        bookmaker='Betfair'
    )
    
    print(f"Bet queued: {queue_result['queued']}")
    if queue_result['queued']:
        print(f"Recommended stake: ${queue_result['actual_stake']:.2f}")
    
    # Get queue status
    queue_status = auto_betting.get_queue_status()
    print(f"Queued bets: {queue_status['queue_status']['queued_bets']}")
    
    # === 7. ADVANCED ML MODEL DEMO ===
    print("\n🧠 7. ADVANCED ML MODEL (50+ Features)")
    print("-" * 42)
    
    # Create demo training data
    demo_training_data = []
    for i in range(100):
        match_data = {
            'home_team': f'Team_{i%10}',
            'away_team': f'Team_{(i+5)%10}',
            'league': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A'][i % 4],
            'home_recent_form': ['W', 'W', 'D', 'L', 'W'][:(i%5)+1],
            'away_recent_form': ['L', 'D', 'W', 'L', 'D'][:(i%5)+1],
            'home_season_wins': 10 + (i % 15),
            'home_season_draws': 5 + (i % 8),
            'home_season_losses': 3 + (i % 10),
            'away_season_wins': 8 + (i % 12),
            'away_season_draws': 6 + (i % 7),
            'away_season_losses': 4 + (i % 9),
            'home_goals_scored_avg': 1.5 + (i % 10) * 0.1,
            'home_goals_conceded_avg': 1.0 + (i % 8) * 0.1,
            'away_goals_scored_avg': 1.2 + (i % 9) * 0.1,
            'away_goals_conceded_avg': 1.3 + (i % 7) * 0.1,
            'home_odds': 1.5 + (i % 20) * 0.1,
            'draw_odds': 3.0 + (i % 15) * 0.1,
            'away_odds': 2.0 + (i % 25) * 0.1,
            'head_to_head': {'home_wins': i % 5, 'draws': i % 3, 'away_wins': i % 4},
            'league_stats': {'avg_goals': 2.5, 'home_win_rate': 0.45},
            'match_importance': 1 + (i % 5),
            'result': ['H', 'D', 'A'][i % 3]  # Target variable
        }
        demo_training_data.append(match_data)
    
    print("Training advanced ML model with 50+ features...")
    
    # Train models
    training_results = advanced_model.train_models(demo_training_data)
    print(f"Training completed! Best model: {training_results['best_model']}")
    print(f"Total features: {training_results['features_count']}")
    print(f"Best accuracy: {training_results['model_results'][training_results['best_model']]['test_accuracy']:.3f}")
    
    # Feature analysis
    feature_analysis = advanced_model.get_feature_analysis()
    print(f"Top 5 most important features:")
    for i, (feature, score) in enumerate(feature_analysis['top_features'][:5]):
        print(f"  {i+1}. {feature}: {score['avg_importance']:.3f}")
    
    # Make prediction
    test_match = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool',
        'league': 'Premier League',
        'home_recent_form': ['W', 'W', 'D', 'W', 'L'],
        'away_recent_form': ['W', 'L', 'W', 'W', 'D'],
        'home_season_wins': 15,
        'home_season_draws': 8,
        'home_season_losses': 5,
        'away_season_wins': 18,
        'away_season_draws': 6,
        'away_season_losses': 4,
        'home_goals_scored_avg': 2.1,
        'home_goals_conceded_avg': 1.2,
        'away_goals_scored_avg': 2.3,
        'away_goals_conceded_avg': 1.0,
        'home_odds': 2.2,
        'draw_odds': 3.1,
        'away_odds': 3.8,
        'head_to_head': {'home_wins': 3, 'draws': 2, 'away_wins': 5},
        'league_stats': {'avg_goals': 2.7, 'home_win_rate': 0.42},
        'match_importance': 4
    }
    
    prediction = advanced_model.predict_match(test_match)
    print(f"\nPrediction for Man United vs Liverpool:")
    print(f"Predicted outcome: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Probabilities: {prediction['probabilities']}")
    
    # === 8. COMPREHENSIVE ANALYSIS ===
    print("\n🎯 8. COMPREHENSIVE BETTING ANALYSIS")
    print("-" * 40)
    
    # Analyze betting opportunity
    opportunity = {
        'id': 'demo_opp_1',
        'match_info': {'home_team': 'Bayern Munich', 'away_team': 'Dortmund', 'league': 'Bundesliga'},
        'probability': 0.68,
        'odds': 1.9,
        'bet_type': 'Home'
    }
    
    comprehensive_analysis = analyzer.analyze_betting_opportunity(
        opportunity, current_bankroll, bookmaker_odds
    )
    
    print(f"Overall Recommendation: {comprehensive_analysis['recommendation']['overall']}")
    print(f"Confidence Level: {comprehensive_analysis['recommendation']['confidence_level']}")
    
    # Generate betting report
    report = analyzer.generate_betting_report(current_bankroll, [opportunity])
    print(f"\nExecutive Summary:")
    exec_summary = report.get('executive_summary', {})
    for key, value in exec_summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 FootyForecast Betting Intelligence Demo Complete!")
    print("All modules successfully demonstrated:")
    print("✅ Portfolio Tracking & ROI Analysis")
    print("✅ Kelly Criterion Optimization")
    print("✅ Advanced Risk Management")
    print("✅ Arbitrage Detection")
    print("✅ Advanced Analytics & Streaks")
    print("✅ Auto-Betting System")
    print("✅ ML Model with 50+ Features")
    print("✅ Comprehensive Analysis")

if __name__ == "__main__":
    main()
