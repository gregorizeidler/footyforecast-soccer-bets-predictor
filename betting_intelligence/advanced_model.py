"""
Advanced Machine Learning Model with Dozens of Features

Comprehensive feature engineering and advanced ML model for soccer prediction
with 50+ features covering team performance, market dynamics, and contextual factors.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFootballModel:
    """
    Advanced ML model with comprehensive feature engineering for football prediction
    """
    
    def __init__(self):
        """Initialize the advanced model"""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                num_leaves=31,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=42
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def engineer_features(self, match_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Engineer comprehensive features from match data
        
        Args:
            match_data: List of match data dictionaries
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        for match in match_data:
            features = {}
            
            # Basic match info
            home_team = match.get('home_team', 'Unknown')
            away_team = match.get('away_team', 'Unknown')
            league = match.get('league', 'Unknown')
            
            # === TEAM PERFORMANCE FEATURES (20 features) ===
            
            # Home team recent form (last 5 matches)
            home_recent = match.get('home_recent_form', [])
            features['home_wins_last_5'] = sum(1 for result in home_recent[-5:] if result == 'W')
            features['home_draws_last_5'] = sum(1 for result in home_recent[-5:] if result == 'D')
            features['home_losses_last_5'] = sum(1 for result in home_recent[-5:] if result == 'L')
            features['home_form_points'] = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in home_recent[-5:])
            
            # Away team recent form (last 5 matches)
            away_recent = match.get('away_recent_form', [])
            features['away_wins_last_5'] = sum(1 for result in away_recent[-5:] if result == 'W')
            features['away_draws_last_5'] = sum(1 for result in away_recent[-5:] if result == 'D')
            features['away_losses_last_5'] = sum(1 for result in away_recent[-5:] if result == 'L')
            features['away_form_points'] = sum(3 if r == 'W' else 1 if r == 'D' else 0 for r in away_recent[-5:])
            
            # Season statistics
            features['home_season_wins'] = match.get('home_season_wins', 0)
            features['home_season_draws'] = match.get('home_season_draws', 0)
            features['home_season_losses'] = match.get('home_season_losses', 0)
            features['away_season_wins'] = match.get('away_season_wins', 0)
            features['away_season_draws'] = match.get('away_season_draws', 0)
            features['away_season_losses'] = match.get('away_season_losses', 0)
            
            # Home/Away specific performance
            features['home_home_wins'] = match.get('home_home_wins', 0)
            features['home_home_losses'] = match.get('home_home_losses', 0)
            features['away_away_wins'] = match.get('away_away_wins', 0)
            features['away_away_losses'] = match.get('away_away_losses', 0)
            
            # Form difference
            features['form_difference'] = features['home_form_points'] - features['away_form_points']
            
            # === GOAL STATISTICS FEATURES (15 features) ===
            
            # Goals scored and conceded
            features['home_goals_scored_avg'] = match.get('home_goals_scored_avg', 0)
            features['home_goals_conceded_avg'] = match.get('home_goals_conceded_avg', 0)
            features['away_goals_scored_avg'] = match.get('away_goals_scored_avg', 0)
            features['away_goals_conceded_avg'] = match.get('away_goals_conceded_avg', 0)
            
            # Goal difference
            features['home_goal_difference'] = features['home_goals_scored_avg'] - features['home_goals_conceded_avg']
            features['away_goal_difference'] = features['away_goals_scored_avg'] - features['away_goals_conceded_avg']
            features['goal_difference_gap'] = features['home_goal_difference'] - features['away_goal_difference']
            
            # Attacking and defensive strength
            features['home_attack_strength'] = match.get('home_attack_strength', 1.0)
            features['home_defense_strength'] = match.get('home_defense_strength', 1.0)
            features['away_attack_strength'] = match.get('away_attack_strength', 1.0)
            features['away_defense_strength'] = match.get('away_defense_strength', 1.0)
            
            # Expected goals
            features['home_xg_avg'] = match.get('home_xg_avg', 0)
            features['away_xg_avg'] = match.get('away_xg_avg', 0)
            features['home_xga_avg'] = match.get('home_xga_avg', 0)  # Expected goals against
            features['away_xga_avg'] = match.get('away_xga_avg', 0)
            
            # === MARKET & ODDS FEATURES (10 features) ===
            
            # Bookmaker odds
            features['home_odds'] = match.get('home_odds', 2.0)
            features['draw_odds'] = match.get('draw_odds', 3.0)
            features['away_odds'] = match.get('away_odds', 3.5)
            
            # Implied probabilities
            total_implied = (1/features['home_odds']) + (1/features['draw_odds']) + (1/features['away_odds'])
            features['home_implied_prob'] = (1/features['home_odds']) / total_implied
            features['draw_implied_prob'] = (1/features['draw_odds']) / total_implied
            features['away_implied_prob'] = (1/features['away_odds']) / total_implied
            
            # Market efficiency indicators
            features['overround'] = total_implied - 1
            features['favorite_odds'] = min(features['home_odds'], features['away_odds'])
            features['underdog_odds'] = max(features['home_odds'], features['away_odds'])
            features['odds_ratio'] = features['underdog_odds'] / features['favorite_odds']
            
            # === HEAD-TO-HEAD FEATURES (8 features) ===
            
            h2h_data = match.get('head_to_head', {})
            features['h2h_home_wins'] = h2h_data.get('home_wins', 0)
            features['h2h_draws'] = h2h_data.get('draws', 0)
            features['h2h_away_wins'] = h2h_data.get('away_wins', 0)
            features['h2h_total_matches'] = features['h2h_home_wins'] + features['h2h_draws'] + features['h2h_away_wins']
            
            if features['h2h_total_matches'] > 0:
                features['h2h_home_win_rate'] = features['h2h_home_wins'] / features['h2h_total_matches']
                features['h2h_away_win_rate'] = features['h2h_away_wins'] / features['h2h_total_matches']
            else:
                features['h2h_home_win_rate'] = 0.33
                features['h2h_away_win_rate'] = 0.33
            
            features['h2h_avg_goals'] = h2h_data.get('avg_goals', 2.5)
            features['h2h_over_2_5_rate'] = h2h_data.get('over_2_5_rate', 0.5)
            
            # === CONTEXTUAL FEATURES (12 features) ===
            
            # League strength and characteristics
            league_stats = match.get('league_stats', {})
            features['league_avg_goals'] = league_stats.get('avg_goals', 2.5)
            features['league_home_win_rate'] = league_stats.get('home_win_rate', 0.45)
            features['league_draw_rate'] = league_stats.get('draw_rate', 0.25)
            features['league_away_win_rate'] = league_stats.get('away_win_rate', 0.30)
            
            # Match importance
            features['match_importance'] = match.get('match_importance', 1.0)  # 1-5 scale
            features['is_derby'] = 1 if match.get('is_derby', False) else 0
            features['is_cup_match'] = 1 if match.get('is_cup_match', False) else 0
            
            # Time factors
            features['days_since_last_match_home'] = match.get('days_since_last_match_home', 7)
            features['days_since_last_match_away'] = match.get('days_since_last_match_away', 7)
            features['rest_advantage'] = features['days_since_last_match_home'] - features['days_since_last_match_away']
            
            # Season timing
            features['matches_played_home'] = match.get('matches_played_home', 10)
            features['matches_played_away'] = match.get('matches_played_away', 10)
            
            # === MOMENTUM & STREAKS FEATURES (8 features) ===
            
            # Current streaks
            features['home_current_streak'] = match.get('home_current_streak', 0)  # Positive for wins, negative for losses
            features['away_current_streak'] = match.get('away_current_streak', 0)
            features['streak_difference'] = features['home_current_streak'] - features['away_current_streak']
            
            # Momentum indicators
            features['home_momentum'] = match.get('home_momentum', 0)  # Points in last 3 matches
            features['away_momentum'] = match.get('away_momentum', 0)
            features['momentum_difference'] = features['home_momentum'] - features['away_momentum']
            
            # Performance trends
            features['home_trend'] = match.get('home_trend', 0)  # Improving/declining performance
            features['away_trend'] = match.get('away_trend', 0)
            
            # === ADVANCED METRICS (10 features) ===
            
            # Possession and style
            features['home_possession_avg'] = match.get('home_possession_avg', 50)
            features['away_possession_avg'] = match.get('away_possession_avg', 50)
            features['possession_difference'] = features['home_possession_avg'] - features['away_possession_avg']
            
            # Shots and accuracy
            features['home_shots_per_game'] = match.get('home_shots_per_game', 10)
            features['away_shots_per_game'] = match.get('away_shots_per_game', 10)
            features['home_shots_on_target_rate'] = match.get('home_shots_on_target_rate', 0.33)
            features['away_shots_on_target_rate'] = match.get('away_shots_on_target_rate', 0.33)
            
            # Defensive metrics
            features['home_clean_sheet_rate'] = match.get('home_clean_sheet_rate', 0.3)
            features['away_clean_sheet_rate'] = match.get('away_clean_sheet_rate', 0.3)
            features['home_cards_per_game'] = match.get('home_cards_per_game', 2.0)
            
            # === DERIVED FEATURES (5 features) ===
            
            # Composite strength indicators
            features['home_overall_strength'] = (
                features['home_form_points'] * 0.3 +
                features['home_goal_difference'] * 0.3 +
                features['home_attack_strength'] * 0.2 +
                (1/features['home_defense_strength']) * 0.2
            )
            
            features['away_overall_strength'] = (
                features['away_form_points'] * 0.3 +
                features['away_goal_difference'] * 0.3 +
                features['away_attack_strength'] * 0.2 +
                (1/features['away_defense_strength']) * 0.2
            )
            
            features['strength_difference'] = features['home_overall_strength'] - features['away_overall_strength']
            
            # Market vs performance discrepancy
            features['value_indicator'] = features['home_implied_prob'] - (features['home_overall_strength'] / 10)
            
            # Total goals expectation
            features['total_goals_expectation'] = (
                features['home_goals_scored_avg'] + features['away_goals_scored_avg'] +
                features['home_goals_conceded_avg'] + features['away_goals_conceded_avg']
            ) / 2
            
            # Add categorical features
            features['home_team'] = home_team
            features['away_team'] = away_team
            features['league'] = league
            
            # Target variable (if available)
            if 'result' in match:
                features['result'] = match['result']  # 'H', 'D', 'A'
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Separate features and target
        if 'result' in df.columns:
            target = df['result']
            features_df = df.drop('result', axis=1)
        else:
            target = None
            features_df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['home_team', 'away_team', 'league']
        
        for col in categorical_cols:
            if col in features_df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    features_df[col] = self.encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(features_df[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    new_values = unique_values - known_values
                    
                    if new_values:
                        # Add new categories to encoder
                        all_values = list(known_values) + list(new_values)
                        self.encoders[col].classes_ = np.array(all_values)
                    
                    features_df[col] = self.encoders[col].transform(features_df[col].astype(str))
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df, target
    
    def train_models(self, match_data: List[Dict[str, Any]], test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all models on the provided data
        
        Args:
            match_data: List of match data with results
            test_size: Proportion of data for testing
            
        Returns:
            Training results and performance metrics
        """
        # Engineer features
        df = self.engineer_features(match_data)
        
        if 'result' not in df.columns:
            raise ValueError("Training data must include 'result' column")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train each model
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Scale features for models that need it
            if model_name in ['neural_network', 'svm', 'logistic_regression']:
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            train_proba = model.predict_proba(X_train_scaled)
            test_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            train_accuracy = (train_pred == y_train).mean()
            test_accuracy = (test_pred == y_test).mean()
            
            # Cross-validation score
            if model_name in ['neural_network', 'svm', 'logistic_regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_names = X.columns.tolist()
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )
            
            results[model_name] = {
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'cv_mean': round(cv_scores.mean(), 4),
                'cv_std': round(cv_scores.std(), 4),
                'classification_report': classification_report(y_test, test_pred, output_dict=True)
            }
        
        self.is_trained = True
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        
        return {
            'training_completed': True,
            'total_samples': len(df),
            'features_count': len(X.columns),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'best_model': best_model_name,
            'model_results': results,
            'feature_importance': self.feature_importance,
            'feature_names': X.columns.tolist()
        }
    
    def predict_match(self, match_data: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """
        Predict match outcome using trained models
        
        Args:
            match_data: Single match data
            model_name: Specific model to use (None for ensemble)
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Engineer features for single match
        df = self.engineer_features([match_data])
        X, _ = self.prepare_data(df)
        
        if model_name:
            # Use specific model
            model = self.models[model_name]
            
            # Scale if needed
            if model_name in ['neural_network', 'svm', 'logistic_regression']:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X
            
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Map probabilities to outcomes
            classes = model.classes_
            prob_dict = dict(zip(classes, probabilities))
            
            return {
                'model_used': model_name,
                'prediction': prediction,
                'probabilities': prob_dict,
                'confidence': max(probabilities)
            }
        
        else:
            # Ensemble prediction
            predictions = {}
            all_probabilities = {}
            
            for name, model in self.models.items():
                # Scale if needed
                if name in ['neural_network', 'svm', 'logistic_regression']:
                    X_scaled = self.scalers[name].transform(X)
                else:
                    X_scaled = X
                
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                predictions[name] = pred
                all_probabilities[name] = dict(zip(model.classes_, proba))
            
            # Ensemble voting
            from collections import Counter
            vote_counts = Counter(predictions.values())
            ensemble_prediction = vote_counts.most_common(1)[0][0]
            
            # Average probabilities
            all_classes = set()
            for probs in all_probabilities.values():
                all_classes.update(probs.keys())
            
            ensemble_probabilities = {}
            for class_name in all_classes:
                class_probs = [probs.get(class_name, 0) for probs in all_probabilities.values()]
                ensemble_probabilities[class_name] = np.mean(class_probs)
            
            return {
                'model_used': 'ensemble',
                'prediction': ensemble_prediction,
                'probabilities': ensemble_probabilities,
                'confidence': max(ensemble_probabilities.values()),
                'individual_predictions': predictions,
                'individual_probabilities': all_probabilities,
                'vote_distribution': dict(vote_counts)
            }
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive feature analysis
        
        Returns:
            Feature analysis results
        """
        if not self.feature_importance:
            return {'error': 'No feature importance data available. Train models first.'}
        
        # Aggregate feature importance across models
        all_features = set()
        for importance_list in self.feature_importance.values():
            all_features.update([feat[0] for feat in importance_list])
        
        feature_scores = {}
        for feature in all_features:
            scores = []
            for model_name, importance_list in self.feature_importance.items():
                importance_dict = dict(importance_list)
                if feature in importance_dict:
                    scores.append(importance_dict[feature])
            
            if scores:
                feature_scores[feature] = {
                    'avg_importance': np.mean(scores),
                    'max_importance': max(scores),
                    'min_importance': min(scores),
                    'std_importance': np.std(scores),
                    'models_count': len(scores)
                }
        
        # Sort by average importance
        sorted_features = sorted(feature_scores.items(), 
                               key=lambda x: x[1]['avg_importance'], 
                               reverse=True)
        
        # Categorize features
        feature_categories = {
            'team_performance': [],
            'goal_statistics': [],
            'market_odds': [],
            'head_to_head': [],
            'contextual': [],
            'momentum': [],
            'advanced_metrics': [],
            'derived': []
        }
        
        for feature_name, _ in sorted_features:
            if any(keyword in feature_name.lower() for keyword in ['wins', 'losses', 'draws', 'form', 'season']):
                feature_categories['team_performance'].append(feature_name)
            elif any(keyword in feature_name.lower() for keyword in ['goals', 'xg', 'attack', 'defense']):
                feature_categories['goal_statistics'].append(feature_name)
            elif any(keyword in feature_name.lower() for keyword in ['odds', 'implied', 'overround', 'favorite']):
                feature_categories['market_odds'].append(feature_name)
            elif 'h2h' in feature_name.lower():
                feature_categories['head_to_head'].append(feature_name)
            elif any(keyword in feature_name.lower() for keyword in ['league', 'derby', 'cup', 'importance', 'days']):
                feature_categories['contextual'].append(feature_name)
            elif any(keyword in feature_name.lower() for keyword in ['streak', 'momentum', 'trend']):
                feature_categories['momentum'].append(feature_name)
            elif any(keyword in feature_name.lower() for keyword in ['possession', 'shots', 'clean', 'cards']):
                feature_categories['advanced_metrics'].append(feature_name)
            else:
                feature_categories['derived'].append(feature_name)
        
        return {
            'total_features': len(all_features),
            'top_features': sorted_features[:20],
            'feature_categories': feature_categories,
            'feature_scores': feature_scores,
            'model_specific_importance': self.feature_importance
        }
    
    def hyperparameter_tuning(self, match_data: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            match_data: Training data
            model_name: Name of model to tune
            
        Returns:
            Tuning results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Engineer features
        df = self.engineer_features(match_data)
        X, y = self.prepare_data(df)
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [6, 8, 10]
            }
        }
        
        if model_name not in param_grids:
            return {'error': f'Hyperparameter tuning not implemented for {model_name}'}
        
        # Perform grid search
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'model_name': model_name,
            'best_parameters': grid_search.best_params_,
            'best_score': round(grid_search.best_score_, 4),
            'tuning_completed': True
        }
