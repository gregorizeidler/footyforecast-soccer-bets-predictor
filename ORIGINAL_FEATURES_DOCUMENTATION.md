# FootyForecast - Original Features Documentation

This document contains the original feature descriptions and visual documentation from the README.

## Stunning Graphical Interface

The user interface is pretty simple: Every action can be done via a menu-bar on the top of the application. There are 5 available menus:

* Application: Create/Load/Delete Leagues
* Analysis: Data Analysis & Feature Importance
* Model: Train/Evaluate Models & Predict Matches
* Theme: Select a Theme for the Application Window
* Help: Additional Resources to Read about Machine Learning Topics

Also, 4 custom themes have been added and can be selected via "Theme" menu. The themes are:

1. Breeze-Light
1. Breeze-Dark
1. Forest-Light
1. Forest-Dark

![gui](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/create_league.png)

![gui](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/loaded_league.png)

## League Statistics

For each league, the application computes several statistics (features) about the teams, including their form, the performance of the last N matches, etc. The stats are computed for both the home team and the away team. More specifically:

1. **Home Wins (HW)**: Last N wins of the home team in its home
2. **Home Losses (HL)**: Last N losses of the home team in its home
3. **Home Goal Forward (HGF)**: Sum of goals that the home team scored in the last N matches in its home
4. **Home Goal Against (HGA)**: Sum of goals that the away teams scored in the last N matches.
5. **Home G-Goal Difference Wins (HGD-W)** Last N wins of the home team with G difference in the final score in its home (${HG - AG ≥ 2}$)
6. **Home G-Goal Difference Losses (HGD-L)** Last N losses of the home team with G difference in the final score in its home (${HG - AG ≥ 2}$)
7. **Home Win Rate (HW%)** Total win rate of the home team from the start of the league in its home
8. **Home Loss Rate (HL%)** Total loss rate of the home team from the start of the league in its home
9. **Away Wins (AW)**: Last N wins of the away team away its home
10. **Away Losses (AL)**: Last N losses of the away team away its home
11. **Away Goal Forward (AGF)**: Sum of goals that the away team scored in the last N matches away its home
12. **Away Goal Against (AGA)**: Sum of goals that the home teams scored in the last N matches.
13. **Away G-Goal Difference Wins (AGD-W)** Last N wins of the away team with G difference in the final score away its home(${HG - AG ≥ 2}$)
14. **Away G-Goal Difference Losses (AGD-L)** Last N losses of the away team with G difference in the final score away its home (${HG - AG ≥ 2}$)
15. **Away Win Rate (AW%)** Total win rate from the start of the league away its home
16. **Away Loss Rate (AL%)** Total loss rate from the start of the league away its home

Each column can be added or removed from a league during the creating phase.

## Leagues

FootyForecast provides 11 main soccer leagues and 2 extras, which are downloaded by https://www.football-data.co.uk/. More specifically, these leagues are:

* 'Argentina': [PrimeraDivision]
* 'Belgium': [JupilerLeague]
* 'Brazil': [BrazilSerieA]
* 'China': [ChinaSuperLeague]
* 'Denmark': [SuperLiga]
* 'England': [PremierLeague, Championship, League1, League2]
* 'Finland': [VeikkausLiiga]
* 'France': [Ligue1, Ligue2]
* 'Germany': [Bundesliga1, Bundesliga2]
* 'Greece': [SuperLeague]
* 'Ireland': [IrelandPremierDivision]
* 'Italy': [SerieA, SerieB]
* 'Japan': [J1]
* 'Mexico': [LigaMX]
* 'Netherlands': [Eredivisie]
* 'Norway': [Eliteserien]
* 'Poland': [Ekstraklasa]
* 'Portugal': [Liga1]
* 'Romania': [RomaniaLiga1]
* 'Russia': [RussiaPremierLeague]
* 'Scotland': [Premiership]
* 'Spain': [LaLiga, SegundaDivision]
* 'Sweden': [Allsvenskan]
* 'Switzerland': [SwitzerlandSuperLeague]
* 'USA': [MLS]
* 'Turkey': [SuperLig]

You can add additional leagues by modifying the `database/leagues.csv` configuration file. In order to add a new league, you need to specify:

1. Country (The country of the league, e.g. Russia)
2. League Name (The name of the league e.g. Premier League)
3. League ID: You can create multiple leagues, but with different ID.
4. The statistical odds that will be used to train the models.

## Feature Correlation Analysis

This is particularly useful when analyzing the quality of the training data. FootyForecast provides a heatmap for the correlation matrix between the features, which shows the correlations between 2 features (columns). The correlation is described by an arithmetic value ${r ∈ [-1.0, 1.0]}$. The closer $r$ is to zero, the weaker the correlation is between 2 columns. The closer to 1.0 or -1.0, the stronger the correlation will be. Ideally, a feature is good if its correlation with the rest of the features is close to zero ($r=0$).

![correlation heatmap analysis](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/correlations.png)

## Feature Importance Analysis

FootyForecast also comes with a built-in module for "**interpretability**". In case you are wondering which stats are the most important, there are 3 methods included:

1. Variance Analysis (https://corporatefinanceinstitute.com/resources/knowledge/accounting/variance-analysis/)
2. Recursive Feature Elimination (https://bookdown.org/max/FES/recursive-feature-elimination.html)
3. Random Forest importance scores

![feature-importance-analysis](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/importance.png)

## Class (Target) Distribution Analysis

It is noticed that the training dataset of several leagues contains imbalanced classes, which means that the number of matches that ended in a win for the home team is a lot larger than the number of the matches that ended in a win for the away team. This often leads models to overestimate their prediction probabilities and tend to have a bias towards the home team. FootyForecast provides a plot to detect such leagues, using the **Target Distribution Plot**, as well as several tools to deal with that, including:

1. Noise Injection (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2771718/)
2. Output Probability Calibration (https://davidrosenberg.github.io/ttml2021/calibration/2.calibration.pdf)
3. Resampling techniques (SMOTE, SMOTE-NN, SVM-SMOTE, NearMiss, Random Resampling)

![class distribution](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/targets.png)

## Training Deep Neural Networks

A detailed description of neural networks can be found in the link below:
https://www.investopedia.com/terms/n/neuralnetwork.asp

![deep neural networks](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/neuralnetwork.png)

## Machine Learning Models

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Random Forest
6. XG-Boost
7. Support Vector Machine (SVM)
8. Deep Neural Networks

## Training Random Forests

A detailed description of random forests can be found in the link below:
https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

![random forests](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/randomforest.png)

## The Ensemble Model

This type combines the predictions of several machine learning models. Typically, a well tuned Random Forest could generate similar predictions with a Neural Network or any other ML model. However, there are some cases where 2 models could output different output probabilities (e.g. Random Forest might give higher probability that an outcome is Home). In that case, the ensemble model (Voting Model) can be used, which averages the output probabilities of several models and decides on the predicted outcome. The idea is that each model makes unique predictions, so their predictions are combined to form a stronger model.

## Evaluating Models

Before using a trained model, it is wise to first evaluate the model on unseen matches. This should reveal the quality of the model training, as well as its output probabilities. You can compare the probabilities of random forest with the neural network's probabilities and choose the most confident and well-trained model. Additionally, you can request an analytical report of the accuracy of the classifiers for specific odd intervals (e.g. the accuracy between 1.0 and 1.3, 1.3, and 1.6, etc., for the home or away team).

![model evaluation](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/evaluate.png)

## Outcome Predictions

In order to request a prediction for a match, You need to select the home/away team, as well as the book odds. You should use both models to make a prediction. If both models agree, then the prediction should probably be good. If the models disagree, then it's best to avoid betting on that match. The outcome prediction includes:

1. Home, Draw or Away
2. Under (2.5) or Over (2.5)

![match predictions](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/predictions.png)

## Fixture Parsing

An alternative way to predict multiple matches at once is to use the "**Fixture Parsing**" option. You may now automatically parse the fixtures using your browser. Once the fixture window pops-up, select your **browser and the fixture date** and the application will automatically download the page & parse the upcoming fixtures of the specified data. This is a new feature, so please report any bugs in the issues page.

![fixture parsing & upcoming match prediction](https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor/blob/main/screenshots/fixtures.png)

---

## How to Recreate Screenshots

To recreate the visual documentation:

1. **Launch the application:**
   ```bash
   python main.py
   ```

2. **Take screenshots of each feature:**
   - **GUI Interface:** Main window, league creation dialog
   - **Analysis Charts:** Correlation heatmaps, feature importance plots
   - **Model Training:** Neural network configuration, random forest settings
   - **Evaluation:** Model performance metrics and comparisons
   - **Predictions:** Match prediction interface and results
   - **Fixture Parsing:** Browser-based fixture import

3. **Save screenshots with these names:**
   - `create_league.png` - League creation interface
   - `loaded_league.png` - Main application with loaded league
   - `correlations.png` - Feature correlation heatmap
   - `importance.png` - Feature importance analysis
   - `targets.png` - Class distribution analysis
   - `neuralnetwork.png` - Neural network training interface
   - `randomforest.png` - Random forest configuration
   - `evaluate.png` - Model evaluation results
   - `predictions.png` - Match prediction interface
   - `fixtures.png` - Fixture parsing functionality
   - `leagues.png` - League selection interface
   - `stats.png` - Statistical analysis views
   - `parameters.png` - Model parameter configuration
   - `validation_analysis.png` - Validation results

4. **Place all screenshots in the `/screenshots/` directory**

This will restore the full visual documentation of FootyForecast's capabilities.
