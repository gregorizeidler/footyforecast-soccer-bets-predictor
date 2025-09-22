# FootyForecast - Soccer Bets Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey.svg)](#supported-platforms)

> 🚀 **AI-Powered Soccer Prediction System** - Analyze team performance and predict match outcomes using advanced Machine Learning algorithms!

## ⚡ Quick Start

### 🎯 **Get Started in 3 Steps:**

1. **📥 Clone & Install**
   ```bash
   git clone https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor.git
   cd FootyForecast-Soccer-Bets-Predictor
   pip install -r requirements.txt
   ```

2. **🚀 Launch Application**
   ```bash
   python main.py
   ```

3. **⚽ Start Predicting**
   - Create/Load a league → Analyze data → Train models → Make predictions!

### 🎮 **What You Can Do:**
- 📊 **Analyze** 25+ soccer leagues worldwide
- 🤖 **Train** 8 different ML algorithms (Neural Networks, Random Forest, XGBoost, etc.)
- 🔮 **Predict** match outcomes (Home/Draw/Away + Over/Under 2.5 goals)
- 📈 **Visualize** team statistics and model performance
- 🎨 **Customize** with 4 beautiful themes

---

## 🎯 Use Cases & Examples

### 🏆 **Perfect For:**

#### 📊 **Data Scientists & ML Enthusiasts**
- **Experiment** with different algorithms on real soccer data
- **Compare** model performance using cross-validation
- **Analyze** feature importance and correlations
- **Tune** hyperparameters automatically with Optuna

#### ⚽ **Soccer Analysts**
- **Track** team performance trends over time
- **Identify** statistical patterns in match outcomes
- **Evaluate** home vs away team advantages
- **Study** goal-scoring patterns (Over/Under analysis)

#### 🎓 **Students & Researchers**
- **Learn** practical machine learning applications
- **Study** sports analytics methodologies
- **Research** predictive modeling techniques
- **Publish** academic papers (citation included!)

#### 💰 **Betting Enthusiasts** *(Bet Responsibly)*
- **Analyze** odds vs model predictions
- **Identify** value betting opportunities
- **Track** model accuracy over time
- **Make** data-driven decisions

### 🌟 **Real-World Examples:**

```
🔍 Example 1: Premier League Analysis
→ Load Premier League data
→ Train Random Forest model (85% accuracy)
→ Predict: Manchester City vs Arsenal
→ Result: Home Win (78% confidence)

📈 Example 2: Feature Analysis
→ Discover: "Home Goals Forward" most important feature
→ Teams scoring 2+ goals at home win 73% of matches
→ Use this insight for better predictions

🎯 Example 3: Multi-League Comparison
→ Compare Bundesliga vs Serie A patterns
→ Bundesliga: More goals per match (2.8 avg)
→ Serie A: More defensive games (2.3 avg)
```

---

## 🚀 System Workflow

```mermaid
flowchart TD
    A[🌐 Start FootyForecast] --> B{📊 League Available?}
    
    B -->|No| C[🏆 Create/Load League]
    B -->|Yes| D[📈 Data Analysis]
    
    C --> C1[🌍 Download League Data]
    C1 --> C2[⚽ Process Team Statistics]
    C2 --> C3[💾 Save League Configuration]
    C3 --> D
    
    D --> E[🔍 Choose Analysis Type]
    
    E --> F[📊 Correlation Analysis]
    E --> G[🎯 Target Distribution]
    E --> H[📈 Feature Importance]
    E --> I[📉 Variance Analysis]
    
    F --> J[🤖 Machine Learning Pipeline]
    G --> J
    H --> J
    I --> J
    
    J --> K[🧠 Select ML Algorithm]
    
    K --> L[🌳 Decision Tree]
    K --> M[🚀 XGBoost]
    K --> N[🔗 Neural Network]
    K --> O[🌲 Random Forest]
    K --> P[📐 SVM]
    K --> Q[📊 Logistic Regression]
    K --> R[🎲 Naive Bayes]
    K --> S[👥 Ensemble Model]
    
    L --> T[⚙️ Model Training]
    M --> T
    N --> T
    O --> T
    P --> T
    Q --> T
    R --> T
    S --> T
    
    T --> U[✅ Cross Validation]
    U --> V[📊 Model Evaluation]
    V --> W{🎯 Good Performance?}
    
    W -->|No| X[🔧 Hyperparameter Tuning]
    X --> T
    
    W -->|Yes| Y[🔮 Make Predictions]
    
    Y --> Z1[⚽ Single Match Prediction]
    Y --> Z2[📅 Fixture Predictions]
    
    Z1 --> AA[🏠 Home Win / ❌ Draw / 🛣️ Away Win]
    Z1 --> BB[⬆️ Over 2.5 / ⬇️ Under 2.5 Goals]
    
    Z2 --> CC[📋 Batch Predictions]
    CC --> DD[💰 Betting Recommendations]
    
    style A fill:#FF6B6B,stroke:#FF4757,stroke-width:3px,color:#fff
    style J fill:#4ECDC4,stroke:#26D0CE,stroke-width:3px,color:#fff
    style T fill:#45B7D1,stroke:#3742FA,stroke-width:3px,color:#fff
    style Y fill:#96CEB4,stroke:#00B894,stroke-width:3px,color:#fff
    style DD fill:#FECA57,stroke:#FF9F43,stroke-width:3px,color:#fff
    
    style L fill:#FF7675,stroke:#D63031,stroke-width:2px,color:#fff
    style M fill:#74B9FF,stroke:#0984E3,stroke-width:2px,color:#fff
    style N fill:#A29BFE,stroke:#6C5CE7,stroke-width:2px,color:#fff
    style O fill:#55A3FF,stroke:#2D3436,stroke-width:2px,color:#fff
    style P fill:#FD79A8,stroke:#E84393,stroke-width:2px,color:#fff
    style Q fill:#FDCB6E,stroke:#E17055,stroke-width:2px,color:#fff
    style R fill:#6C5CE7,stroke:#A29BFE,stroke-width:2px,color:#fff
    style S fill:#00B894,stroke:#00CEC9,stroke-width:2px,color:#fff
```

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "🖥️ User Interface Layer"
        GUI[🎨 Tkinter GUI]
        MENU[📋 Menu System]
        DIALOG[💬 Dialog Windows]
    end
    
    subgraph "🧠 Core Logic Layer"
        ANALYSIS[📊 Analysis Engine]
        ML[🤖 ML Pipeline]
        PREDICT[🔮 Prediction Engine]
    end
    
    subgraph "💾 Data Layer"
        DB[🗄️ League Repository]
        MODEL[🧪 Model Repository]
        CONFIG[⚙️ Configuration]
    end
    
    subgraph "🌐 External Data Sources"
        FOOTBALL[⚽ Football-Data.co.uk]
        FOOTY[📈 FootyStats.org]
    end
    
    subgraph "🔧 Processing Modules"
        PREPROCESS[🔄 Data Preprocessing]
        STATS[📊 Statistics Calculator]
        FEATURES[🎯 Feature Engineering]
    end
    
    subgraph "🤖 ML Algorithms"
        DT[🌳 Decision Tree]
        RF[🌲 Random Forest]
        NN[🔗 Neural Network]
        XGB[🚀 XGBoost]
        SVM[📐 SVM]
        NB[🎲 Naive Bayes]
        LR[📊 Logistic Regression]
        ENS[👥 Ensemble]
    end
    
    GUI --> ANALYSIS
    GUI --> ML
    GUI --> PREDICT
    
    ANALYSIS --> PREPROCESS
    ML --> PREPROCESS
    PREDICT --> MODEL
    
    PREPROCESS --> STATS
    STATS --> FEATURES
    
    ML --> DT
    ML --> RF
    ML --> NN
    ML --> XGB
    ML --> SVM
    ML --> NB
    ML --> LR
    ML --> ENS
    
    DB --> FOOTBALL
    DB --> FOOTY
    
    FEATURES --> DB
    MODEL --> CONFIG
    
    style GUI fill:#FF6B6B,stroke:#FF4757,stroke-width:3px,color:#fff
    style ML fill:#4ECDC4,stroke:#26D0CE,stroke-width:3px,color:#fff
    style PREDICT fill:#96CEB4,stroke:#00B894,stroke-width:3px,color:#fff
    style DB fill:#45B7D1,stroke:#3742FA,stroke-width:3px,color:#fff
    style FOOTBALL fill:#FECA57,stroke:#FF9F43,stroke-width:3px,color:#fff
    style FOOTY fill:#FECA57,stroke:#FF9F43,stroke-width:3px,color:#fff
```

## 📖 What is FootyForecast?

FootyForecast represents a comprehensive **open-source solution** for soccer match prediction powered by artificial intelligence. This innovative platform merges "Footy" (soccer terminology) with "Forecast" (predictive analytics) to deliver data-driven insights.

**🎯 Primary Capabilities:**
- **Performance Analytics:** Deep dive into team dynamics using cutting-edge ML algorithms and interactive visualizations
- **Statistical Computing:** Generate detailed metrics from historical match data across multiple leagues
- **Outcome Prediction:** Leverage ensemble learning approaches for accurate match result forecasting

**🤖 AI Engine Portfolio:**
- Advanced Neural Networks, Gradient Boosting (XGBoost), Random Forest Ensembles
- Support Vector Machines, K-Nearest Neighbors, Probabilistic Classifiers
- Decision Trees, Linear Models, and Hybrid Ensemble Architectures

**🔧 Intelligent Processing Pipeline:**
- **Data Preprocessing:** Automated normalization, feature scaling, and imbalanced dataset handling
- **Model Validation:** Robust cross-validation frameworks with holdout testing protocols
- **Optimization Engine:** Automated hyperparameter search using advanced optimization algorithms

**📊 Data Integration:**
- Comprehensive league coverage via [football-data.co.uk](https://www.football-data.co.uk/) API integration
- Real-time fixture parsing through [FootyStats.org](https://footystats.org/) connectivity
- **Network connectivity essential** for live data synchronization

## 🖥️ Desktop Application Interface

FootyForecast features an intuitive desktop environment built on modern GUI principles. The application architecture centers around a streamlined navigation system accessible through the primary menu bar, offering five distinct operational modules:

**📋 Navigation Structure:**
* **Application Hub:** League management operations (creation, loading, deletion)
* **Analytics Suite:** Statistical analysis tools and feature engineering workspace
* **ML Laboratory:** Model training environment, evaluation metrics, and prediction engine
* **Visual Customization:** Interface theming and appearance configuration
* **Learning Resources:** Comprehensive guides and machine learning documentation

**🎨 Visual Theme Collection:**
The platform includes four professionally designed interface themes optimized for different usage scenarios:

- **Breeze-Light:** Minimalist design for daytime productivity sessions
- **Breeze-Dark:** Low-light optimized interface for extended analysis periods  
- **Forest-Light:** Nature-inspired aesthetics for comfortable long-term usage
- **Forest-Dark:** Professional dark mode with enhanced visual contrast

## 📊 Advanced Team Analytics Engine

FootyForecast employs sophisticated statistical modeling to extract meaningful insights from team performance data. The analytics engine processes comprehensive datasets to generate **16 distinct performance indicators** for each participating team, creating detailed behavioral profiles for both home and visiting sides.

**🏠 Home Team Performance Metrics:**

1. **Domestic Victory Count (HW)**: Recent winning streak analysis for home venue performance
2. **Home Defeat Frequency (HL)**: Loss pattern identification within home territory
3. **Offensive Home Output (HGF)**: Cumulative scoring performance in recent home fixtures
4. **Defensive Home Vulnerability (HGA)**: Goals conceded analysis during home matches
5. **Dominant Home Victories (HGD-W)**: High-margin wins with significant goal differential (≥2 goals)
6. **Heavy Home Defeats (HGD-L)**: Substantial losses indicating defensive weaknesses
7. **Home Success Percentage (HW%)**: Season-long home venue win ratio
8. **Home Failure Rate (HL%)**: Cumulative home defeat percentage

**🛣️ Away Team Performance Indicators:**
9. **Road Victory Analysis (AW)**: Away fixture success patterns and trends
10. **Travel Defeat Metrics (AL)**: Loss frequency during away campaigns
11. **Away Scoring Efficiency (AGF)**: Goal production capability in foreign venues
12. **Road Defensive Stability (AGA)**: Defensive resilience away from home
13. **Commanding Away Wins (AGD-W)**: Decisive victories with substantial goal margins
14. **Significant Away Losses (AGD-L)**: Major defeats highlighting away vulnerabilities
15. **Away Success Ratio (AW%)**: Overall away performance success rate
16. **Road Defeat Percentage (AL%)**: Away fixture failure frequency

**⚙️ Customizable Analytics:** All performance indicators can be dynamically configured during league setup, allowing users to tailor the analytical framework to specific research requirements.

## 🌍 Global League Coverage

FootyForecast maintains comprehensive coverage of **13 premier soccer competitions** across multiple continents, sourcing official match data through the football-data.co.uk platform. The supported league ecosystem includes:
* 'Argentina': [PrimeraDivision]
* 'Belgium': [JupilerLeague]
* 'Brazil': [BrazilSerieA]
* 'China': [ChinaSuperLeague]
* 'Denmark': [SuperLiga]
* 'England': [PremierLeague, Championshio, League1, League2]
* 'Finland': [VeikkausLiiga]
* 'France': [Ligue1, Ligue2]
* 'Germany': [Bundesliga1, Bundesliga2]
* 'Greece': [SuperLeague]
* 'Ireland': [IrelandPremierDivision]
* 'Italy': [SerieA, SerieB]
* 'Japan': [J1]
* 'Mexico': [LigaMX]
* 'Netherlands': [Eredivisie]
* 'Norgway': [Eliteserien]
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

## 🔗 Statistical Correlation Framework

The correlation analysis module serves as a critical component for evaluating dataset integrity and feature relationships. FootyForecast generates interactive correlation matrices that visualize the interdependencies between statistical variables through advanced heatmap representations.

**📈 Correlation Coefficient Analysis:**
The system employs Pearson correlation coefficients (r ∈ [-1.0, 1.0]) to quantify linear relationships between feature pairs. Optimal feature selection occurs when variables demonstrate minimal correlation (approaching r = 0), indicating independent predictive value. Strong correlations (|r| > 0.7) may suggest redundant features requiring dimensionality reduction.


## 🎯 Feature Significance Assessment

The platform integrates sophisticated **model interpretability frameworks** designed to illuminate the relative importance of statistical variables in predictive accuracy. This analytical capability addresses the critical question: "Which performance metrics drive the most reliable predictions?"

**🔍 Multi-Method Importance Evaluation:**
- **Variance Analysis:** Statistical variance decomposition to identify high-impact variables
- **Recursive Feature Elimination:** Systematic backward selection using cross-validated performance metrics  
- **Tree-Based Importance Scoring:** Random Forest-derived feature ranking through impurity reduction analysis


## ⚖️ Class Distribution & Imbalance Management

Dataset analysis reveals that numerous soccer leagues exhibit **inherent class imbalances**, where home team victories significantly outnumber away team successes. This statistical skew can introduce systematic bias into machine learning models, leading to overconfident predictions favoring home teams.

**📊 Imbalance Detection & Visualization:**
FootyForecast employs **Target Distribution Analysis** to identify and quantify class imbalances across different leagues, providing visual representations of outcome frequency distributions.

**🔧 Bias Mitigation Strategies:**
- **Stochastic Noise Injection:** Controlled randomization to improve model generalization
- **Probability Calibration:** Post-processing techniques to adjust prediction confidence levels
- **Advanced Resampling Methods:** SMOTE variants, Near-Miss algorithms, and hybrid sampling approaches


## 🧠 Deep Learning Architecture Training

FootyForecast implements state-of-the-art **artificial neural network architectures** specifically optimized for soccer match prediction tasks. The deep learning framework supports multi-layer perceptron configurations with customizable activation functions, regularization techniques, and optimization algorithms.

**🔗 Technical Reference:** [Neural Network Fundamentals](https://www.investopedia.com/terms/n/neuralnetwork.asp)

## 🤖 Machine Learning Algorithm Portfolio

The platform provides access to **eight distinct algorithmic approaches**, each offering unique advantages for different prediction scenarios:

1. **K-Nearest Neighbors (KNN)** - Instance-based learning with distance metrics
2. **Logistic Regression** - Linear probabilistic classification framework
3. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
4. **Decision Tree** - Rule-based hierarchical decision structures
5. **Random Forest** - Ensemble of decision trees with bootstrap aggregation
6. **XGBoost** - Gradient boosting with advanced regularization
7. **Support Vector Machine (SVM)** - Maximum margin hyperplane optimization
8. **Deep Neural Networks** - Multi-layer artificial neural architectures

## 🌲 Ensemble Forest Training Methodology

The Random Forest implementation leverages **bootstrap aggregating (bagging)** combined with random feature selection to create robust ensemble predictions. This approach reduces overfitting while maintaining high predictive accuracy across diverse league characteristics.

**🔗 Algorithmic Details:** [Random Forest Implementation Guide](https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/)


## 🎭 Hybrid Ensemble Architecture

The ensemble methodology represents an advanced **meta-learning approach** that synthesizes predictions from multiple algorithmic sources. While individual models like Random Forest and Neural Networks may demonstrate comparable performance under optimal tuning, prediction divergence scenarios often arise where different algorithms assign varying confidence levels to the same outcome.

**🔄 Voting Mechanism:**
The ensemble framework employs **weighted probability averaging** across constituent models, leveraging the collective intelligence of diverse algorithmic perspectives. This approach capitalizes on the principle that individual model biases can be mitigated through strategic combination, resulting in superior predictive robustness.

## 📊 Model Performance Evaluation

Comprehensive model assessment constitutes a critical phase before deployment in live prediction scenarios. The evaluation framework provides detailed insights into model reliability, prediction confidence distributions, and performance characteristics across different match contexts.

**🎯 Multi-Dimensional Assessment:**
- **Cross-Model Comparison:** Systematic analysis of prediction agreement between different algorithms
- **Confidence Interval Analysis:** Statistical evaluation of prediction certainty levels
- **Stratified Performance Reports:** Accuracy metrics segmented by betting odds ranges and team strength categories


## 🔮 Match Outcome Prediction Engine

The prediction interface requires specification of **competing teams and current betting odds** to generate comprehensive match forecasts. Optimal prediction reliability occurs when multiple models demonstrate consensus; divergent predictions across algorithms typically indicate higher uncertainty scenarios requiring cautious interpretation.

**📈 Prediction Categories:**
1. **Match Result Classification:** Home Victory / Draw / Away Victory
2. **Goal Total Forecasting:** Over 2.5 Goals / Under 2.5 Goals

## 📅 Automated Fixture Processing

The **Fixture Parsing Module** enables batch prediction capabilities through automated web scraping of upcoming match schedules. This feature streamlines the prediction workflow by eliminating manual fixture entry requirements.

**🔄 Processing Workflow:**
Users select their preferred web browser and target date, triggering automated page retrieval and fixture extraction for the specified timeframe. The system processes all identified matches and generates comprehensive prediction reports for the entire fixture list.

# Requirements & Installation

Below are the steps of installing this application to your machine. First, download this code and extract it into a directory. Then, follow the steps below:

1. Download & Install python. During the installation, you should choose  **add to "Path"**. It is recommended to download **python 3.9.** or higher version.
2. After you download & install python, you can Download the above libraries using pip module (e.g. `pip install numpy==VERSION`). The version can be found in *requirements.txt* file. These modules can be installed via the cmd (in windows) or terminal (in linux). **IMPORTANT**: To download the correct versions, just add "==" after pip install to specify version, as described on requirements.txt file. For example, to install `tensorflow 2.9.1`, you can use: `pip install tensorflow==2.9.1`.
3. On windows, you can double click the main.py file. Alternatively (Both Windows & Linux), You can open the cmd on the project directory and run: `python main.py`. 

**A `requirements.txt` file has been added to the project directory. The table below presents the required libraries, however, you should check the `requirements.txt` file for the required library versions.**

| Library/Module  | Download Url | Installation |
| ------------- | ------------- | -------------
| Python Language | https://www.python.org/ | Download from website |
| Numpy  | https://numpy.org/ | `pip install numpy` |
| Pandas  | https://pandas.pydata.org/ | `pip install pandas` |
| Matplotlib  | https://matplotlib.org/ | `pip install matplotlib` |
| Seaborn  | https://seaborn.pydata.org/ | `pip install seaborn` |
| Scikit-Learn  | https://scikit-learn.org/stable/ | `pip install scikit-learn` |
| Imbalanced-Learn  | https://imbalanced-learn.org/stable/ | `pip install imbalanced-learn` |
| XGBoost  | https://xgboost.readthedocs.io/en/stable/ | `pip install xgboost` |
| Tensorflow  | https://www.tensorflow.org/ | `pip install tensorflow` |
| Tensorflow-Addons  | https://www.tensorflow.org/addons | `pip install tensorflow_addons` |
| TKinter  | https://docs.python.org/3/library/tkinter.html | `pip install tk ` |
| Optuna | https://optuna.org/ | `pip install optuna` |
| Fuzzy-Wuzzy | https://pypi.org/project/py-stringmatching | `pip install fuzzywuzzy` |
| Python-Levenshtein | https://pypi.org/project/python-Levenshtein/ | `pip install python-Levenshtein` |
| Tabulate | https://pypi.org/project/tabulate/ | `pip install tabulate` |
| Selenium | https://pypi.org/project/selenium/ | `pip install selenium` |
| LXML | https://pypi.org/project/lxml/ | `pip install lxml` |

To run `pip` commands, open CMD (windows) using Window Key + R or by typing cmd on the search. In linux, You can use the linux terminal. You can also install multiple libraries at once (e.g. `pip install numpy==1.22.4 pandas==1.4.3 ...`

# Common Errors
1. `Cannot install tensorflow.` Sometimes, it requires visual studio to be installed. Download the community edition which is free here:  [https://pypi.org/project/py-stringmatching](https://visualstudio.microsoft.com/downloads/)
2. `pip command was not found` in terminal. In this case, you forgot to choose **add to Path** option during the the installation of python. Delete python and repeat download instructions 1-3.
3. `File main.py was not found`. This is because when you open command line (cmd) tool on windows, or terminal on linux, the default directory that cmd is looking at is the home directory, not FootyForecast directory. You need to navigate to FootyForecast directory, where the main.py file exists. To do that, you can use the `cd` command. e.g. if FootyForecast is downloaded on "Downloads" folder, then type `cd Downloads/FootyForecast-Soccer-Bets-Predictor` and then type `python main.py`
4. `python command not found` on linux. This is because python command is `python3` on linux systems
5. `Parsing date is wrong` when trying to parse fixtures from the html file. The html file has many fixtures. Each fixture has a date. You need to specify the correct date of the fixture you are requesting, so the parser identifies the fixtures from the given date and grab the matches. You need to specify the date before importing the fixture file into program.
6. `<<library>> module was not found` This means that a library has been installed, but it is not included in the documentation or requirements.txt file. Try to install it via `pip` command or open an issue so that i can update the documentation.

# Supported Platforms
1. Windows
2. Linux
3. Mac

# Open An Issue
In case there is an error with the application, open a Github Issue so that I can get informed and (resolve the issue if required).

# Known Issues

1. **Neural Network's Training Dialog Height is too large and as a result, "Train" button cannot be displayed.**

Solution: You can press "ENTER" button to start training. The same applies to Random Forest Training Dialog, as well as the tuning dialogs.

# Contribution

If you liked the app and would like to contribute, You are allowed to make changes to the code and make a pull request! Usually, it takes 1-3 days for me to
review the changes and accept them or reply to you if there is something wrong.

# Citation

If you are writing an academic paper, please cite us!

```
@software{footyForecast2025,
  author = {Gregori Zeidler},
  month = {1},
  title = {{FootyForecast - An Open Source Soccer Prediction App}},
  url = {https://github.com/gregorizeidler/FootyForecast-Soccer-Bets-Predictor},
  version = {2.0.0},
  year = {2025}
}
```
