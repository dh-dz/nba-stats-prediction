# NBA Stats Prediction and Betting

A machine learning–driven framework for predicting NBA game outcomes and designing a profitable betting strategy on **moneyline bets**.

## Overview

This project leverages recent NBA game statistics to model the score spread distribution and derive betting decisions. The core workflow is:

1. **Feature Engineering**  
   For each upcoming game, the primary features include the difference in recent performance statistics between **Team A** and **Team B**, as well as the pre-game betting odds. Additional factors such as **home-court advantage**, **team fatigue (tiredness)**, and **head-to-head performance in recent matchups** are also incorporated.

1. **Feature Engineering**  
   For each upcoming game, primary features are the difference in recent game statistics between **Team A** and **Team B** and the betting odds data before the game. We also consider home court advantage, team tiredness and statistics difference in recent games against the specific opponent. 
   
3. **Rolling Training & Testing**  
   Use a rolling window setup with reasonable training and test lengths to better capture temporal dynamics.

4. **Modeling with NGBoost**  
   Train an [NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/) model to predict the probability distribution of the game score spread.

5. **Betting Strategy**  
   Compare the model-derived win probabilities with sportsbook moneyline odds.  
   - Calculate the **adjusted expected value (adjusted EV)** for each bet.  
   - Choose to bet on Team A, Team B, or take the **"No Bet"** option.

## Usage

- **Training**: Run the Jupyter notebook [`model/training.ipynb`](model/training.ipynb) to train the NGBoost model.  
- **Results & Backtesting**: Run [`model/results.ipynb`](model/results.ipynb) to visualize results and evaluate backtested performance.

## Data

All CSV data files are located in [`data/data_files`](data/data_files):

- **Original datasets**:
  - NBA game statistics (from [Kaggle: NBA Games](https://www.kaggle.com/datasets/nathanlauga/nba-games))  
    - `games.csv`, `teams.csv`
  - Pre-game moneyline odds (from [Kaggle: NBA Odds Data](https://www.kaggle.com/datasets/christophertreasure/nba-odds-data))  
    - `oddsData.csv`

- **Intermediate files**:  
  Files of data preprocessing, feature engineering, and backtesting results
## Requirements

- [NGBoost](https://github.com/stanfordmlgroup/ngboost)  
  Install via pip:  
  ```bash
  pip install ngboost
