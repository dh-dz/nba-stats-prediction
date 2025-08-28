# NBA Stats Prediction And Betting
Profitable betting strategy on NBA game moneyline, powered by a machine learning model to predict the game score spread distribution.

The main idea is to use recent game statistics (prior to a game) difference between team A and team B as the basic features, with a rolling setup for training and test window with reasonable length, and NGboost to predict the score spread (distribution) of the game. Then by comparing the probability of team A/B winning given by the predicted distribtion and the moneyline, a betting strategy can be obtained by calculating the adjusted EV. An option of "No betting" is availible for each game.

To train the model, run the Jupyter notebook "model/train.ipynb". Then the results can be seen by running the Jupyter notebook "model/results.ipynb"

All the data csv files are stored in "data/data_files", including original data: historical NBA game statistics data (from https://www.kaggle.com/datasets/nathanlauga/nba-games, games.csv and teams.csv) and corresponding sports betting odds data (before each game, from https://www.kaggle.com/datasets/christophertreasure/nba-odds-data, oddsData.csv). Other data files are those stored middle steps with data preprocessing, feature engineering and the backtesting results.

NGboost is required to be installed to run the training.

# NBA Stats Prediction and Betting

A machine learning–driven framework for predicting NBA game outcomes and designing a profitable betting strategy on **moneyline bets**.

## Overview

This project leverages recent NBA game statistics to model the score spread distribution and derive betting decisions. The core workflow is:

1. **Feature Engineering**  
   For each upcoming game, compute the difference in recent statistics between **Team A** and **Team B** as the primary features.

2. **Rolling Training & Testing**  
   Use a rolling window setup with reasonable training and test lengths to better capture temporal dynamics.

3. **Modeling with NGBoost**  
   Train an [NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/) model to predict the probability distribution of the game score spread.

4. **Betting Strategy**  
   Compare the model-derived win probabilities with sportsbook moneyline odds.  
   - Calculate the **adjusted expected value (adjusted EV)** for each bet.  
   - Choose to bet on Team A, Team B, or take the **"No Bet"** option.

## Usage

- **Training**: Run the Jupyter notebook [`model/train.ipynb`](model/train.ipynb) to train the NGBoost model.  
- **Results & Backtesting**: Run [`model/results.ipynb`](model/results.ipynb) to visualize results and evaluate backtested performance.

## Data

All CSV data files are located in [`data/data_files`](data/data_files):

- **Original datasets**:
  - NBA game statistics (from [Kaggle: NBA Games](https://www.kaggle.com/datasets/nathanlauga/nba-games))  
    - `games.csv`, `teams.csv`
  - Pre-game moneyline odds (from [Kaggle: NBA Odds Data](https://www.kaggle.com/datasets/christophertreasure/nba-odds-data))  
    - `oddsData.csv`

- **Intermediate files**:  
  Processed data for feature engineering, model training, and backtesting results.

## Requirements

- [NGBoost](https://github.com/stanfordmlgroup/ngboost)  
  Install via pip:  
  ```bash
  pip install ngboost
