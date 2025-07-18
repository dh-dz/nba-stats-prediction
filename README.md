# nba-stats-prediction
The goal is to use historical NBA game statistics data (from https://www.kaggle.com/datasets/nathanlauga/nba-games, games.csv and teams.csv) and corresponding sports betting odds data (before each game, from https://www.kaggle.com/datasets/christophertreasure/nba-odds-data, oddsData.csv) to predict the game score difference, then design a betting strategy based on the score difference prediction, the moneyline and the spread (information when placing a bet). 

To see the main training result, run regression1.py and then see more analysis on the results in backtesting_analysis.ipynb.

The notebooks are used to generate merged_data_features.csv (the dataset for training) from the original dataset: data_preprocessing.ipynb cleans and combines the the games statistics data and betting odds data; feature engineering is in features.ipynb.

Currently the betting strategy is to determine whether a team is significantly undervalued in the betting by comparing the predicted score difference with the spread adding an error term (a fixed number, manually assigned). 

There are more updates coming regarding improvement in feature enigneering and betting strategy.
