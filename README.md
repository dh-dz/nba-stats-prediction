# NBA Stats Prediction And Betting
Profitable and consistent betting strategy on NBA game moneyline, powered by a machine learning model to predict the game score distribution.

To run the training, run see the results, 
Orginal data: historical NBA game statistics data (from https://www.kaggle.com/datasets/nathanlauga/nba-games, games.csv and teams.csv) and corresponding sports betting odds data (before each game, from https://www.kaggle.com/datasets/christophertreasure/nba-odds-data, oddsData.csv)

To see the main training result, run regression1.py and then see more analysis on the results in backtesting_analysis.ipynb.

The notebooks are used to generate merged_data_features.csv (the dataset for training) from the original dataset: data_preprocessing.ipynb cleans and combines the the games statistics data and betting odds data; feature engineering is in features.ipynb.

Currently the betting strategy is to determine whether a team is significantly undervalued in the betting by comparing the predicted score difference with the spread adding an error term (a fixed number, manually assigned). 
