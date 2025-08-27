# NBA Stats Prediction And Betting
Profitable betting strategy on NBA game moneyline, powered by a machine learning model to predict the game score spread distribution.

The main idea is to use recent game statistics (prior to a game) difference between team A and team B as the basic features, with a rolling setup for training and test window with reasonable length, and NGboost to predict the score spread (distribution) of the game. Then by comparing the probability of team A/B winning given by the predicted distribtion and the moneyline, a betting strategy can be obtained by calculating the adjusted EV. An option of "No betting" is availible for each game.

To train the model, run the Jupyter notebook "model/train.ipynb". Then the results can be seen by running the Jupyter notebook "model/results.ipynb"

All the data csv files are stored in "data/data_files", including original data: historical NBA game statistics data (from https://www.kaggle.com/datasets/nathanlauga/nba-games, games.csv and teams.csv) and corresponding sports betting odds data (before each game, from https://www.kaggle.com/datasets/christophertreasure/nba-odds-data, oddsData.csv). Other data files are those stored middle steps with data preprocessing, feature engineering and the backtesting results.

NGboost is required to be installed to run the training.
