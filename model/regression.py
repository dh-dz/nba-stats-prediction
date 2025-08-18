#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
import os

# Load dataset
df_model = pd.read_csv("merged_data_features.csv", parse_dates=["date"])

# Set feature and target columns
feature_cols = ['fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 'ast_diff', 'reb_diff', 'moneyline_diff', 'teamA_home']
target_col = 'score_diff'

# Drop NaNs
df_model = df_model.dropna(subset=feature_cols + [target_col, 'spread']).copy()

# Sort by date
df_model = df_model.sort_values('date')

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=6)
all_fold_dfs = []

for train_index, test_index in tscv.split(df_model):
    train_df, test_df = df_model.iloc[train_index], df_model.iloc[test_index]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_df[feature_cols], train_df[target_col])

    test_df = test_df.copy()
    test_df['pred_score_diff'] = rf.predict(test_df[feature_cols])

    # Betting logic
    edge_margin = 1.5
    test_df['bet_A'] = test_df['pred_score_diff'] > (test_df['spread'] + edge_margin)
    test_df['bet_B'] = test_df['pred_score_diff'] < (test_df['spread'] - edge_margin)
    test_df['bet_team'] = np.where(test_df['bet_A'], 'A', np.where(test_df['bet_B'], 'B', 'None'))

    def bet_return(row):
        if row['bet_team'] == 'A':
            if row['score_diff'] > row['spread']:
                return (row['teamA_moneyLine'] / 100) * 100 if row['teamA_moneyLine'] > 0 else 10000 / abs(row['teamA_moneyLine'])
            else:
                return -100
        elif row['bet_team'] == 'B':
            if row['score_diff'] < row['spread']:
                return (row['teamB_moneyLine'] / 100) * 100 if row['teamB_moneyLine'] > 0 else 10000 / abs(row['teamB_moneyLine'])
            else:
                return -100
        return 0

    test_df['profit'] = test_df.apply(bet_return, axis=1)

    # Append full fold df
    all_fold_dfs.append(test_df)

# Concatenate results
final_df = pd.concat(all_fold_dfs, ignore_index=True)

# Ensure correct data types and sorting
final_df['date'] = pd.to_datetime(final_df['date'])
#final_df = final_df.sort_values('date')

# Cumulative Profit Plot
final_df = final_df.sort_values('date')
final_df['cumulative_profit'] = final_df['profit'].cumsum()

# Final evaluation
total_bets = (final_df['bet_team'] != 'None').sum()
total_profit = final_df['profit'].sum()
roi = total_profit / (total_bets * 100) if total_bets > 0 else 0
rmse = mean_squared_error(final_df['score_diff'], final_df['pred_score_diff'], squared=False)

# Print summary
print("Backtest Summary")
print("---------------------------")
print("Total Bets:      ", int(total_bets))
print("Total Profit ($):", round(total_profit, 2))
print("ROI:             ", round(roi * 100, 2), "%")
print("RMSE:            ", round(rmse, 2), "points")

# Save & plot
final_df = final_df.sort_values('date')
final_df['cumulative_profit'] = final_df['profit'].cumsum()
final_df.to_csv("model_predictions.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(final_df['date'], final_df['cumulative_profit'])
plt.title('Cumulative Profit Over Time (Time Series CV)')
plt.xlabel('Date')
plt.ylabel('Profit ($)')
plt.grid(True)
plt.tight_layout()
plt.show()
