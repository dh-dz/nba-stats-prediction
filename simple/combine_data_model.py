#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:56:35 2025

@author: daohengniu
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load merged dataset
merged_df = pd.read_csv("merged_game_odds_data.csv")
merged_df['date'] = pd.to_datetime(merged_df['date'])

# Assign team IDs for both sides
merged_df['teamA_id'] = merged_df['HOME_TEAM_ID']
merged_df['teamB_id'] = merged_df['VISITOR_TEAM_ID']

# Build long-form stats for both sides
team_stats = merged_df[['date', 'teamA_id', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_home']].copy()
team_stats.columns = ['date', 'team_id', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'PTS']

opp_stats = merged_df[['date', 'teamB_id', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'PTS_away']].copy()
opp_stats.columns = ['date', 'team_id', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'PTS']

# Combine and sort for rolling averages
long_stats = pd.concat([team_stats, opp_stats])
long_stats = long_stats.sort_values(['team_id', 'date'])

# Rolling 5-game average stats
rolling = (
    long_stats
    .groupby('team_id', group_keys=False)
    .apply(lambda df: df.sort_values('date').rolling(window=5, on='date').mean())
    .reset_index()
)

rolling = rolling.rename(columns={
    'FG_PCT': 'FG_PCT_avg',
    'FT_PCT': 'FT_PCT_avg',
    'FG3_PCT': 'FG3_PCT_avg',
    'AST': 'AST_avg',
    'REB': 'REB_avg',
    'PTS': 'PTS_avg'
})

# Merge rolling averages back to merged_df
merged_df = merged_df.merge(rolling, left_on=['date', 'teamA_id'], right_on=['date', 'team_id'], how='left')
merged_df = merged_df.merge(rolling, left_on=['date', 'teamB_id'], right_on=['date', 'team_id'], how='left', suffixes=('_A', '_B'))

# Feature engineering: stat differences between teams
merged_df['fg_pct_diff'] = merged_df['FG_PCT_avg_A'] - merged_df['FG_PCT_avg_B']
merged_df['ft_pct_diff'] = merged_df['FT_PCT_avg_A'] - merged_df['FT_PCT_avg_B']
merged_df['fg3_pct_diff'] = merged_df['FG3_PCT_avg_A'] - merged_df['FG3_PCT_avg_B']
merged_df['ast_diff'] = merged_df['AST_avg_A'] - merged_df['AST_avg_B']
merged_df['reb_diff'] = merged_df['REB_avg_A'] - merged_df['REB_avg_B']
merged_df['pts_diff'] = merged_df['PTS_avg_A'] - merged_df['PTS_avg_B']
merged_df['moneyline_diff'] = merged_df['teamA_moneyLine'] - merged_df['teamB_moneyLine']

# Target
merged_df['score_diff'] = merged_df['teamA_score'] - merged_df['teamB_score']

# Clean NaNs
feature_cols = ['fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 'ast_diff', 'reb_diff', 'pts_diff', 'moneyline_diff']
merged_df = merged_df.dropna(subset=feature_cols + ['score_diff', 'spread'])

# Time-based split
merged_df = merged_df.sort_values('date')
split_date = merged_df['date'].quantile(0.8)
train_df = merged_df[merged_df['date'] < split_date]
test_df = merged_df[merged_df['date'] >= split_date]

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_df[feature_cols], train_df['score_diff'])

# Predict
test_df = test_df.copy()
test_df['pred_score_diff'] = rf.predict(test_df[feature_cols])

# Betting strategy
edge_margin = 1.5
test_df['bet'] = test_df['pred_score_diff'] > (test_df['spread'] + edge_margin)

def bet_return(row):
    if not row['bet']:
        return 0
    if row['score_diff'] > row['spread']:
        if row['teamA_moneyLine'] > 0:
            return (row['teamA_moneyLine'] / 100) * 100
        else:
            return 10000 / abs(row['teamA_moneyLine'])
    else:
        return -100

test_df['profit'] = test_df.apply(bet_return, axis=1)

# Evaluation
total_bets = test_df['bet'].sum()
total_profit = test_df['profit'].sum()
roi = total_profit / (total_bets * 100) if total_bets > 0 else 0
rmse = mean_squared_error(test_df['score_diff'], test_df['pred_score_diff'], squared=False)

# Output results
print("Backtest Summary")
print("---------------------------")
print("Total Bets:      ", int(total_bets))
print("Total Profit ($):", round(total_profit, 2))
print("ROI:             ", round(roi * 100, 2), "%")
print("RMSE:            ", round(rmse, 2), "points")

import matplotlib.pyplot as plt

# Ensure test_df is sorted by date
test_df = test_df.sort_values('date')

# Compute cumulative profit
test_df['cumulative_profit'] = test_df['profit'].cumsum()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(test_df['date'], test_df['cumulative_profit'])
plt.title('Cumulative Profit Over Time')
plt.xlabel('Date')
plt.ylabel('Profit ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

