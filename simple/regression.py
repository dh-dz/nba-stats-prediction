import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
cleaned_df = pd.read_csv("cleaned_odds_data.csv")
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

# Create target: score difference
cleaned_df['score_diff'] = cleaned_df['teamA_score'] - cleaned_df['teamB_score']
cleaned_df['teamA_win'] = (cleaned_df['score_diff'] > 0).astype(int)

# Implied probability from moneyline
def moneyline_to_implied_prob(ml):
    return 100 / (ml + 100) if ml > 0 else abs(ml) / (abs(ml) + 100)

cleaned_df['teamA_implied_prob'] = cleaned_df['teamA_moneyLine'].apply(moneyline_to_implied_prob)

# Rolling historical features (teamA only)
cleaned_df = cleaned_df.sort_values(by='date')
cleaned_df['teamA_winrate_5'] = (
    cleaned_df.groupby('teamA')['teamA_win']
    .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
)
cleaned_df['teamA_avg_ml_5'] = (
    cleaned_df.groupby('teamA')['teamA_moneyLine']
    .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
)

# Drop rows with missing rolling features
cleaned_df = cleaned_df.dropna(subset=['teamA_winrate_5', 'teamA_avg_ml_5']).reset_index(drop=True)

# Time-aware split
split_date = cleaned_df['date'].quantile(0.8)
train_df = cleaned_df[cleaned_df['date'] < split_date]
test_df = cleaned_df[cleaned_df['date'] >= split_date]

# Features and target
features = ['teamA_winrate_5', 'teamA_avg_ml_5', 'teamA_implied_prob']
target = 'score_diff'

# Train model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(train_df[features], train_df[target])

# Predict
test_df = test_df.copy()
test_df['pred_score_diff'] = regressor.predict(test_df[features])

# Define betting strategy: bet if predicted score_diff beats spread by margin
edge_margin = 1.5
test_df['bet'] = test_df['pred_score_diff'] > (test_df['spread'] + edge_margin)

# Profit calculation based on actual moneyline
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

# Optional: Save or analyze result
test_df[['date', 'teamA', 'teamB', 'score_diff', 'spread', 'pred_score_diff', 'bet', 'teamA_moneyLine', 'profit']].to_csv("betting_predictions.csv", index=False)

# Define result metrics
total_bets = test_df['bet'].sum()
total_profit = test_df['profit'].sum()
roi = total_profit / (total_bets * 100) if total_bets > 0 else 0
rmse = mean_squared_error(test_df['score_diff'], test_df['pred_score_diff'], squared=False)

# Print result summary
print("Backtest Summary")
print("---------------------------")
print("Total Bets:      ", int(total_bets))
print("Total Profit ($):", round(total_profit, 2))
print("ROI:             ", round(roi * 100, 2), "%")
print("RMSE:            ", round(rmse, 2), "points")

import matplotlib.pyplot as plt

# Cumulative profit over time
test_df['cumulative_profit'] = test_df['profit'].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(test_df['date'], test_df['cumulative_profit'], label='Cumulative Profit')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Cumulative Profit ($)')
plt.title('Cumulative Profit Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
