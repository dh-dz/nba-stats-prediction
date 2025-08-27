import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

combined_data = pd.read_csv("merged_data.csv")

numeric_cols = ['FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

teamA_stats = combined_data[['date', 'teamA', 'teamA_FG_PCT', 'teamA_FT_PCT', 'teamA_FG3_PCT', 'teamA_AST', 'teamA_REB']].copy()
teamA_stats.columns = ['date', 'team', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

teamB_stats = combined_data[['date', 'teamB', 'teamB_FG_PCT', 'teamB_FT_PCT', 'teamB_FG3_PCT', 'teamB_AST', 'teamB_REB']].copy()
teamB_stats.columns = ['date', 'team', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

long_stats = pd.concat([teamA_stats, teamB_stats])
long_stats = long_stats.sort_values(['team', 'date'])

rolling_stats = (
    long_stats
    .groupby('team', group_keys=False)
    .apply(lambda x: x.sort_values('date')
           .rolling(window=5, on='date', closed='left')[numeric_cols].mean())
    .reset_index()
)
rolling_stats['team'] = long_stats['team'].values
rolling_stats['date'] = long_stats['date'].values

rolling_stats = rolling_stats.rename(columns={
    'FG_PCT': 'FG_PCT_avg',
    'FT_PCT': 'FT_PCT_avg',
    'FG3_PCT': 'FG3_PCT_avg',
    'AST': 'AST_avg',
    'REB': 'REB_avg'
})

df = combined_data.copy()

df = df.merge(rolling_stats, left_on=['date', 'teamA'], right_on=['date', 'team'], how='left')
df = df.rename(columns={col: col + '_A' for col in ['FG_PCT_avg', 'FT_PCT_avg', 'FG3_PCT_avg', 'AST_avg', 'REB_avg']})
df = df.drop(columns=['team'])

df = df.merge(rolling_stats, left_on=['date', 'teamB'], right_on=['date', 'team'], how='left')
df = df.rename(columns={col: col + '_B' for col in ['FG_PCT_avg', 'FT_PCT_avg', 'FG3_PCT_avg', 'AST_avg', 'REB_avg']})
df = df.drop(columns=['team'])

df['fg_pct_diff'] = df['FG_PCT_avg_A'] - df['FG_PCT_avg_B']
df['ft_pct_diff'] = df['FT_PCT_avg_A'] - df['FT_PCT_avg_B']
df['fg3_pct_diff'] = df['FG3_PCT_avg_A'] - df['FG3_PCT_avg_B']
df['ast_diff'] = df['AST_avg_A'] - df['AST_avg_B']
df['reb_diff'] = df['REB_avg_A'] - df['REB_avg_B']
df['moneyline_diff'] = df['teamA_moneyLine'] - df['teamB_moneyLine']

df['score_diff'] = df['teamA_score'] - df['teamB_score']

feature_cols = ['fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 'ast_diff', 'reb_diff', 'moneyline_diff']
df_model = df.dropna(subset=feature_cols + ['score_diff']).copy()
df_model.to_csv("merged_data_features.csv", index=False)

# Feature columns and target
feature_cols = ['fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 'ast_diff', 'reb_diff', 'moneyline_diff','teamA_home']
target_col = 'score_diff'

# Correlation heatmap
corr = df_model[feature_cols + [target_col]].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation with Target (score_diff)")
plt.tight_layout()
plt.show()

# Feature importance via Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(df_model[feature_cols], df_model[target_col])
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()