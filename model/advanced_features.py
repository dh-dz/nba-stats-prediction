import pandas as pd
# Load the dataset
df_model = pd.read_csv("../data/data_files/merged_data_features.csv", parse_dates=["date"])

# Choose a time window
tiredness_window_days = 5

# Prepare long-format game list
games_A = df_model[['date', 'teamA']].rename(columns={'teamA': 'team'})
games_B = df_model[['date', 'teamB']].rename(columns={'teamB': 'team'})
all_games = pd.concat([games_A, games_B], ignore_index=True).drop_duplicates()

# Ensure datetime format
all_games['date'] = pd.to_datetime(all_games['date'])

# Compute tiredness: past X days' games per team
tiredness_list = []
for team in all_games['team'].unique():
    team_games = all_games[all_games['team'] == team].copy()
    team_games = team_games.sort_values('date')
    team_games['tiredness'] = team_games['date'].apply(
        lambda d: ((team_games['date'] < d) & 
                   (team_games['date'] >= d - pd.Timedelta(days=tiredness_window_days))).sum()
    )
    tiredness_list.append(team_games)

tiredness_df = pd.concat(tiredness_list, ignore_index=True).drop_duplicates(subset=['date', 'team'])

# Merge tiredness back for teamA
df_model = df_model.merge(
    tiredness_df.rename(columns={'team': 'teamA', 'tiredness': 'tiredness_A'}),
    on=['date', 'teamA'], how='left'
)

# Merge tiredness back for teamB
df_model = df_model.merge(
    tiredness_df.rename(columns={'team': 'teamB', 'tiredness': 'tiredness_B'}),
    on=['date', 'teamB'], how='left'
)
# Compute tiredness difference
df_model['tiredness_diff'] = df_model['tiredness_A'] - df_model['tiredness_B']

# === Add PTS and Create Long Format ===
teamA_records = df_model[['date', 'teamA', 'teamB', 'teamA_PTS', 'teamA_FG_PCT', 'teamA_FT_PCT', 'teamA_FG3_PCT', 'teamA_AST', 'teamA_REB']].copy()
teamA_records.columns = ['date', 'team', 'opponent', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

teamB_records = df_model[['date', 'teamB', 'teamA', 'teamB_PTS', 'teamB_FG_PCT', 'teamB_FT_PCT', 'teamB_FG3_PCT', 'teamB_AST', 'teamB_REB']].copy()
teamB_records.columns = ['date', 'team', 'opponent', 'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

# Combine both sides
long_vs = pd.concat([teamA_records, teamB_records])
long_vs = long_vs.sort_values(['team', 'opponent', 'date'])

# === Rolling Average vs Specific Opponent (excluding current game) ===
grouped_vs = long_vs.groupby(['team', 'opponent'], group_keys=False)
rolling_vs = (
    grouped_vs
    .apply(lambda x: x.sort_values('date')
           .rolling(window=3, on='date', closed='left')[['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']].mean())
    .reset_index(drop=True)
)

# Restore columns for merging
rolling_vs['team'] = long_vs['team'].values
rolling_vs['opponent'] = long_vs['opponent'].values
rolling_vs['date'] = long_vs['date'].values
# Drop duplicates from rolling_vs before merging
rolling_vs = rolling_vs.drop_duplicates(subset=['team', 'opponent', 'date'])

# Rename columns for clarity
rolling_vs = rolling_vs.rename(columns=lambda c: f"{c}_vs_opp" if c not in ['date', 'team', 'opponent'] else c)


# === Merge A team's rolling stats ===
df_model = df_model.merge(
    rolling_vs,
    left_on=['teamA', 'teamB', 'date'],
    right_on=['team', 'opponent', 'date'],
    how='left'
)
df_model = df_model.rename(columns={col: col + '_A' for col in rolling_vs.columns if col not in ['date', 'team', 'opponent']})
df_model = df_model.drop(columns=['team', 'opponent'])

# === Merge B team's rolling stats ===
df_model = df_model.merge(
    rolling_vs,
    left_on=['teamB', 'teamA', 'date'],
    right_on=['team', 'opponent', 'date'],
    how='left'
)
df_model = df_model.rename(columns={col: col + '_B' for col in rolling_vs.columns if col not in ['date', 'team', 'opponent']})
df_model = df_model.drop(columns=['team', 'opponent'])

# === Feature Engineering: Stat Differences (A - B) ===
df_model['fg_pct_vs_opp_diff']   = df_model['FG_PCT_vs_opp_A'] - df_model['FG_PCT_vs_opp_B']
df_model['ft_pct_vs_opp_diff']   = df_model['FT_PCT_vs_opp_A'] - df_model['FT_PCT_vs_opp_B']
df_model['fg3_pct_vs_opp_diff']  = df_model['FG3_PCT_vs_opp_A'] - df_model['FG3_PCT_vs_opp_B']
df_model['ast_vs_opp_diff']      = df_model['AST_vs_opp_A']     - df_model['AST_vs_opp_B']
df_model['reb_vs_opp_diff']      = df_model['REB_vs_opp_A']     - df_model['REB_vs_opp_B']
df_model['pts_vs_opp_diff']      = df_model['PTS_vs_opp_A']     - df_model['PTS_vs_opp_B']

# Drop rows with NaN in key columns
feature_cols = ['fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 'ast_diff', 'reb_diff', \
                'moneyline_diff','tiredness_diff','fg_pct_vs_opp_diff', \
                'ft_pct_vs_opp_diff', 'fg3_pct_vs_opp_diff', 'ast_vs_opp_diff', \
                'reb_vs_opp_diff', 'pts_vs_opp_diff']
target_col = 'score_diff'
df_model = df_model.dropna(subset=feature_cols + [target_col, 'spread'])

df_model = df_model[(df_model['season'] >= 2018) & (df_model['season'] <= 2021)]
# Save filtered dataset
df_model.to_csv("../data/data_files/data_advanced_features.csv", index=False)