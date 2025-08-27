import pandas as pd

# Load data
df = pd.read_csv("data_files/oddsData.csv")

# Create a consistent game_id: sorted teams + date
df['team_pair'] = df[['team', 'opponent']].apply(lambda x: tuple(sorted(x)), axis=1)
df['game_id'] = df['team_pair'].astype(str) + '_' + df['date'].astype(str)

# Drop duplicates (only one row per game is needed)
df = df.drop_duplicates(subset='game_id', keep='first')

# Assign teamA, teamB in alphabetical order
df[['teamA', 'teamB']] = pd.DataFrame(df['team_pair'].tolist(), index=df.index)

# Determine if teamA is the home team
df['teamA_home'] = ((df['team'] == df['teamA']) & (df['home/visitor'] == 'vs')) | \
                   ((df['team'] == df['teamB']) & (df['home/visitor'] == '@'))
df['teamA_home'] = df['teamA_home'].astype(int)

# Assign scores and moneylines using vectorized logic
is_teamA = df['team'] == df['teamA']

df['teamA_score'] = df['score'].where(is_teamA, df['opponentScore'])
df['teamB_score'] = df['opponentScore'].where(is_teamA, df['score'])

df['teamA_moneyLine'] = df['moneyLine'].where(is_teamA, df['opponentMoneyLine'])
df['teamB_moneyLine'] = df['opponentMoneyLine'].where(is_teamA, df['moneyLine'])

# Final cleaned dataframe
cleaned_df = df[[
    'date', 'season', 'teamA', 'teamB', 'teamA_home',
    'teamA_score', 'teamB_score',
    'teamA_moneyLine', 'teamB_moneyLine',
    'total', 'spread', 'secondHalfTotal'
]]

cleaned_df.to_csv("data_files/cleaned_odds_data.csv", index=False)

# Load datasets
odds_df = pd.read_csv("data_files/cleaned_odds_data.csv")
games_df = pd.read_csv("data_files/games.csv")
teams_df = pd.read_csv("data_files/teams.csv")

# Ensure SEASON is numeric
odds_df['season'] = odds_df['season'].astype(int)
games_df['SEASON'] = games_df['SEASON'].astype(int)

# Filter for seasons between 2016-2017 and 2021-2022 inclusive
odds_df = odds_df[(odds_df['season'] >= 2017) & (odds_df['season'] <= 2022)]
games_df = games_df[(games_df['SEASON'] >= 2016) & (games_df['SEASON'] <= 2021)]

# Create mapping from TEAM_ID to CITY and NICKNAME
id_to_city = dict(zip(teams_df['TEAM_ID'], teams_df['CITY']))
id_to_nickname = dict(zip(teams_df['TEAM_ID'], teams_df['NICKNAME']))

# Define function to resolve team name using city unless it's lakers or clippers
def resolve_team_name(team_id):
    nickname = id_to_nickname.get(team_id, "").lower()
    if nickname == 'lakers':
        return 'la lakers'
    elif nickname == 'clippers':
        return 'la clippers'
    else:
        return id_to_city.get(team_id, "").lower()

# Apply to home and away team columns
games_df['home_team'] = games_df['TEAM_ID_home'].apply(resolve_team_name)
games_df['away_team'] = games_df['TEAM_ID_away'].apply(resolve_team_name)

# Normalize names
odds_df['teamA'] = odds_df['teamA'].str.lower()
odds_df['teamB'] = odds_df['teamB'].str.lower()
games_df['home_team'] = games_df['home_team'].str.lower()
games_df['away_team'] = games_df['away_team'].str.lower()

# Convert date columns
odds_df['date'] = pd.to_datetime(odds_df['date'])
games_df['date'] = pd.to_datetime(games_df['GAME_DATE_EST'])

# Split odds_df based on teamA_home
home_odds = odds_df[odds_df['teamA_home'] == 1]
away_odds = odds_df[odds_df['teamA_home'] == 0]

# Case 1: teamA is home
merge_home = home_odds.merge(
    games_df,
    how='left',
    left_on=['date', 'teamA', 'teamB'],
    right_on=['date', 'home_team', 'away_team']
)

# Case 2: teamA is away
merge_away = away_odds.merge(
    games_df,
    how='left',
    left_on=['date', 'teamA', 'teamB'],
    right_on=['date', 'away_team', 'home_team']
)

# Rename stats in both to teamA_* and teamB_*
stats = ['PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB']

for col in stats:
    merge_home[f'teamA_{col}'] = merge_home[f'{col}_home']
    merge_home[f'teamB_{col}'] = merge_home[f'{col}_away']
    merge_away[f'teamA_{col}'] = merge_away[f'{col}_away']
    merge_away[f'teamB_{col}'] = merge_away[f'{col}_home']

# Combine both
merged = pd.concat([merge_home, merge_away], ignore_index=True)
merged = merged.sort_values('date').reset_index(drop=True)

merged.to_csv("data_files/merged_data.csv", index=False)
