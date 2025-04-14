import pandas as pd

def main():

    dfs = []
    for start_year in range(2013, 2023):
        # Calculate the ending year (we only use the last two digits)
        end_year = str(start_year + 1)[-2:]
        filename = f'sportsreveiwsdata/NBA{start_year}-{end_year}.csv'
        
        # Read the CSV file into a DataFrame and append it to the list
        df = pd.read_csv(filename)
        dfs.append(df)

    # Combine all DataFrames into one
    data = pd.concat(dfs, ignore_index=True)


    numeric_cols = ['1st', '2nd', '3rd', '4th', 'Final', 'Open', 'Close', 'ML', '2H', 'Date', 'Rot']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')


    games = []
    for i in range(0, len(data), 2):
        game = data.iloc[i:i+2]
        team1 = game.iloc[0]
        team2 = game.iloc[1]
        
        # Aggregate quarter scores from both teams
        total_1st = team1['1st'] + team2['1st']
        total_2nd = team1['2nd'] + team2['2nd']
        total_3rd = team1['3rd'] + team2['3rd']
        total_4th = team1['4th'] + team2['4th']
        
        # The game total (actual total points) is the sum of the final scores from both teams
        total_final = team1['Final'] + team2['Final']
        
        # For the betting line, use only team1's values
        total_open = team1['Open'] if team1['Open'] > team2['Open'] else team2['Open']
        total_close = team1['Close'] if team1['Close'] > team2['Close'] else team2['Close']
        total_2H = team1['2H'] if team1['2H'] > team2['2H'] else team2['2H']
        
        # Create the over/under target:
        # Label as 1 ("Over") if total_final > total_open, else 0 ("Under")
        label = 1 if total_final > total_open else 0
        
        games.append({
            'team1': team1['Team'],  # Keep team1 identity
            'team2': team2['Team'],  # Keep team2 identity
            'total_1st': total_1st,
            'total_2nd': total_2nd,
            'total_3rd': total_3rd,
            'total_4th': total_4th,
            'total_close': total_close,
            'total_2H': total_2H,
            'total_open': total_open,
            'total_final': total_final,  # For reference only (not used as predictor)
            'label': label
        })

    games_df = pd.DataFrame(games)


    print(games_df.head())
if __name__ == '__main__':
    main()