"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.12
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_processing(formatted_bbb_df:pd.DataFrame):
    #group data by player name and calculate important KPIs
    df = formatted_bbb_df
    runs_df = df.groupby('batsman_striker_name')['batsman_striker_runs'].sum().reset_index()
    balls_faced_df = df.groupby('batsman_striker_name')['batter_balls_faced'].count().reset_index()
    dismissal_df = df.groupby('batsman_striker_name')['wicket_how'].count().reset_index()
    total_matches_df = df.groupby('batsman_striker_name')['event_id'].nunique().reset_index()
    not_out_df = pd.DataFrame()
    not_out_df['batsman_striker_name'] = total_matches_df['batsman_striker_name']
    not_out_df['not_out'] = total_matches_df['event_id'] - dismissal_df['wicket_how']
    def batting_average():
    
        if isinstance(runs_df['batsman_striker_runs'], pd.Series) or isinstance(dismissal_df['wicket_how'], pd.Series):
            average = round(runs_df['batsman_striker_runs'] / dismissal_df['wicket_how'],2)
            average[dismissal_df['wicket_how'] == 0] =  runs_df['batsman_striker_runs'] # Set batting average to 0 for cases with 0 outs
            return average
        else:
            if dismissal_df['wicket_how'] == 0:
                return runs_df['batsman_striker_runs']
            else:
                average = round(runs_df['batsman_striker_runs'] / dismissal_df['wicket_how'],2)
                return average
    def batting_strike_rate():
    
        if isinstance(runs_df['batsman_striker_runs'], pd.Series) or isinstance(balls_faced_df['batter_balls_faced'], pd.Series):
            strike_rate = round((runs_df['batsman_striker_runs'] / balls_faced_df['batter_balls_faced'])*100,2)
            strike_rate[balls_faced_df['batter_balls_faced'] == 0] =  0 # Set batting average to 0 for cases with 0 outs
            return strike_rate
        else:
            if balls_faced_df['batter_balls_faced'] == 0:
                return 0
            else:
                strike_rate = round((runs_df['batsman_striker_runs'] / balls_faced_df['batter_balls_faced'])*100,2)
                return strike_rate
            
    runs_df['batting_average'] = batting_average()
    runs_df['batting_strike_rate'] = batting_strike_rate()
    batsman_df = pd.DataFrame()
    batsman_df['player_name'] = runs_df['batsman_striker_name']
    batsman_df['total_matches'] = total_matches_df['event_id']
    batsman_df['batsman_runs'] = runs_df['batsman_striker_runs']
    batsman_df['batting_avg'] = runs_df['batting_average']
    batsman_df['batting_strike_rate'] = runs_df['batting_strike_rate']
    
    conceded_df = df[(df['play_type_id'] != 7) & (df['play_type_id'] != 8)]
    total_runs_conceded = conceded_df.groupby('bowler_name')['score_value'].sum().reset_index()
    lst2 = ['caught', 'bowled' ,'leg before wicket', 'stumped',
       'hit wicket']
    conceded_df_1 = df[(df['wicket_how'].isin(lst2))]
    total_wickets = conceded_df_1.groupby('bowler_name')['wicket_how'].count().reset_index()
    total_runs_conceded = total_runs_conceded.merge(total_wickets,on='bowler_name',how='outer')
    total_runs_conceded.rename(columns={'bowler_name': 'player_name'}, inplace=True)
    total_runs_conceded['bowling_avg'] = round(total_runs_conceded['score_value']/total_runs_conceded['wicket_how'],2)
    #total_matches_bowler = df.groupby('bowler_name')['event_id'].nunique().reset_index()
    def balls_to_overs(balls):
        overs = balls // 6
        remaining_balls = balls % 6
        overs_decimal = overs + (remaining_balls / 10)
        return overs_decimal
    total_overs = df.groupby('bowler_name')['batter_balls_faced'].count().reset_index()

    total_over = balls_to_overs(total_overs['batter_balls_faced'])
    total_overs['total_overs'] = total_over
    total_overs.rename(columns={'bowler_name':'player_name'},inplace=True)
    total_runs_conceded = total_runs_conceded.merge(total_overs,on='player_name',how='outer')
    total_runs_conceded['bowling_strike_rate'] = round(total_runs_conceded['batter_balls_faced']/total_runs_conceded['wicket_how'],2)
    total_runs_conceded['economy'] = round(total_runs_conceded['score_value']/total_runs_conceded['total_overs'],2)
    bowler_df = pd.DataFrame()
    bowler_df['player_name'] = total_runs_conceded['player_name']
    bowler_df = bowler_df.merge(total_runs_conceded,on='player_name',how='outer')
    #bowler_df = bowler_df.drop('total_matches_bowl',axis=1)
    merged_df = batsman_df.merge(bowler_df, on='player_name', how='outer')
    merged_df = merged_df.fillna(0)
    merged_df.rename(columns={'wicket_how':'total_wickets'},inplace=True)
    #merged_df= merged_df.drop('Unnamed: 0',axis=1)
    merged_df['(Average * Strike Rate)/100 '] = merged_df['batting_avg']*merged_df['batting_strike_rate']*merged_df['total_matches']/100
    merged_df['[(Player Average/Tournament Average) + (Player Strike Rate/Tournament Strike Rate)] * Runs'] = \
    ((merged_df['batting_avg'] / merged_df['batting_avg'].mean()) +
     (merged_df['batting_strike_rate'] / merged_df['batting_strike_rate'].mean())) * merged_df['batsman_runs']
    
    merged_df['Bowling [(Player Average/Tournament Average) + (Player Strike Rate/Tournament Strike Rate)] * Runs'] = \
    ((merged_df['bowling_avg'] / merged_df['bowling_avg'].mean()) +
     (merged_df['bowling_strike_rate'] / merged_df['bowling_strike_rate'].mean())) * merged_df['total_wickets']
    merged_df['(Bowling Average * Strike Rate)/100 '] = merged_df['bowling_avg']*merged_df['bowling_strike_rate']*merged_df['total_matches']/100
    runs_df = df.groupby(['batsman_striker_name','event_id'])['batsman_striker_runs'].sum().reset_index()
    balls_faced_df = df.groupby(['batsman_striker_name','event_id'])['batter_balls_faced'].count().reset_index()
    dismissal_df = df.groupby(['batsman_striker_name','event_id'])['wicket_how'].count().reset_index()
    total_matches_df = df.groupby('batsman_striker_name')['event_id'].nunique().reset_index()
    def batting_strike_rate():
    
        if isinstance(runs_df['batsman_striker_runs'], pd.Series) or isinstance(balls_faced_df['batter_balls_faced'], pd.Series):
            strike_rate = round((runs_df['batsman_striker_runs'] / balls_faced_df['batter_balls_faced'])*100,2)
            strike_rate[balls_faced_df['batter_balls_faced'] == 0] =  0 # Set batting average to 0 for cases with 0 outs
            return strike_rate
        else:
            if balls_faced_df['batter_balls_faced'] == 0:
                return 0
            else:
                strike_rate = round((runs_df['batsman_striker_runs'] / balls_faced_df['batter_balls_faced'])*100,2)
                return strike_rate
    runs_df['batting_strike_rate'] = batting_strike_rate()
    runs_df['balls_faced'] = balls_faced_df['batter_balls_faced']
    batsman_df_1 = pd.DataFrame()
    batsman_df_1 = runs_df
    batsman_df_1.rename(columns={'batsman_striker_name':'player_name','batsman_striker_runs':'total_runs'},inplace=True)
    
    bowler_df_1 = df.groupby(['bowler_name','event_id']).agg({'bowler_wickets':'max',
                                                             'bowler_overs':'max'}).reset_index()
    bowler_df_1.rename(columns={'bowler_name':'player_name','bowler_wickets':'total_wickets','bowler_overs':'total_overs'},inplace=True)
    
    return merged_df,bowler_df_1,batsman_df_1

def do_pca(merged_df:pd.DataFrame) -> pd.DataFrame:
    
    # Select the relevant columns for PCA
    selected_columns = ['total_matches', 'batting_avg','batting_strike_rate','(Average * Strike Rate)/100 ','[(Player Average/Tournament Average) + (Player Strike Rate/Tournament Strike Rate)] * Runs']

    # Create a DataFrame with the selected columns
    df_pca = merged_df[selected_columns]


# Impute NaN values with column means
    df_pca.fillna(0, inplace=True)

# Standardize the data before applying PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_pca)

# Create the correlation matrix
#    correlation_matrix = pd.DataFrame(df_scaled, columns=df_pca.columns).corr()

# Display the correlation matrix
#print(correlation_matrix)

# Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(df_scaled)

# Create a DataFrame for the principal components
    df_principal_components = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3','PC4','PC5'])

# Concatenate the player_name column with the principal components DataFrame
    df_final = pd.concat([merged_df['player_name'], df_principal_components], axis=1)
    first_principal_component_weights = pca.components_[0]

# Create a DataFrame to store the weights
    weights_df = pd.DataFrame({'Feature': df_pca.columns, 'Weight': first_principal_component_weights})
    
    weighted_kpi_values = df_scaled * first_principal_component_weights
    # Sum up the weighted KPI values for each player
    df_aggregated_batting_strength = pd.Series(weighted_kpi_values.sum(axis=1))
    merged_df['batting_strength_score'] = df_aggregated_batting_strength
    
    
    selected_columns_1 = ['total_wickets','(Bowling Average * Strike Rate)/100 ','Bowling [(Player Average/Tournament Average) + (Player Strike Rate/Tournament Strike Rate)] * Runs']

# Create a DataFrame with the selected columns
    df_pca_1 = merged_df[selected_columns_1]

# Impute NaN values with column means
    df_pca_1.fillna(0, inplace=True)

# Standardize the data before applying PCA
    scaler = StandardScaler()
    df_scaled_1 = scaler.fit_transform(df_pca_1)

# Apply PCA
    pca_1 = PCA()
    principal_components_1 = pca_1.fit_transform(df_scaled_1)

# Create a DataFrame for the principal components
    df_principal_components_1 = pd.DataFrame(data=principal_components_1, columns=['PC1', 'PC2', 'PC3'])

# Concatenate the player_name column with the principal components DataFrame
    df_final_1 = pd.concat([merged_df['player_name'], df_principal_components_1], axis=1)
    first_principal_component_weights_1 = pca_1.components_[0]

# Create a DataFrame to store the weights
    weights_df_1 = pd.DataFrame({'Feature': df_pca_1.columns, 'Weight': first_principal_component_weights_1})
    weighted_kpi_values_1 = df_scaled_1 * first_principal_component_weights_1
    # Sum up the weighted KPI values for each player
    df_aggregated_bowling_strength = pd.Series(weighted_kpi_values_1.sum(axis=1))
    merged_df['bowling_strength_score'] = df_aggregated_bowling_strength
    
    return merged_df

def change_to_dict(batsman_df_1:pd.DataFrame,bowler_df_1:pd.DataFrame):
    event_id_counts = bowler_df_1['player_name'].value_counts()
    event_id_counts_bat = batsman_df_1['player_name'].value_counts()
    event_ids_to_keep = event_id_counts[event_id_counts > 2].index
    event_ids_to_keep_bat = event_id_counts_bat[event_id_counts_bat > 2].index
    filtered_bowler_df = bowler_df_1[bowler_df_1['player_name'].isin(event_ids_to_keep)]
    filtered_batsman_df = batsman_df_1[batsman_df_1['player_name'].isin(event_ids_to_keep_bat)]
    test_bowl = pd.DataFrame(filtered_bowler_df)
    test_bat = pd.DataFrame(filtered_batsman_df)
    return test_bowl,test_bat

formatted_bbb_df = pd.read_csv("/Users/macbook/Downloads/formatted_bbb_df.csv")
merged_df,bowler_df_1,batsman_df_1=data_processing(formatted_bbb_df)
do_pca(merged_df)
change_to_dict(batsman_df_1,bowler_df_1)


