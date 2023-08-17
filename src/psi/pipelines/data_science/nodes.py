"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.12
"""
import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing


# with open("/Users/macbook/Downloads/psi/data/02_intermediate/test_bowl.csv", 'rb') as f:
#     test_bowl = pd.read_csv(f, encoding='latin1')
# test_bowl = pd.read_csv("/Users/macbook/Downloads/test_bowl.csv")
# test_bat = pd.read_csv("/Users/macbook/Downloads/test_bat.csv")
def predict_next_values_bowl(test_bowl:pd.DataFrame)->ExponentialSmoothing:
    # Create a DataFrame
    df = pd.DataFrame(test_bowl)
    #print(df.head())
    columns_to_predict = ['total_overs','total_wickets']
    smoothing_type='add'
    damping_slope=None
    lag=1
    # Create lag features for specified columns
    lagged_columns = [f'{column}_lag' for column in columns_to_predict]
    for column in columns_to_predict:
        df[f'{column}_lag'] = df.groupby('player_name')[column].shift(lag)

    # Remove rows with NaN values
    df = df.dropna()

    # Prepare data for autoregressive model
    X = df[lagged_columns]
    players = df['player_name'].unique()

    # Initialize predictions dictionary
    predictions_bowl = {}

    # Predict the next values for each player and each column
    for player in players:
        player_df = df[df['player_name'] == player]
        player_predictions = {}

        for column_to_predict in columns_to_predict:
            player_X = player_df[lagged_columns]
            player_y = player_df[column_to_predict]  # Select the specific column for prediction

            # Create and fit Exponential Smoothing model
            model = ExponentialSmoothing(player_y, trend=smoothing_type, damped_trend=(damping_slope is not None), seasonal=None)
            model_fit = model.fit(smoothing_level=0.2, damping_slope=damping_slope)

            # Predict the next values using the model
            next_predictions = model_fit.forecast(steps=1)
            
            # Limit the predicted values of "total_overs" between 0 and 4
            if column_to_predict == 'total_overs':
                next_predictions = np.clip(next_predictions, 0, 4)
            if column_to_predict == 'total_wickets':
                next_predictions = np.clip(next_predictions, 0, 10)
                
            player_predictions[column_to_predict] = (next_predictions.iloc[0]).astype('int')

        predictions_bowl[player] = player_predictions
    #predicted_df = pd.DataFrame(predictions)
    predictions_bowl = pd.DataFrame(predictions_bowl)
    
    predictions_bowl = predictions_bowl.transpose()
    predictions_bowl = predictions_bowl.rename_axis('player_name').reset_index()
    return predictions_bowl

def predict_next_values_bat(test_bat:pd.DataFrame)->ExponentialSmoothing:
    # Create a DataFrame
    df = pd.DataFrame(test_bat)
    print(df.head())
    columns_to_predict = ['total_runs','balls_faced']
    smoothing_type='add'
    damping_slope=None
    lag=1
    # Create lag features for specified columns
    lagged_columns = [f'{column}_lag' for column in columns_to_predict]
    for column in columns_to_predict:
        df[f'{column}_lag'] = df.groupby('player_name')[column].shift(lag)

    # Remove rows with NaN values
    df = df.dropna()

    # Prepare data for autoregressive model
    X = df[lagged_columns]
    players = df['player_name'].unique()

    # Initialize predictions dictionary
    predictions_bat = {}

    # Predict the next values for each player and each column
    for player in players:
        player_df = df[df['player_name'] == player]
        player_predictions = {}

        for column_to_predict in columns_to_predict:
            player_X = player_df[lagged_columns]
            player_y = player_df[column_to_predict]  # Select the specific column for prediction

            # Create and fit Exponential Smoothing model
            model = ExponentialSmoothing(player_y, trend=smoothing_type, damped_trend=(damping_slope is not None), seasonal=None)
            model_fit = model.fit(smoothing_level=0.2, damping_slope=damping_slope)

            # Predict the next values using the model
            next_predictions = model_fit.forecast(steps=1)
            
            # Limit the predicted values of "total_overs" between 0 and 4
            if column_to_predict == 'total_runs':
                next_predictions = np.clip(next_predictions, 0, 200)
            if column_to_predict == 'balls_faced':
                next_predictions = np.clip(next_predictions, 0, 100)                
            player_predictions[column_to_predict] = (next_predictions.iloc[0]).astype('int')

        predictions_bat[player] = player_predictions
    #predicted_df = pd.DataFrame(predictions)
    predictions_bat = pd.DataFrame(predictions_bat)
    
    predictions_bat = predictions_bat.transpose()
    predictions_bat = predictions_bat.rename_axis('player_name').reset_index()
    return predictions_bat
# x = predict_next_values_bowl(test_bowl)
# y = predict_next_values_bat(test_bat)
# print(x.head())
# print(y.head())