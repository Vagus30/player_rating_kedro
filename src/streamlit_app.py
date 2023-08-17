#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:55:06 2023

@author: macbook
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from st_data import MyDataCatalog


def streamlit_app():
    @st.cache_data(show_spinner="Building data catalog")
    def data_catalog() -> MyDataCatalog:
        catalog = MyDataCatalog()
        datasets = catalog.build_data_catalog()
        print('Available datasets:', datasets)
        return catalog
    def load_data():
        merged_df = data_catalog().load('merged_df')
        return merged_df
    def load_data_bat():
        batsman_df = data_catalog().load('batsman_df_1')
        return batsman_df
    def load_data_bowl():
        bowler_df = data_catalog().load('bowler_df_1')
        return bowler_df
    def load_data_forecast_bowl():
        predicted_data_bowl = data_catalog().load('predicted_bowl')
        return predicted_data_bowl
    def load_data_forecast_bat():
        predicted_data_bat = data_catalog().load('predicted_bat')
        return predicted_data_bat
    pca_data = load_data()
    bowler_df = load_data_bat()
    batsman_df = load_data_bowl()
    predicted_data_bowl = load_data_forecast_bowl()
    predicted_data_bat = load_data_forecast_bat()
    print(bowler_df.columns)
    final_data = pca_data[['player_name','batting_strength_score','bowling_strength_score']]

    
# Get the predictions for the selected player

    
    st.title("Cricket Player Comparison")

# Player selection
    player1_name = st.selectbox("Select Player 1:", final_data['player_name'])
    player2_name = st.selectbox("Select Player 2:", final_data['player_name'])

# Filter the DataFrame for the selected players
    player1_data = pca_data[final_data['player_name'] == player1_name]
    player_1_runs_data = batsman_df[batsman_df['player_name'] == player1_name]
    player2_data = pca_data[final_data['player_name'] == player2_name]
    player_2_runs_data = batsman_df[batsman_df['player_name'] == player2_name]

# Display player data
    st.write(f"### {player1_name}")
    st.write("Batting Strength Score:", player1_data['batting_strength_score'].values[0].astype('int'))
    st.write("Batsman Total Run:", player_1_runs_data['total_runs'].sum().astype('int'))
    st.write("Batting Strike Rate", player_1_runs_data['batting_strike_rate'].mean().astype('int'))
    st.write("Batting Average", player1_data['batting_avg'].mean().astype('int'))
    st.write("Bowling Strength Score:", player1_data['bowling_strength_score'].values[0].astype('int'))
    st.write("Total Wickets:", player1_data['total_wickets'].sum().astype('int'))
    st.write("Total Overs:", player1_data['total_overs'].sum().astype('int'))
    st.write("Economy:", player1_data['economy'].mean().astype('float'))
    # Create a bar graph
# Group the data by event ID and sum the runs scored
    player1_plot_data = batsman_df[batsman_df['player_name'] == player1_name]
    # Filter the data for the last 10 event IDs
    last_10_event_ids = player1_plot_data['event_id'].tail(10)
    filtered_data = player1_plot_data[player1_plot_data['event_id'].isin(last_10_event_ids)]
    runs_by_event = filtered_data.groupby('event_id')['total_runs'].sum()
    str_by_event = filtered_data.groupby('event_id')['batting_strike_rate'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Plot a bar graph
    runs_by_event.plot(kind='bar', color='blue', ax=ax1)
    for idx, value in enumerate(runs_by_event):
        ax1.text(idx, value + 1, str(int(value)), ha='center', va='bottom')
    ax1.set_xlabel('Match ID')
    ax1.set_ylabel('Total Runs Scored')
    ax1.set_title('Total Runs Scored in Last 10 Matches')
    
    str_by_event.plot(kind='bar', color='green', ax=ax2)
    for idx, value in enumerate(str_by_event):
        ax2.text(idx, value + 1, f'{value:.2f}', ha='center', va='bottom')
    ax2.set_xlabel('Match ID')
    ax2.set_ylabel('Batting Strike Rate')
    ax2.set_title('Batting Strike Rate in Last 10 matches')
    

# Display the plot using Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    

    st.write(f"### {player2_name}")
    st.write("Batting Strength Score:", player2_data['batting_strength_score'].values[0].astype('int'))
    st.write("Batsman Total Run:", player_2_runs_data['total_runs'].sum().astype('int'))
    st.write("Batting Strike Rate", player_2_runs_data['batting_strike_rate'].mean().astype('int'))
    st.write("Batting Average", player2_data['batting_avg'].mean().astype('int'))
    st.write("Bowling Strength Score:", player2_data['bowling_strength_score'].values[0].astype('int'))
    st.write("Total Wickets:", player2_data['total_wickets'].sum().astype('int'))
    st.write("Total Overs:", player2_data['total_overs'].sum().astype('int'))
    st.write("Economy:", player2_data['economy'].mean().astype('float'))
    # Create a bar graph
# Group the data by event ID and sum the runs scored
    player2_plot_data = batsman_df[batsman_df['player_name'] == player2_name]
    # Filter the data for the last 10 event IDs
    last_10_event_ids = player2_plot_data['event_id'].tail(10)
    filtered_data = player2_plot_data[player2_plot_data['event_id'].isin(last_10_event_ids)]
    runs_by_event = filtered_data.groupby('event_id')['total_runs'].sum()
    str_by_event = filtered_data.groupby('event_id')['batting_strike_rate'].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Plot a bar graph
    runs_by_event.plot(kind='bar', color='blue', ax=ax1)
    for idx, value in enumerate(runs_by_event):
        ax1.text(idx, value + 1, str(int(value)), ha='center', va='bottom')
    ax1.set_xlabel('Match ID')
    ax1.set_ylabel('Total Runs Scored')
    ax1.set_title('Total Runs Scored in Last 10 Matches')
    
    str_by_event.plot(kind='bar', color='green', ax=ax2)
    for idx, value in enumerate(str_by_event):
        ax2.text(idx, value + 1, f'{value:.2f}', ha='center', va='bottom')
    ax2.set_xlabel('Match ID')
    ax2.set_ylabel('Batting Strike Rate')
    ax2.set_title('Batting Strike Rate in Last 10 Matches')

# Display the plot using Streamlit
    plt.tight_layout()
    st.pyplot(fig)
# Comparison
    st.write("### Comparison")
    
    st.write(player1_data[['player_name','total_matches','batsman_runs','batting_avg','batting_strike_rate','total_wickets','bowling_avg','total_overs','economy']])
    st.write(player2_data[['player_name','total_matches','batsman_runs','batting_avg','batting_strike_rate','total_wickets','bowling_avg','total_overs','economy']])
    if player1_data['batting_strength_score'].values[0] > player2_data['batting_strength_score'].values[0]:
        st.write(f"{player1_name} has a higher Batting Strength Score than {player2_name}.")
    else:
        st.write(f"{player2_name} has a higher Batting Strength Score than {player1_name}.")

    if player1_data['bowling_strength_score'].values[0] > player2_data['bowling_strength_score'].values[0]:
        st.write(f"{player1_name} has a higher Bowling Strength Score than {player2_name}.")
    else:
        st.write(f"{player2_name} has a higher Bowling Strength Score than {player1_name}.")


# # Initialize the app
    st.title("Player Performance Prediction")

    selected_player = st.selectbox("Select Player:", batsman_df['player_name'].unique())
# # Display the selected player
    st.write(f"Selected Player: {selected_player}")

# # Filter data for the selected player
    selected_player_bat = predicted_data_bat[predicted_data_bat['player_name'] == selected_player]
    selected_player_bowl = predicted_data_bowl[predicted_data_bowl['player_name'] == selected_player]

# # Display the predictions######################################################
    st.subheader("Predicted Batting Performance")
    st.write(f"Predicted Runs: {selected_player_bat.iloc[0,1]}")
    st.write(f"Predicted Balls Faced: {selected_player_bat.iloc[0,2]}")
#     st.write(f"Predicted Balls Faced: {balls_faced_prediction[0]}")
    
    st.subheader("Predicted Bowling Performance")
    if selected_player not in selected_player_bowl.values:
       st.write("No historical data for the selected player.")
    else:
        
        st.write(f"Predicted Overs Bowled: {selected_player_bowl.iloc[0,1]}")
        st.write(f"Predicted Wickets Taken: {selected_player_bowl.iloc[0,2]}")
if __name__ =='__main__':
    streamlit_app()