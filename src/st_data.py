#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:41:33 2023

@author: macbook
"""

import pandas as pd
from kedro.io import DataCatalog
from kedro.extras.datasets.pandas import CSVDataSet
from typing import List
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

def __exists(var: str):
     return var in globals()

if not __exists('__DATA_CATALOG__'):
    # Create a DataCatalog to manage the data for project
    global __DATA_CATALOG__
    __DATA_CATALOG__: DataCatalog = DataCatalog()

def catalog_add_dataset(name, dataset):
    global __DATA_CATALOG__
    if not __DATA_CATALOG__:
        raise Exception('Data catalog has not been initialised.')
    else:
        __DATA_CATALOG__.add(name, dataset)

def catalog_load_dataset_data(name):
    global __DATA_CATALOG__
    if not __DATA_CATALOG__:
        raise Exception('Data catalog has not been initialised.')
    else:
        # Load the data registered with catalog
        df_data = __DATA_CATALOG__.load(name)
    return df_data
class MyDataCatalog(DataCatalog):
    def load_all_data(self) -> pd.DataFrame:
            merged_df = CSVDataSet(filepath="./data/02_intermediate/processed_merged_df.csv")
            self.add('merged_df', merged_df)
            df_data: pd.DataFrame = self.load('merged_df')
            return df_data
        
    def load_batsman_data(self) -> pd.DataFrame:
            # COMMENT: Data loading code written by ChatGPT was incorrect
            batsman_df = CSVDataSet(filepath="./data/02_intermediate/batsman_df_1.csv", load_args=None, save_args={'index': False})
            self.add('batsman_df_1', batsman_df)
            df_data: pd.DataFrame = self.load('batsman_df_1')
            return df_data
        
    def load_bowler_data(self) -> pd.DataFrame:
            # COMMENT: Data loading code written by ChatGPT was incorrect
            bowler_df = CSVDataSet(filepath="./data/02_intermediate/bowler_df_1.csv", load_args=None, save_args={'index': False})
            self.add('bowler_df_1', bowler_df)
            df_data: pd.DataFrame = self.load('bowler_df_1')
            return df_data
        
    def load_bowler_forecasting_data(self) -> pd.DataFrame:
                # COMMENT: Data loading code written by ChatGPT was incorrect
            predicted_bowl = CSVDataSet(filepath="./data/07_model_output/prediction_bowl.csv", load_args=None, save_args={'index': False})
            self.add('predicted_bowl', predicted_bowl)
            df_data: pd.DataFrame = self.load('predicted_bowl')
            return df_data
            
    def load_batsman_forecasting_data(self) -> pd.DataFrame:
                # COMMENT: Data loading code written by ChatGPT was incorrect
            predicted_bat = CSVDataSet(filepath="./data/07_model_output/prediction_bat.csv", load_args=None, save_args={'index': False})
            self.add('predicted_bat', predicted_bat)
            df_data: pd.DataFrame = self.load('predicted_bat')
            return df_data
        
    def build_data_catalog(self) -> List[str]:
        merged_df = self.load_all_data()
        batsman_df = self.load_batsman_data()
        bowler_df = self.load_bowler_data()
        predicted_bat = self.load_batsman_forecasting_data()
        predicted_bowl = self.load_bowler_forecasting_data()
        # symbols = df_data['symbol'].unique()
        # for symbol in symbols:
        #     self.filter_data_and_build_features(symbol, df_data)
        catalog_datasets = self.list()
        print(catalog_datasets)
        return catalog_datasets
