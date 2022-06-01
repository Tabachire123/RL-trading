import os

import pandas as pd

from finrl.apps import config
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader

class DataLoader():
    """ Loads and stores the data."""
    def __init__(self, technical_indicators):
        """Class constructor.

        :param technical_indicators: list of technical indicators to be used
        :type technical_indicators: list
        """

        # Create directories if they don't exist
        for directory in [config.DATA_SAVE_DIR, config.TRAINED_MODEL_DIR, config.TENSORBOARD_LOG_DIR, config.RESULTS_DIR]:
            if not os.path.exists("./" + directory):
                os.makedirs("./" + directory)

        self.technical_indicators = technical_indicators
        self.df = pd.DataFrame()

    def get_data(self, start: str = '2008-01-01', end: str = '2021-09-02'):
        """Downloads, preprocess and stores the data if it's not already stored. Returns the dataframe.

        :param start: start date
        :type start: str
        :param end: end date
        :type end: str
        :return df: dataframe with the data
        :rtype df: pandas.DataFrame
        """
        try:
            # Load data if it's already stored
            self.df = pd.read_pickle(f"{config.DATA_SAVE_DIR}/data.pkl")
            return self.df
        except:
            # Download data
            self.df = YahooDownloader(start_date = start, end_date = end, ticker_list = config.DOW_30_TICKER).fetch_data()

            # Preprocess data
            feature_engineer = FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature = False)
            feature_engineer.tech_indicator_list = self.technical_indicators

            self.df = feature_engineer.preprocess_data(self.df)

            self.__add_covariance()

            # Save data
            self.df.to_pickle(f"{config.DATA_SAVE_DIR}/data.pkl")

            return self.df

    def __add_covariance(self):
        # add covariance matrix as states
        self.df=self.df.sort_values(['date','tic'],ignore_index=True)
        self.df.index = self.df.date.factorize()[0]

        cov_list = []
        return_list = []

        # Compute covariance matrix
        lookback=252
        for i in range(lookback,len(self.df.index.unique())):
            data_lookback = self.df.loc[i-lookback:i,:]
            price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)

            covs = return_lookback.cov().values 
            cov_list.append(covs)


        # Add covariance matrix as states to the dataframe
        df_cov = pd.DataFrame({'date':self.df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        self.df = self.df.merge(df_cov, on='date')
        self.df = self.df.sort_values(['date','tic']).reset_index(drop=True)
