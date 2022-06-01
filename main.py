import logging

import numpy as np
import pandas as pd

from data_loader import DataLoader
from explainability import calculate_meta_q
from finrl.finrl_meta.preprocessor.preprocessors import data_split
from trader import Trader
from trainer import Trainer


logging.basicConfig(level=logging.DEBUG)

TECHNICAL_INDICATORS = [
    'macd',
    'rsi_30',
    'rsi_5',
    'dx_30',
    'dx_5',
    'atr_30',
    'atr_5'
]

# Load the data
data_loader = DataLoader(TECHNICAL_INDICATORS)
df = data_loader.get_data()
logging.info('----- Data loaded! -----')


# Load the Agent and train it
train = data_split(df, '2009-01-01','2020-06-30')
trainer = Trainer(train, TECHNICAL_INDICATORS)

a2c = trainer.train_a2c()
#ppo = trainer.train_ppo()

logging.info('----- Model trained! -----')


# Backtesting
test = data_split(df,'2020-07-01', '2021-09-02')
trader = Trader(test, TECHNICAL_INDICATORS)
df_actions = trader.backtest(a2c)
df_actions.to_csv('a2c_actions.csv') # Save actions

logging.info('----- Backtest Finished! -----')


# Explainability

#meta_q = calculate_meta_q(trader, ppo, 'ppo', df_actions, TECHNICAL_INDICATORS)
meta_q = calculate_meta_q(trader, a2c, 'A2C', df_actions, TECHNICAL_INDICATORS)
meta_q.to_csv('meta_q_0_a2c.csv') # Save the explainability values

logging.info('----- Meta_q Finished! -----')
