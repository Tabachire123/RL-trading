import pandas as pd

from environment import StockPortfolioEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import (
    backtest_plot,
    backtest_stats,
    convert_daily_return_to_pyfolio_ts,
    get_baseline,
    get_daily_return,
)
from pyfolio import timeseries


class Trader():
    """
        Create an agent that can trade on the selected time frame.
    """
    def __init__(self, trade, tech_indicator_list):
        """Initialize the trader.
        
        :param trade: trading information
        :type trade: pd.DataFrame
        :param tech_indicator_list: technical indicators
        :type tech_indicator_list: list
        """
        stock_dimension = len(trade.tic.unique())
        state_space = stock_dimension

        env_kwargs = {
            "initial_amount": 1000000, 
            "transaction_cost_pct": 0, 
            "state_space": state_space, 
            "stock_dim": stock_dimension, 
            "tech_indicator_list": tech_indicator_list, 
            "action_space": stock_dimension, 
        }

        self.trade = trade
        self.e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)

    def get_trade(self):
        """Get the trade information.
        """
        return self.trade


    def compute_baseline(self):
        """Compute baseline.
        """
        baseline_df = get_baseline(
            ticker="^DJI", 
            start = pd.to_datetime(self.trade.date, format='%Y-%m-%d').min().strftime('%Y-%m-%d'),
            end =  pd.to_datetime(self.trade.date, format='%Y-%m-%d').max().strftime('%Y-%m-%d'))

        baseline_df_stats = backtest_stats(baseline_df, value_col_name = 'close')
        baseline_returns = get_daily_return(baseline_df, value_col_name="close")

        dji_cumpod =(baseline_returns+1).cumprod()-1
        
    def backtest(self, model):
        """Backtest the model.

        :param model: model
        :type model: stable_baselines.common.models.Model
        :return: backtest results
        :rtype: pd.DataFrame
        """
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model, environment = self.e_trade_gym)

        time_ind = pd.Series(df_daily_return.date)
        model_cumpod =(df_daily_return.daily_return+1).cumprod()-1
        DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)

        perf_func = timeseries.perf_stats 
        perf_stats_all = perf_func(returns=DRL_strat, factor_returns=DRL_strat, positions=None, transactions=None, turnover_denom="AGB")

        return df_actions
