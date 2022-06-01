import gym
from gym import spaces
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv
matplotlib.use('Agg')


class StockPortfolioEnv(gym.Env):
    """Build a stock trading environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                stock_dim,
                initial_amount,
                transaction_cost_pct,
                state_space,
                action_space,
                tech_indicator_list,
                lookback=252,
                day = 0):
        """Constructor for the StockPortfolioEnv class.

        :param df: pandas dataframe with the stock prices and other information
        :type df: pandas.DataFrame
        :param stock_dim: number of unique stocks
        :type stock_dim: int
        :param initial_amount: initial amount of money
        :type initial_amount: int
        :param transaction_cost_pct: transaction cost percentage per trade
        :type transaction_cost_pct: float
        :param state_space: dimension of the state space
        :type state_space: int
        :param action_space: dimension of the action space
        :type action_space: int
        :param tech_indicator_list: list of technical indicator names
        :type tech_indicator_list: list
        :param lookback: number of days to look back for the state
        :type lookback: int
        :param day: day of the simulation
        :type day: int
        """
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.portfolio_value = self.initial_amount
        self.terminal = False

        # action_space and observation_space definition
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))

        # load data starting at self.day
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]

        # Build state (covariance matrix + technical indicators)
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]

        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]

        
    def step(self, actions):
        """Perform one step of the environment's dynamics.
           At each step the agent will perform the actions,
           calculate the reward, and return the next observation.

        :param actions: actions to be performed
        :type actions: numpy.ndarray
        :return (state, reward, terminal, _): new state, reward, terminal flag, and auxiliary data
        :type (numpy.ndarray, float, bool, dict): tuple
        """
        # Check if the episode is done
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            self.__plot_results()
            self.__sumarize_results()
        else:
            # Normalize actions
            weights = self.softmax_normalization(actions) 
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # Create next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            log_portfolio_return = np.log(sum((self.data.close.values / last_day_memory.close.values)*weights))

            # Update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            former_portfolio_value = self.portfolio_value
            self.portfolio_value = new_portfolio_value

            # Save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # Compute reward
            self.reward = new_portfolio_value / former_portfolio_value
            

        return self.state, self.reward, self.terminal, {}

    def __plot_results(self):
        """Plot the results of the simulation.

        :return: None
        """
        df = pd.DataFrame(self.portfolio_return_memory)
        df.columns = ['daily_return']

        plt.plot(df.daily_return.cumsum(),'r')
        plt.savefig('results/cumulative_reward.png')
        plt.close()
        
        plt.plot(self.portfolio_return_memory,'r')
        plt.savefig('results/rewards.png')
        plt.close()


    def __sumarize_results(self):
        """Summarize and print the results of the experiment.

        :return: None
        """
        print("-"*20)
        print(f"Initial portfolio value: {self.asset_memory[0]}")
        print(f"Final portfolio value: {self.portfolio_value}")

        df_daily_return = pd.DataFrame(self.portfolio_return_memory)
        df_daily_return.columns = ['daily_return']
        if df_daily_return['daily_return'].std() !=0:
          sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                   df_daily_return['daily_return'].std()
          print("Sharpe: ",sharpe)
        print("-"*20)


    def reset(self):
        """Reset the environment.
        """
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]

        # Load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.portfolio_value = self.initial_amount

        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]] 

        return self.state
    
    def render(self, mode='human'):
        """Render the environment.
        """
        return self.state
        
    def softmax_normalization(self, actions):
        """Normalize actions to be between 0 and 1.

        :param actions: actions to be normalized
        :type actions: numpy.ndarray
        :return: normalized actions
        :type: numpy.ndarray
        """
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output

    def save_asset_memory(self):
        """ Compute account value at each time step.
        """
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """ Return actions and positions at each time step.
        """
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
