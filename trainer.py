import logging

from environment import StockPortfolioEnv
from finrl.apps import config
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, PPO

class Trainer():
    """Create an agent that will be used to train the model.
    """
    def __init__(self, train, tech_indicator_list):
        """Initialize the trainer.

        :params train: training information.
        :type train: pd.DataFrame
        :params tech_indicator_list: list of technical indicators
        :type tech_indicator_list: list
        """
        stock_dimension = len(train.tic.unique())
        state_space = stock_dimension

        env_kwargs = {
            "initial_amount": 1000000, 
            "transaction_cost_pct": 0, 
            "state_space": state_space, 
            "stock_dim": stock_dimension, 
            "tech_indicator_list": tech_indicator_list, 
            "action_space": stock_dimension, 
        }

        e_train_gym = StockPortfolioEnv(df = train, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()

        self.agent = DRLAgent(env = env_train)


    def train_a2c(self, load=True):
        """Train the agent using A2C.

        :params load: load the model from disk.
        :type load: bool
        :returns: trained agent
        :rtype: stable_baselines3.A2C
        """
        if load:
            try:
                model = A2C.load(f"{config.TRAINED_MODEL_DIR}/a2c")
                logging.info("----- Model loaded! ----")
                return model
            except:
                A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
                model = self.agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

                model = self.agent.train_model(model=model, tb_log_name='a2c', total_timesteps=40000)

                model.save(f"{config.TRAINED_MODEL_DIR}/a2c")
                return model

        else:
            A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
            model = self.agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

            model = self.agent.train_model(model=model, tb_log_name='a2c', total_timesteps=40000)

            model.save(f"{config.TRAINED_MODEL_DIR}/a2c")
            return model

        

    def train_ppo(self, load=True):
        """Train the agent using PPO.

        :params load: load the model from disk.
        :type load: bool
        :returns: trained agent
        :rtype: stable_baselines3.PPO
        """
        if load:
            try:
                model = PPO.load(f"{config.TRAINED_MODEL_DIR}/ppo")
                logging.info("----- Model loaded! ----")
                return model
            except:
                PPO_PARAMS = {
                    "n_steps": 2048,
                    "ent_coef": 0.005,
                    "learning_rate": 0.001,
                    "batch_size": 128,
                }
                model_ppo = self.agent.get_model("ppo",model_kwargs = PPO_PARAMS)

                model_ppo =  self.agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=40000)

                model_ppo.save(f"{config.TRAINED_MODEL_DIR}/ppo")
                return model_ppo
        else:
            PPO_PARAMS = {
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.001,
                "batch_size": 128,
            }
            model_ppo = self.agent.get_model("ppo",model_kwargs = PPO_PARAMS)

            model_ppo =  self.agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=40000)

            model_ppo.save(f"{config.TRAINED_MODEL_DIR}/ppo")
            return model_ppo
