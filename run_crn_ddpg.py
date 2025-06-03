# === CRN-DDPG Trading Script ===
# CausalReinforceNet implementation using DDPG for a single run.
# This script trains and evaluates a CRN agent on a given altcoin.

import os
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from env import CryptoTradingEnv

# === Configuration ===
coin_name =  # Set your altcoin symbol: ETH, XRP, LTC, etc.
roi_reward_weight = 7
training_timesteps =  # Set number of training timesteps
initial_balance = 5000

# === Paths ===
base_path =  # Set path to your local or cloned project directory, e.g., r"C:\RL\data"
train_path = os.path.join(base_path, "train_sets", f"train_set_{coin_name}.csv")
test_path = os.path.join(base_path, "test_sets", f"test_set_{coin_name}.csv")
output_path = os.path.join(base_path, "RL_automated_results_final")
os.makedirs(output_path, exist_ok=True)

# === Load Data ===
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# === Initialize Environments ===
env_train = CryptoTradingEnv(df_train, roi_reward_weight)
env_test = CryptoTradingEnv(df_test, roi_reward_weight)

# === Define DDPG Model ===
n_actions = env_train.action_space.shape[-1]
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)
policy_kwargs = dict(net_arch=[256, 256])

model = DDPG(
    "MlpPolicy",
    env_train,
    policy_kwargs=policy_kwargs,
    action_noise=ou_noise,
    verbose=0
)

model.learn(total_timesteps=training_timesteps)

# === Evaluate on Test Set ===
obs = env_test.reset()
for _ in range(len(df_test)):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env_test.step(action)
    if done:
        (
            final_balance, max_return, min_return, avg_return,
            sell_no, buy_no, hold_no,
            sell_avg, sell_max, buy_avg, buy_max
        ) = env_test.render(print_step=True, graph=False)

        net_profit = final_balance - initial_balance
        roi_total = (net_profit / initial_balance) * 100
        roi_anual = roi_total / (len(df_test) / 365)

        results = pd.DataFrame([[
            final_balance, roi_total, roi_anual,
            max_return, min_return, avg_return,
            sell_no, buy_no, hold_no,
            sell_avg, sell_max, buy_avg, buy_max
        ]], columns=[
            "Final balance", "roi_total", "roi_anual", "Max_return", "Min_return", "AVG_return",
            "sell_no", "buy_no", "hold_no", "sell_position_avg", "sell_position_max",
            "buy_position_avg", "buy_position_max"
        ])

        result_file = f"CRN_DDPG_{coin_name}_ROI{roi_reward_weight}.csv"
        results.to_csv(os.path.join(output_path, result_file), index=False)
        print(results.head())
        break
