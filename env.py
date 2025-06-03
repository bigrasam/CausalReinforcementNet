import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features



class CryptoTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, roi_weight, sharpe_ratio_weight):
        # Auto-detect technical indicators in DataFrame
        self.df = df.copy()
        self.roi_weight = roi_weight
        self.sharpe_ratio_weight = sharpe_ratio_weight

        # Determine which columns to use
        self.base_features = ["Open", "High", "Low", "Close", "Volume"]
        self.dbn_features = ["DBN_Up", "DBN_Down"]
        self.tech_features = [
            "RSI", "MACD", "EMA", "SMA", "OBV", "BBands", "AD", "Stoch", "AROON", "CCI"
        ]
        self.internal_features = ["balance_ratio", "step_return", "shares_held"]

        # Ensure technical indicators are present in the DataFrame
        available_features = list(self.df.columns)
        self.used_tech_features = [f for f in self.tech_features if any(f in c for c in available_features)]

        # Calculate total features for observation space
        n_obs = len(self.base_features) + len(self.dbn_features) + len(self.internal_features) + len(self.used_tech_features)

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_obs,))

        # Trading state variables
        self.current_step = 0
        self.balance = 5000
        self.initial_balance = self.balance
        self.shares_held = 0
        self.last_wallet_value = self.balance
        self.done = False
        self.action_sell = 0
        self.action_buy = 0
        self.action_hold = 0
        self.max_return = 0
        self.min_return = 0
        self.avg_return = 0
        self.step_return = 0
        self.step_profit_percentage = 0

        self.returns = []
        self.array_balance = []
        self.array_profits = []
        self.sell_positions = []
        self.buy_positions = []

        self.sell_position_max = 0
        self.sell_position_avg = 0
        self.buy_position_max = 0
        self.buy_position_avg = 0
        self.step_profit = 0

    def _next_observation(self):
        row = self.df.iloc[self.current_step]

        base = np.array([row[f] for f in self.base_features])
        dbn = np.array([row[f] for f in self.dbn_features])
        tech = np.array([row[f] for f in self.used_tech_features]) if self.used_tech_features else np.array([])
        internal = np.array([
            self.balance / self.initial_balance,
            self.step_return,
            self.shares_held
        ])

        # Normalize base and tech together
        full_data_cols = self.base_features + self.used_tech_features
        data_slice = self.df[full_data_cols].values
        min_val = np.min(data_slice, axis=0)
        max_val = np.max(data_slice, axis=0)
        norm_base_tech = (np.concatenate([base, tech]) - min_val) / (max_val - min_val + 1e-8)

        obs = np.concatenate([norm_base_tech, dbn, internal])
        return obs


    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        action_type = int(np.sign(action[0]))
        position_size = (action[1] + 1) * 0.1 + 0.4

        if action_type == -1:
            number_shares_sell = position_size * self.shares_held
            transaction_cost = number_shares_sell * current_price * 0.001
            money_received = number_shares_sell * current_price
            self.shares_held -= number_shares_sell
            self.balance += money_received - transaction_cost
            self.sell_positions.append(position_size)
            self.action = "sell"
            self.action_sell += 1

        elif action_type == 0:
            self.action = "hold"
            self.action_hold += 1

        elif action_type == 1:
            money_to_spend = position_size * self.balance
            transaction_cost = money_to_spend * 0.001
            number_shares_bought = money_to_spend / current_price
            self.balance -= money_to_spend + transaction_cost
            self.shares_held += number_shares_bought
            self.buy_positions.append(position_size)
            self.action = "buy"
            self.action_buy += 1

        self.step_profit = self.balance - self.last_wallet_value
        self.step_return = (self.step_profit / self.last_wallet_value) * 100
        self.last_wallet_value = self.balance

        self.array_balance.append(self.balance)
        self.returns.append(self.step_return)

        if self.returns:
            self.max_return = max(self.returns)
            self.min_return = min(self.returns)
            self.avg_return = sum(self.returns) / len(self.returns)

        if self.buy_positions:
            self.buy_position_max = max(self.buy_positions)
            self.buy_position_avg = sum(self.buy_positions) / len(self.buy_positions)

        if self.sell_positions:
            self.sell_position_max = max(self.sell_positions)
            self.sell_position_avg = sum(self.sell_positions) / len(self.sell_positions)

    def step(self, action):
        if np.isnan(action).any():
            action = np.random.uniform(-1, 1, size=2)
        else:
            assert self.action_space.contains(action), f"Action {action} is not valid"

        if self.done:
            return self._next_observation(), 0, self.done, {}

        self._take_action(action)
        self.current_step += 1

        reward = self.reward_function()
        done = self.current_step >= len(self.df) - 1
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.balance = 5000
        self.current_step = 0
        self.shares_held = 0
        self.last_wallet_value = self.balance
        self.initial_balance = self.balance
        self.action_sell = 0
        self.action_buy = 0
        self.action_hold = 0
        self.max_return = 0
        self.min_return = 0
        self.avg_return = 0
        self.step_return = 0
        self.step_profit_percentage = 0
        self.done = False

        self.returns = []
        self.array_profits = []
        self.array_balance = []
        self.sell_positions = []
        self.buy_positions = []
        self.sell_position_max = 0
        self.sell_position_avg = 0
        self.buy_position_max = 0
        self.buy_position_avg = 0

        return self._next_observation()

    def render(self, print_step=False, graph=False, *args, mode="human", close=False):
        final_balance = self.balance + (
            self.shares_held * self.df.loc[self.current_step, "Close"]
        )
        if print_step:
            print("***********************************************")
            print(f"last action: {self.action}")
            print(f"Step: {self.current_step}")
            print(f"Balance: {self.balance:.2f} (Profit: {final_balance:.2f})")
        return (
            final_balance,
            self.max_return,
            self.min_return,
            self.avg_return,
            self.action_sell,
            self.action_buy,
            self.action_hold,
            self.sell_position_avg,
            self.sell_position_max,
            self.buy_position_avg,
            self.buy_position_max,
        )

    def reward_function(self):
        calculated_reward = self.combine_rewards()
        delay_modifier = self.current_step / len(self.df.loc[:, "Open"].values)
        reward = calculated_reward * delay_modifier
        return reward

    def combine_rewards(self):
        sharpe_ratio_weight = self.sharpe_ratio_weight / 10
        roi_weight = self.roi_weight / 10
        sharpe_ratio_reward = self.calculate_sharpe_ratio()
        roi_reward = self.calculate_roi()
        return sharpe_ratio_weight * sharpe_ratio_reward + roi_weight * roi_reward

    def calculate_sharpe_ratio(self):
        returns = np.array(self.returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        daily_rf = self.risk_free_rate / 365
        if std_return == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (mean_return - daily_rf) / std_return

        if sharpe_ratio >= 4:
            return 10
        elif 1 <= sharpe_ratio < 4:
            return 4
        elif 0 < sharpe_ratio < 1:
            return 1
        elif sharpe_ratio == 0:
            return 0
        elif -1 <= sharpe_ratio < 0:
            return -1
        elif -4 <= sharpe_ratio < -1:
            return -4
        else:
            return -10

    def calculate_roi(self):
        initial_investment = self.initial_balance
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        price = self.df.iloc[self.current_step]["Close"]
        current_investment = self.balance + (self.shares_held * price)
        net_profit_step = current_investment - initial_investment
        roi = net_profit_step / initial_investment

        if roi >= 0.5:
            return 10
        elif 0.2 <= roi < 0.5:
            return 4
        elif 0.1 <= roi < 0.2:
            return 1
        elif abs(roi) < 1e-6:
            return 0
        elif -0.1 <= roi < 0:
            return -1
        elif -0.2 <= roi < -0.1:
            return -4
        else:
            return -10
