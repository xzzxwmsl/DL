import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_discount=0.9, e_greedy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_discount = reward_discount
        self.e_greedy = 0.9
        self.Q_table = pd.DataFrame(columns=actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # 有 e_greedy 的概率按照最优解走
        if np.random.uniform() < self.e_greedy:
            # 获取动作列表的Q值
            state_actions = self.Q_table.loc[observation, :]
            # 获取 state_actions 里Q值最大的几个元素的index，并随即选取一个返回(因为可能最大值相同)
            action = np.random.choice(
                state_actions[state_actions == np.max(state_actions)].index)
            # Note：

            # state_actions == np.max(state_actions) 会返回形如：
            # a     True
            # b    False
            # c     True
            # 的真值表

            # state_actions[state_actions == np.max(state_actions)] 会返回形如：
            # a    3
            # c    3
            # 的最大值列表

            # state_actions[state_actions == np.max(state_actions)].index 会返回形如：Index(['a', 'c'], dtype='object') 的index表
        else:
            # 有 1-e_greedy的概率随机选择一个action，这样可以避免陷入局部最优
            action = np.random.choice(self.actions)

        return action

    def learn(self, state, action, reward, state_next):
        self.check_state_exist(state_next)
        q_predict = self.Q_table.loc[state, action]

        if state_next != 'terminal':
            # 不是终结状态，那么实际的q值为进行这个action得到的奖励加上折扣后的未来奖励最大值
            q_target = reward + self.reward_discount * \
                self.Q_table.loc[state_next, :].max()
        else:
            q_target = reward

        # 更新Q_table
        self.Q_table.loc[state, action] += self.learning_rate*(q_target-q_predict)

    def check_state_exist(self, state):
        # 不存在这个状态，所以加入到新的Q表中去
        if state not in self.Q_table.index:
            self.Q_table = self.Q_table.append(
                pd.Series([0]*len(self.actions), index=self.Q_table.columns, name=state))
