# Dyna-Q Algorithm: Combining Model-Free Learning with Model-Based Planning

Dyna-Q is an advanced reinforcement learning algorithm that extends conventional Q-Learning with model-based planning capabilities. By integrating real-world and simulated experiences, this algorithm accelerates the learning process and enhances the efficiency of the intelligent agent.

## Core Concept of Dyna-Q

Dyna-Q essentially combines model-free learning (Q-Learning) with model-based planning. While standard Q-Learning only uses the agent's real experiences to update the Q-table, Dyna-Q maintains a model of the environment and uses it to generate simulated experiences. This allows the agent to utilize available data more efficiently and converge to the optimal policy faster[1].

In Dyna-Q, after each real interaction with the environment, multiple simulated experiences are also performed. These simulated experiences are generated using a model built from the environment and are used to further update the Q-table. This enables the agent to learn multiple times from each real experience, accelerating the learning process[2].

This implementation provides a basic framework for using Dyna-Q in reinforcement learning projects. With a deeper understanding of this algorithm and how it works, it can be applied to a wide range of dynamic decision-making problems.

Here is the complete implementation of the **Dyna-Q** algorithm in Python. This implementation includes two main classes: `QLearningTable` for Q-learning and `EnvModel` for modeling the environment and generating simulated experiences.


import numpy as np
import pandas as pd
from copy import deepcopy

class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, agent=""):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.agent = agent
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if self.agent == "RANDOM_AGENT":
            action = np.random.choice(self.actions)
            return action
        
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            max_value = 0
            for act in list(self.q_table.columns.values):
                if self.q_table.ix[observation, act] >= max_value:
                    max_action = act
                    max_value = self.q_table.ix[observation, act]
            action = max_action
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, dn):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        
        if s_ != 'terminal' and dn != True:
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

class EnvModel:
    """Similar to the memory buffer in DQN, you can store past experiences in here.
    Alternatively, the model can generate next state and reward signal accurately."""
    def __init__(self, actions):
        self.actions = actions
        self.database = pd.DataFrame(columns=actions, dtype=np.object)

    def store_transition(self, s, a, r, s_):
        if s not in self.database.index:
            self.database = self.database.append(
                pd.Series(
                    [None] * len(self.actions),
                    index=self.database.columns,
                    name=s,
                ))
        self.database.at[s, a] = deepcopy((r, s_))

    def sample_s_a(self):
        s = np.random.choice(self.database.index)
        a = np.random.choice(self.database.ix[s].dropna().index)  # filter out the None value
        return s, a

    def get_r_s_(self, s, a):
        r, s_ = self.database.ix[s, a]
        return r, s_

    def get_env(self):
        print(self.database)




How to Use the Dyna-Q Algorithm  
Now that we have implemented the required classes, we can use them to build and train a Dyna-Q agent. The following code demonstrates how to use the Dyna-Q algorithm in practice.

from planning_env import Maze  # محیط شبیه‌سازی
from RL_brain import QLearningTable, EnvModel

def update():
    counter = 0
    sum = 0
    for episode in range(2000):
        env.reset()
        print("episode=" + str(episode))
        s_position = [0, 0]
        
        while True:
            a = RL.choose_action(str(s_position))
            s_next_position, r, done, comp_results = env.step(a)
            
            # یادگیری از تجربه واقعی
            RL.learn(str(s_position), a, r, str(s_next_position), done)
            
            # ذخیره تجربه برای استفاده در برنامه‌ریزی
            env_model.store_transition(str(s_position), a, r, s_next_position)
            
            # برنامه‌ریزی: یادگیری از تجربیات شبیه‌سازی شده
            for n in range(10):  # 10 بار برنامه‌ریزی برای هر تجربه واقعی
                ms, ma = env_model.sample_s_a()  # نمونه‌گیری یک وضعیت و عمل
                mr, ms_ = env_model.get_r_s_(ms, ma)  # دریافت پاداش و وضعیت بعدی
                RL.learn(ms, ma, mr, str(ms_), done)  # یادگیری از تجربه شبیه‌سازی شده
            
            s_position = s_next_position.copy()
            
            if done:
                sum = sum + r
                if episode % 50 == 0:
                    output_data.append(sum / 50)
                    sum = 0
                    indexes.append(episode)
                counter = counter + 1
                break
    
    print('episodes over')

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env_model = EnvModel(actions=list(range(env.n_actions)))
    
    output_data = []
    indexes = []
    
    update()
    
    # نمایش نتایج
    env_model.get_env()
    import matplotlib.pyplot as plt
    plt.plot(indexes, output_data, label='RL')
    plt.show()

## Comparison with Conventional Q-Learning

The main advantage of Dyna-Q over conventional Q-Learning is its learning speed. By utilizing planning and simulated experiences, Dyna-Q can achieve the optimal policy with fewer real-world interactions with the environment. This is particularly beneficial in environments where interacting with the environment is costly or time-consuming[1][2].

In practical examples, Dyna-Q can reach the optimal path with only a few episodes of real-world interaction, while conventional Q-Learning may require more episodes. For instance, in a maze problem, Dyna-Q with n=50 planning steps can achieve the optimal path in just 3 episodes[5].

## Advanced Features and Versions of Dyna-Q

A more advanced version of Dyna-Q called Dyna-Q+ is designed for dynamic and changing environments. Dyna-Q+ adds a "curiosity" mechanism to Dyna-Q, allowing the agent to explore states that haven't been visited for a long time. This feature helps the agent react more quickly to changes in dynamic environments.

Some implementations of Dyna-Q also use techniques like "experience replay," similar to the mechanism found in deep reinforcement learning algorithms such as DQN. These techniques can further improve the algorithm's efficiency[2].

## Conclusion

Dyna-Q is a powerful reinforcement learning algorithm that significantly improves learning efficiency by combining model-free learning (Q-Learning) with model-based planning. By using simulated experiences in addition to real experiences, this algorithm can achieve the optimal policy with less data.

The implementation provided in this report offers a basic framework for using Dyna-Q in reinforcement learning projects. With a deeper understanding of this algorithm and how it works, it can be applied to a wide range of dynamic decision-making problems.

http://intelligentonlinetools.com/blog/rl-dyna-q/

---



