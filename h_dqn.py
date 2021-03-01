import numpy as np
import torch
import time

from environment import Env
from module_nn import ACNet, TSNet
from dqn import DQN, BATCH_SIZE
from configuration import *


class HDQN(object):
    def __init__(self):
        self.ac = DQN(ACNet)
        self.ts = DQN(TSNet)
        self.ac_agent, self.ts_agent = self.ac.eval_net, self.ts.eval_net

    def learn(self, env, max_episodes=MAX_EPISODES):
        for episode in range(0, max_episodes):
            print(episode)
            state = env.init()
            done = False
            while not done:
                goal = self.ac_agent.choose_goal(state)
                # low-level learn action according to sub-goal
                best_action, r_in_his, train_limit = [], [], 500
                while len(r_in_his) < train_limit:
                    action = self.ts_agent.choose_action(state, [goal])
                    r_in = Env.intrinsic_reward(state, goal, action)
                    if r_in_his == [] or r_in > max(r_in_his): best_action = action
                    r_in_his.append(r_in)
                    self.ts.memory.push(state, action, r_in)
                    self.ts.train()
                # high-level learn sub-goal
                action = best_action
                next_state, reward, done, env_state = env.step(state, action)
                print(reward, env_state)
                if not done:
                    self.ac.memory.push(state, action, reward, next_state)
                    self.ac.train()
                state = next_state
            # anneal exploration probability
            self.ac.anneal_epsilon()
            self.ts.anneal_epsilon()
        # save model
        self.save_model()

    def save_model(self, ac_path="model/ac_agent", ts_path="model/ts_agent"):
        torch.save(self.ac_agent, ac_path)
        torch.save(self.ts_agent, ts_path)

    def load_model(self, ac_path="model/ac_agent", ts_path="model/ts_agent"):
        self.ac_agent = torch.load(ac_path)
        self.ts_agent = torch.load(ts_path)


if __name__ == '__main__':
    t1 = time.time()
    tasks = np.load(r'data\tasks.npy')
    networks = np.load(r'data\networks.npy')
    edges = np.load(r'data\edges.npy')
    env = Env(tasks, networks, edges)
    hdqn = HDQN()
    hdqn.learn(env)
    print(time.time()-t1)
