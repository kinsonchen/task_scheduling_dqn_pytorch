import numpy as np
import random
import copy

from configuration import *

cal_data_amount = lambda bit_rate, resolution: resolution * 25 * 5 * 1/8/1024/1024    # 60z*10s/b2mb
cal_tran_time = lambda data_amount, band_width, propagation_daley: data_amount/band_width + propagation_daley
cal_compute_time = lambda data_amout, power: data_amout/power


class Env(object):
    def __init__(self, tasks_set, networks_set, edges_set):
        self.tasks_set = tasks_set.copy()           # tasks[task_index][TIME_LIMIT | ACCURACY_DEMAND]
        self.networks_set = networks_set.copy()     # networks[task_index][edge_index][BAND_WIDTH | PROPAGATION_DELAY]
        self.edges_set = edges_set.copy()           # edges[edge_index][COMPUTING_POWER | STORAGE]

        self.tasks, self.networks, self.edges, self.remain_tasks_index = [], [], [], []
        self.t_i = 0    #
        self.init()

    def init(self):
        """initialize the environment"""
        # sample from data sets
        i = random.randint(0, len(self.tasks_set)-1)
        self.tasks, self.networks, self.edges = self.tasks_set[i], self.networks_set[i].copy(), self.edges_set[i].copy()
        self.remain_tasks_index = list(range(CONSIDER_TASKS-1, len(self.tasks)))
        # randomly choose task
        # t_i = random.sample(self.remain_tasks_index, 1)[0]
        # self.remain_tasks_index.remove(t_i)

        self.t_i = 0
        task_inf = np.array([self.tasks[0], self.tasks[0], self.tasks[0]])
        return [task_inf, self.networks[self.t_i], self.edges]

    def get_consider_tasks(self):
        if self.t_i == len(self.tasks): return None
        if self.t_i == 0: return np.array([self.tasks[0], self.tasks[0], self.tasks[0]])
        if self.t_i == 1: return np.array([self.tasks[0], self.tasks[1], self.tasks[1]])
        return np.array([self.tasks[self.t_i-2], self.tasks[self.t_i-1], self.tasks[self.t_i]])     # CONSIDER_TASKS =3

    @staticmethod
    def cal_accuracy(bit_rate, resolution):
        best_br, best_resolution = 5, 1920*1080
        noise = np.random.uniform(0.01, 0.1)
        return 0.25*(bit_rate / best_br) + 0.6*(resolution / best_resolution) + noise

    @staticmethod
    def state2info(s, a):
        time_limit, accuracy_demand = s[TASK][-1][TIME_LIMIT], s[TASK][-1][ACCURACY_DEMAND]
        edge_index = a[TARGET_EDGE]
        power, storage = s[EDGE][a[TARGET_EDGE]][COMPUTING_POWER], s[EDGE][a[TARGET_EDGE]][STORAGE]
        bandwidth, delay = s[NETWORK][edge_index][BAND_WIDTH], s[NETWORK][edge_index][PROPAGATION_DELAY]
        bitrate, resolution = BITRATES[a[BIT_RATE]], RESOLUTIONS[a[RESOLUTION]]
        return time_limit, accuracy_demand, power, storage, bandwidth, delay, bitrate, resolution

    @staticmethod
    def cal_reward(s, a):
        [time_limit, accuracy_demand, power, storage, bandwidth, delay, bitrate, resolution] = Env.state2info(s, a)
        data_amount = cal_data_amount(bitrate, resolution)
        tran_time = cal_tran_time(data_amount, bandwidth, delay)
        calcul_time = cal_compute_time(data_amount, power)
        times = tran_time + calcul_time
        accuracy = Env.cal_accuracy(bitrate, resolution)

        alpha = 0.7
        beta = 1 - alpha
        reward = alpha * ((time_limit - times) / time_limit) + beta * ((accuracy - accuracy_demand) / accuracy_demand)
        if times < time_limit and accuracy > accuracy_demand: reward += 1
        if reward < -5:
            print(reward)
        return reward, [times, time_limit, accuracy, accuracy_demand]

    def step(self, s, a, is_update=True):
        [time_limit, accuracy_demand, power, storage, bandwidth, delay, bitrate, resolution] = Env.state2info(s, a)
        reward, [times, time_limit, accuracy, accuracy_demand] = Env.cal_reward(s, a)

        # if update env to get next_state in rl
        if is_update:
            # nor_use_bw = nor(bit_rate, self.max_n[BAND_WIDTH], self.min_n[BAND_WIDTH])
            for n in self.networks:
                for d in n:
                    d[BAND_WIDTH] -= bitrate
                    if d[BAND_WIDTH] < 0:
                        d[BAND_WIDTH] = 0.01
            # choose next_task from remain unprocessed task
            self.t_i += 1
            if self.t_i == len(self.tasks):
                return [], reward, True, [times, time_limit, accuracy, accuracy_demand]
            next_task = self.get_consider_tasks()
            next_state = [next_task, self.networks[self.t_i], self.edges]
            return next_state, reward, False, [times, time_limit, accuracy, accuracy_demand]
        else:
            return reward, [times, time_limit, accuracy, accuracy_demand]

    @staticmethod
    def intrinsic_reward(s, g, a):
        #
        [time_limit, accuracy_demand, power, storage, bandwidth, delay, bitrate, resolution] = Env.state2info(s, a)
        data_amount = cal_data_amount(bitrate, resolution)
        tran_time = cal_tran_time(data_amount, bandwidth, delay)
        calcul_time = cal_compute_time(data_amount, power)
        times = tran_time + calcul_time
        accuracy = Env.cal_accuracy(bitrate, resolution)

        alpha = 0.8
        beta = 1 - alpha
        reward = alpha * ((time_limit - times) / time_limit) + beta * ((accuracy - accuracy_demand) / accuracy_demand)
        return reward


if __name__ == '__main__':
    tasks = np.load(r'data\tasks.npy')
    networks = np.load(r'data\networks.npy')
    edges = np.load(r'data\edges.npy')
    env = Env(tasks, networks, edges)