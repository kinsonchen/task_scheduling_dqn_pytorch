import numpy as np

from environment import *

EDGES_NUM = 5
BIRATE_TYPE, RESOLUTION_TYPE = 3, 3


class Stochastic(object):
    def __init__(self, edges_num=EDGES_NUM, b_t=BIRATE_TYPE, r_t=RESOLUTION_TYPE):
        self.edges_num = edges_num
        self.bitrate_type = b_t
        self.resolution_type = r_t

    @staticmethod
    def choose_action():
        bit_rate_i = np.random.randint(0, BIRATE_TYPE)
        resolution_i = np.random.randint(0, RESOLUTION_TYPE)
        edge_i = np.random.randint(0, EDGES_NUM)
        action = [bit_rate_i, resolution_i, edge_i]
        return action


class Greedy(object):
    """
    greedy_traget include: 1.transmission delay 2.calculation time 3.accuracy
    """
    def __init__(self, env):
        self.env = env

    def choose_action(self, state, trans_delay=False, cal_delay=False, accuracy=False):
        # net info
        network = state[NETWORK]

        if trans_delay:     # minimum trans delay
            best_t_d = 999999
            best_e_i = None
            bit_rate, resolution = 3, 720
            data_amount = cal_data_amount(bit_rate, resolution)
            for e_i in range(EDGES_NUM):
                band_width = network[e_i][BAND_WIDTH]
                propagation_daley = network[e_i][PROPAGATION_DELAY]
                tran_delay = cal_tran_time(data_amount, band_width, propagation_daley)
                if tran_delay < best_t_d:
                    best_t_d = tran_delay
                    best_e_i = e_i
            return [0, 0, best_e_i]
        elif cal_delay:     # minimum calculate delay
            best_c_d = 999999
            best_e_i = None
            bit_rate, resolution = 3, 720
            data_amount = cal_data_amount(bit_rate, resolution)
            for e_i in range(EDGES_NUM):
                power = state[EDGE][e_i][COMPUTING_POWER]
                cal_delay = cal_compute_time(data_amount, power)
                if cal_delay < best_c_d:
                    best_c_d = cal_delay
                    best_e_i = e_i
            return [0, 0, best_e_i]
        elif accuracy:      # take maximum config with minimum completion time
            best_t = 999999
            best_e_i = None
            bit_rate, resolution = 5, 1080  # take maximum config
            data_amount = cal_data_amount(bit_rate, resolution)
            for e_i in range(EDGES_NUM):
                band_width = network[e_i][BAND_WIDTH]
                propagation_daley = network[e_i][PROPAGATION_DELAY]
                tran_delay = cal_tran_time(data_amount, band_width, propagation_daley)
                power = state[EDGE][e_i][COMPUTING_POWER]
                cal_delay = cal_compute_time(data_amount, power)
                time = tran_delay + cal_delay
                if time < best_t:
                    best_t = time
                    best_e_i = e_i
            return [2, 2, best_e_i]
        else:
            return Stochastic.choose_action()


if __name__ == '__main__':
    tasks = np.load(r'data\tasks.npy')
    networks = np.load(r'data\networks.npy')
    edges = np.load(r'data\edges.npy')

    env = Env(tasks, networks, edges)
    s = env.init()

    greedy = Greedy(env)
    td_action = greedy.choose_action(s, trans_delay=True)
    cd_action = greedy.choose_action(s, cal_delay=True)
    ac_action = greedy.choose_action(s, accuracy=True)
    print(td_action, cd_action, ac_action)