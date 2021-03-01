import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['figure.dpi'] = 600    # resolution


class ReplayBuffer(object):
    """docstring for ReplayBuffer"""
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state=[None]):
        # state = np.expand_dims(state, 0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))  # zip(*data)
        # cat_state = [[np.concatenate([[s[i]] for s in state])] for i in range(len(state[0]))]
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


class SaveTool(object):
    def __init__(self):
        self.datas = dict()  # datas={'dic1':[d1, d2], 'dic2':[d1, d2]}

    def save_data_in_dict(self, data, dic=""):
        if dic in self.datas.keys():
            self.datas[dic].append(data)
        else:
            self.datas[dic] = [data]

    def output_npy(self, datas=None, dir="data\\"):
        if not datas:
            datas = self.datas
        for (key, v) in datas.items():
            if not isinstance(v, np.ndarray):
                v = np.array(v)
            np.save(dir+str(key)+".npy", v)
            print(dir+key+".npy \t", v.shape)

    @staticmethod
    def save_data_npy(datas, filename, dir="result\\"):
        if not isinstance(datas, np.ndarray):
            datas = np.array(datas)
        np.save(dir+filename, datas)
        print(dir+filename+".npy", datas.shape)


class Plot(object):
    def __init__(self):
        # history [[r, t, t_l, a, a_d]]
        self.his = []

    def reinit(self):
        self.his = []

    def record_his(self, his):
        #
        his = np.array(his)
        r = np.mean(his[:, 0])
        t, t_limit = np.mean(his[:, 1]), np.mean(his[:, 2])
        a, a_demand = np.mean(his[:, 3]), np.mean(his[:, 4])
        self.his.append([r, t, t_limit, a, a_demand])

    @staticmethod
    def mean_records(data, mean_num=50):
        d = np.array(data).reshape(len(data) // mean_num, mean_num)
        d = [np.mean(i) for i in d]
        return d

    def plot_history(self, title='0'):
        # mean 50 records
        his = np.array(self.his)
        reward_his = Plot.mean_records(his[:, 0])
        time_his = Plot.mean_records(his[:, 1])
        time_limit_his = Plot.mean_records(his[:, 2])
        accuracy_his = Plot.mean_records(his[:, 3])
        accuracy_demand_his = Plot.mean_records(his[:, 4])

        # plt.figure()
        plt.title("Reward Analysis")
        plt.plot(range(len(reward_his)), reward_his)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.savefig(r"fig\Reward Analysis_"+title+".jpg")
        plt.clf()
        # plt.show()

        # plt.figure()
        plt.title("Completion Time Analysis")
        x = range(len(time_his))
        plt.plot(x, time_his, color='green', label='Completion time')
        plt.plot(x, time_limit_his, color='red', label='time limit')
        plt.xlabel('Episode')
        plt.ylabel('Time(s)')
        plt.legend()
        plt.savefig(r"fig\Completion Time Analysis_"+title+".jpg")
        plt.clf()
        # plt.show()

        # plt.figure()
        plt.title("Accuracy Analysis")
        x = range(len(accuracy_his))
        plt.plot(x, accuracy_his, color='green', label='accuracy')
        plt.plot(x, accuracy_demand_his, color='red', label='accuracy demand')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(r"fig\Accuracy Analysis_"+title+".jpg")
        # plt.show()
        plt.clf()

    @staticmethod
    def multi_plot(datas, index=0, episode=0, title="", xlabel="Episode", ylabel="", loc=1):
        SaveTool.save_data_npy(datas, filename="multi_plot"+str(episode))
        his_dqn, his_sto, his_gtd, his_gcd, his_gac = [np.array(d) for d in datas]

        his_dqn = Plot.mean_records(his_dqn[:, index])
        his_sto = Plot.mean_records(his_sto[:, index])
        his_gtd = Plot.mean_records(his_gtd[:, index])
        his_gcd = Plot.mean_records(his_gcd[:, index])
        his_gac = Plot.mean_records(his_gac[:, index])

        # plt.title(title+" Analysis")
        x = range(0, len(his_dqn)*50, 50)
        plt.plot(x, his_dqn, 'g^-', label='HDQN')
        plt.plot(x, his_sto, 'rD-', label='Stochastic')
        plt.plot(x, his_gtd, 'y+-', label='Minimum trans delay')
        plt.plot(x, his_gcd, 'bx-', label='Minimum calcul delay')
        plt.plot(x, his_gac, 'mp-', label='Maximum accuracy')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=loc)
        plt.savefig("fig\\" + title + str(episode) + ".jpg")
        plt.clf()

    @staticmethod
    def plot_twinx(x, y1, y2, x_label, y1_label, y2_label, episode=0, tittle=None, need_mean=True):
        SaveTool.save_data_npy([x, y1, y2], filename="multi_plot" + str(episode))
        if need_mean:
            y1 = Plot.mean_records(y1)
            y2 = Plot.mean_records(y2)
        x = range(0, len(y1) * 50, 50) if not x else x
        fig = plt.figure()
        if tittle:
            plt.title(tittle)
        plt.xlabel(x_label)
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, 'r^-', label="Delay")
        ax1.legend(loc=2)
        ax1.set_ylabel(y1_label)
        ax2 = ax1.twinx()
        ax2.plot(x, y2, 'g*--', label="Accuracy")
        ax2.legend(loc=1)
        ax2.set_ylabel(y2_label)
        ax = plt.gca()
        ax.spines['left'].set_color('r')
        ax.spines['right'].set_color('g')
        plt.savefig('fig\\trade_off' + str(episode) + '.png')
        plt.clf()


def accuracy_with_diff_edges_num():
    delay, accuracy = [], []
    for i in [1, 3]:
        for month_i in range(5):
            d_10_10 = np.mean(np.load("data\\10_10_all.npy")[0][month_i][:, i])
            d_10_15 = np.mean(np.load("data\\10_15_all.npy")[0][month_i][:, i])
            d_10_20 = np.mean(np.load("data\\10_20_all.npy")[0][month_i][:, i])
            d_10_25 = np.mean(np.load("data\\10_25_all.npy")[0][month_i][:, i])
            d_10_30 = np.mean(np.load("data\\10_30_all.npy")[0][month_i][:, i])
            d_10_35 = np.mean(np.load("data\\10_35_all.npy")[0][month_i][:, i])
            d_10_40 = np.mean(np.load("data\\10_40_all.npy")[0][month_i][:, i])
            if i == 1:
                delay.append([d_10_10, d_10_15, d_10_20, d_10_25, d_10_30, d_10_35, d_10_40])
            elif i == 3:
                accuracy.append([d_10_10, d_10_15, d_10_20, d_10_25, d_10_30, d_10_35, d_10_40])

    x = [10, 15, 20, 25, 30, 35, 40]
    plt.plot(x, accuracy[0], 'g^-', label='HDQN')
    plt.plot(x, accuracy[1], 'rD-', label='Stochastic')
    plt.plot(x, accuracy[2], 'y+-', label='Minimum delay')
    # plt.plot(x, accuracy[3], color='blue', label='Minimum calcul delay')
    plt.plot(x, accuracy[4], 'mp-', label='Maximum accuracy')
    plt.xlabel('Number of edge servers')
    plt.ylabel('Accuracy')
    plt.legend(loc=1)
    plt.savefig("fig\\" + 'accuracy_with_diff_edges_num' + str(2000) + ".jpg")
    plt.clf()

    x = [10, 15, 20, 25, 30, 35, 40]
    plt.plot(x, np.flip(delay[0])/10, 'g^-', label='HDQN')
    plt.plot(x, np.flip(delay[1])/10, 'rD-', label='Stochastic')
    plt.plot(x, np.flip(delay[2])/10, 'y+-', label='Minimum delay')
    plt.plot(x, np.flip(delay[4])/10, 'mp-', label='Maximum accuracy')
    plt.xlabel('Number of edge servers')
    plt.ylabel('Delay(s)')
    plt.legend(loc=1)
    plt.savefig("fig\\" + 'delay_with_diff_edges_num' + str(2000) + ".jpg")
    plt.clf()


def accuracy_with_diff_tasks_num():
    delay, accuracy = [], []
    for i in [1, 3]:
        for month_i in range(5):
            t_5_30_all = np.mean(np.load("data\\t_5_30_all.npy")[0][month_i][:, i])
            t_10_30_all = np.mean(np.load("data\\t_10_30_all.npy")[0][month_i][:, i])
            t_15_30_all = np.mean(np.load("data\\t_15_30_all.npy")[0][month_i][:, i])
            t_20_30_all = np.mean(np.load("data\\t_20_30_all.npy")[0][month_i][:, i])
            t_25_30_all = np.mean(np.load("data\\t_25_30_all.npy")[0][month_i][:, i])
            t_30_30_all = np.mean(np.load("data\\t_30_30_all.npy")[0][month_i][:, i])
            if i == 1:
                delay.append([t_5_30_all, t_10_30_all, t_15_30_all, t_20_30_all, t_25_30_all, t_30_30_all])
            elif i == 3:
                accuracy.append([t_5_30_all, t_10_30_all, t_15_30_all, t_20_30_all, t_25_30_all, t_30_30_all])

    x = [5, 10, 15, 20, 25, 30]
    plt.plot(x, accuracy[0], 'g^-', label='HDQN')
    plt.plot(x, accuracy[1], 'rD-', label='Stochastic')
    plt.plot(x, accuracy[2], 'y+-', label='Minimum delay')
    # plt.plot(x, accuracy[3], color='blue', label='Minimum calcul delay')
    plt.plot(x, accuracy[4], 'mp-', label='Maximum accuracy')
    plt.xlabel('Number of tasks')
    plt.ylabel('Accuracy')
    plt.legend(loc=1)
    plt.savefig("fig\\" + 'accuracy_with_diff_task_num' + str(2000) + ".jpg")
    plt.clf()

    x = [5, 10, 15, 20, 25, 30]
    plt.plot(x, delay[0], 'g^-', label='HDQN')
    plt.plot(x, delay[1], 'rD-', label='Stochastic')
    plt.plot(x, delay[2], 'y+-', label='Minimum delay')
    plt.plot(x, delay[4], 'mp-', label='Maximum accuracy')
    plt.xlabel('Number of tasks')
    plt.ylabel('Delay')
    plt.legend(loc=1)
    plt.savefig("fig\\" + 'delay_with_diff_task_num' + str(2000) + ".jpg")
    plt.clf()
