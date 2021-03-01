import numpy as np
import pandas as pd
from configuration import *

np.random.seed(666)


def gen_task_info(tasks_num, tl_range=[1, 5], ad_range=[0.65, 0.8]):
    time_limit = np.random.uniform(tl_range[0], tl_range[1], size=tasks_num).astype(np.float32)
    accuracy_demand = np.random.uniform(ad_range[0], ad_range[1], size=tasks_num).astype(np.float32)
    tasks = [list(t) for t in zip(time_limit, accuracy_demand)]
    return tasks


def gen_network_info(tasks_num, edges_num, bw_range=[10, 30], p_range=[0.1, 0.5]):
    networks = []
    for i in range(tasks_num):
        band_width = np.random.uniform(bw_range[0], bw_range[1], size=edges_num).astype(np.float32)
        propagation_delay = np.random.uniform(p_range[0], p_range[1], size=edges_num).astype(np.float32)
        networks.append([list(t) for t in zip(band_width, propagation_delay)])
    return networks


def gen_edge_info(edges_num, cp_range=[10, 25], s_range=[10, 20]):
    computing_power = np.random.uniform(cp_range[0], cp_range[1], size=edges_num).astype(np.float64)
    # computing_power = torch.normal(15, 5, size=(1, edges_num)).data.numpy()[0]
    storage = np.random.uniform(s_range[0], s_range[1], size=edges_num).astype(np.float64)
    edges = [list(e) for e in zip(computing_power, storage)]
    return edges


def gen_data_from_excel(file):
    df = pd.read_excel(file)
    datas = np.array(df.values)
    networks = [[d[2], d[3]] for d in datas]
    edges = [[d[2], d[3]] for d in datas]
    return networks, edges


if __name__ == '__main__':
    tasks_num, edges_name = TASKS_NUM, EDGES_NUM
    batch_num = 100
    tasks = [gen_task_info(tasks_num) for i in range(batch_num)]
    networks = [gen_network_info(tasks_num, edges_name) for i in range(batch_num)]
    edges = [gen_edge_info(edges_name) for i in range(batch_num)]
    # get case data
    # net, edge = gen_data_from_excel('data\\input_data.xlsx')
    # networks = [[net for i in range(tasks_num)] for i in range(batch_num)]
    # edges = [edge for i in range(batch_num)]

    np.save(r'data\tasks.npy', np.array(tasks))
    np.save(r'data\networks.npy', np.array(networks))
    np.save(r'data\edges.npy', np.array(edges))

    # tasks = np.load(r'data\tasks.npy')
    # networks = np.load(r'data\networks.npy')
    # edges = np.load(r'data\edges.npy')
    #
    # print("Complete generate data !")
    # print("Tasks:", tasks.shape)
    # print("Networks:", networks.shape)
    # print("Edges:", edges.shape)
