import numpy as np
import torch
import torch.nn as nn

from configuration import *

CONV1D_KERNEL = 2
CONV1D_OUT = 5
POOL1D_KERNEL = 2
FU_FEATURES = int(EDGES_NUM*1.5)
AC_EPSILON, TS_EPSILON = 0.8, 0.8   # epsilon-greedy to select action


def state2tensor(state, is_batch=False):
    # data pre-process
    if is_batch:   # if batch_sample
        # Handle "IndexError: too many indices for array"
        tl, ad, bw, pd, cp, sc = [], [], [], [], [], []
        for ss in state:
            tl.append(ss[TASK][:, TIME_LIMIT])
            ad.append(ss[TASK][:, ACCURACY_DEMAND])
            bw.append(ss[NETWORK][:][:, BAND_WIDTH])
            pd.append(ss[NETWORK][:][:, PROPAGATION_DELAY])
            cp.append(ss[EDGE][:, COMPUTING_POWER])
            sc.append(ss[EDGE][:, STORAGE])
        tl, ad = torch.FloatTensor(tl).unsqueeze(1), torch.FloatTensor(ad).unsqueeze(1)
        bw, pd = torch.FloatTensor(bw), torch.FloatTensor(pd)
        cp, sc = torch.FloatTensor(cp), torch.FloatTensor(sc)
        return [tl, ad, bw, pd, cp, sc]
    tl, ad = state[TASK][:, TIME_LIMIT], state[TASK][:, ACCURACY_DEMAND]
    bw, pd = state[NETWORK][:][:, BAND_WIDTH], state[NETWORK][:][:, PROPAGATION_DELAY]
    cp, sc = state[EDGE][:, COMPUTING_POWER], state[EDGE][:, STORAGE]
    # expand the batch dimension like (data) -> (batch, data)
    tl, ad = torch.FloatTensor(tl).unsqueeze(0).unsqueeze(0), torch.FloatTensor(ad).unsqueeze(0).unsqueeze(0)
    bw, pd = torch.FloatTensor(bw).unsqueeze(0), torch.FloatTensor(pd).unsqueeze(0)
    cp, sc = torch.FloatTensor(cp).unsqueeze(0), torch.FloatTensor(sc).unsqueeze(0)
    return [tl, ad, bw, pd, cp, sc]


class ACNet(nn.Module):
    """
        High-level Adaptive configuration Net
    """
    def __init__(self):
        super(ACNet, self).__init__()
        conv_features = int((CONSIDER_TASKS-CONV1D_KERNEL+1)*CONV1D_OUT/POOL1D_KERNEL)
        self.conv1d_tl = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=CONV1D_OUT, kernel_size=CONV1D_KERNEL, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(POOL1D_KERNEL),
            nn.BatchNorm1d(conv_features)
        )
        self.conv1d_ac = nn.Sequential(
            nn.Conv1d(1, CONV1D_OUT, CONV1D_KERNEL, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(POOL1D_KERNEL),
            nn.BatchNorm1d(conv_features)
        )
        self.fc_bw = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_pd = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_cp = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_sc = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        fc_in = (CONSIDER_TASKS-CONV1D_KERNEL+1)*CONV1D_OUT/POOL1D_KERNEL*2 + FU_FEATURES*4
        self.fc = nn.Sequential(nn.Linear(int(fc_in), BIRATE_TYPE + RESOLUTION_TYPE), nn.ReLU())
        self.epsilon = AC_EPSILON

    def forward(self, s):
        time_limit, accuracy_demand, band_width, propagation_delay, computing_power, storage_capacity = s
        tl = self.conv1d_tl(time_limit).flatten(1)
        ad = self.conv1d_ac(accuracy_demand).flatten(1)
        bw = self.fc_bw(band_width)
        pd = self.fc_pd(propagation_delay)
        cp = self.fc_cp(computing_power)
        sc = self.fc_sc(storage_capacity)
        inp = torch.cat((tl, ad, bw, pd, cp, sc), 1)
        q_value = self.fc(inp)
        return q_value

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def choose_goal(self, s):
        s = state2tensor(s)
        if np.random.uniform() > self.epsilon:
            self.eval()
            q_value = self.forward(s)
            bit_rate_i = q_value[0, 0:BIRATE_TYPE].max(0)[1].item()
            resolution_i = q_value[0, -RESOLUTION_TYPE:].max(0)[1].item()
        else:
            bit_rate_i = np.random.randint(0, BIRATE_TYPE-1)
            resolution_i = np.random.randint(0, RESOLUTION_TYPE-1)
        return [bit_rate_i, resolution_i]


class TSNet(nn.Module):
    """
        Low-level Task scheduling Net
    """
    def __init__(self, ):
        super(TSNet, self).__init__()
        self.fc_config = nn.Sequential(nn.Linear(BIRATE_TYPE + RESOLUTION_TYPE, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(2))
        self.fc_bw = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_pd = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_cp = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        self.fc_sc = nn.Sequential(nn.Linear(EDGES_NUM, FU_FEATURES), nn.ReLU(), nn.BatchNorm1d(FU_FEATURES))
        fc_in = FU_FEATURES*4 + 2
        self.fc = nn.Sequential(nn.Linear(int(fc_in), EDGES_NUM), nn.ReLU())
        self.epsilon = TS_EPSILON

    def forward(self, s, g):
        time_limit, accuracy_demand, band_width, propagation_delay, computing_power, storage_capacity = s
        bw = self.fc_bw(band_width)
        pd = self.fc_pd(propagation_delay)
        cp = self.fc_cp(computing_power)
        sc = self.fc_sc(storage_capacity)
        inp = torch.cat((g, bw, pd, cp, sc), 1)
        q_value = self.fc(inp)
        return q_value

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def choose_action(self, s, g):
        s = state2tensor(s)
        bit_rate_i, resolution_i = g[0]
        g = torch.IntTensor(g)
        if np.random.uniform() > self.epsilon:
            self.eval()
            q_value = self.forward(s, g)
            bit_rate_i = g[0][0].item()
            resolution_i = g[0][1].item()
            edge_i = q_value[0].max(0)[1].item()
        else:
            edge_i = np.random.randint(0, EDGES_NUM-1)
        return bit_rate_i, resolution_i, edge_i
