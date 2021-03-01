import torch
import torch.nn as nn

from configuration import *
from environment import *
from tools import ReplayBuffer, Plot
from module_nn import ACNet, TSNet, state2tensor


class DQN(object):
    def __init__(self, nn_module):
        self.eval_net, self.target_net = nn_module(), nn_module()
        self.eval_net.initialize_weights()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.consider_tasks, self.edges_num = CONSIDER_TASKS, EDGES_NUM
        self.bitrate_type, self.resolution_type = BIRATE_TYPE, RESOLUTION_TYPE

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0     # for storing memory
        self.memory_size = MEMORY_SIZE
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def soft_update(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        return

    def anneal_epsilon(self, anneal=0.999):
        if self.eval_net.epsilon <= 0.1: return
        self.eval_net.epsilon = self.eval_net.epsilon * anneal


    def train(self):
        self.learn_step_counter += 1
        if self.learn_step_counter < BATCH_SIZE or self.learn_step_counter % 5 != 0: return
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # batch sample
        b_s, b_a, b_r, b_s_ = self.memory.sample(BATCH_SIZE)
        b_s = state2tensor(b_s, is_batch=True)
        b_a = [[bb[0], bb[1], bb[2]] for bb in b_a]     # Adjust the subscript

        # q_eval w.r.t action in sample
        g = torch.from_numpy(np.array(b_a)[:, 0:2])
        self.eval_net.train()
        if type(self.eval_net) == TSNet:
            q_eval = self.eval_net(b_s, g)     # q_eval.shape (batch, data)
            b_a = [bb[-1:] for bb in b_a]
        else:
            q_eval = self.eval_net(b_s)
            b_a = [bb[:2] for bb in b_a]
        q_eval_wrt_a = torch.gather(q_eval, 1, index=torch.LongTensor(np.array(b_a)))
        q_eval_wrt_a = q_eval_wrt_a.sum(dim=1).unsqueeze(0).t()

        # q_target with the maximum q of next_state
        if type(self.eval_net) == TSNet:
            q_target = torch.FloatTensor(list(b_r)).unsqueeze(0).t()    #
        else:
            b_s_ = state2tensor(b_s_, is_batch=True)
            q_next = self.target_net(b_s_).detach()     # detach() from graph, don't back propagate
            max_q_next = torch.cat([q_next[:, 0: BIRATE_TYPE].max(1)[0],
                                    q_next[:, -EDGES_NUM:].max(1)[0]], dim=0).reshape(2, BATCH_SIZE).t()
            max_q_next = max_q_next.sum(dim=1).unsqueeze(0).t()
            b_r = torch.FloatTensor(list(b_r)).unsqueeze(0).t()
            q_target = b_r + DISCOUNT * max_q_next

        # MSELoss
        loss = self.loss_func(q_eval_wrt_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
