import numpy as np

from environment import Env
from dqn import DQN
from baseline import Stochastic, Greedy
from tools import SaveTool, Plot
from h_dqn import HDQN



EPISODE = 5000


if __name__ == '__main__':
    tasks = np.load(r'data\tasks.npy')
    networks = np.load(r'data\networks.npy')
    edges = np.load(r'data\edges.npy')

    env = Env(tasks, networks, edges)
    # dqn = DQN()
    hdqn = HDQN()
    hdqn.load_model("model/ac_agent", "model/ts_agent")
    sto = Stochastic()
    greed = Greedy(env)

    stool = SaveTool()
    plt_hdqn, plt_sto, plt_gtd, plt_gcd, plt_gac = Plot(), Plot(), Plot(), Plot(), Plot()

    step = 0
    for episode in range(1, EPISODE):
        # history [[r, t, t_l, a, a_d]]
        his_hdqn, his_sto = [], []
        his_gtd, his_gcd, his_gac = [], [], []

        state = env.init()
        while True:
            # 1. DQN
            # action = dqn.choose_action(state)   # RL choose action based on state
            # next_state, reward, done, env_data = env.step(state, action)    # RL take action and get next observation and reward
            # dqn.store_transition(state, action, reward, next_state)
            # if dqn.memory_counter > dqn.memory_size and step % 5 == 0:
            #     dqn.learn()

            # HDQN
            goal = hdqn.ac_agent.choose_goal(state)
            action = hdqn.ts_agent.choose_action(state, [goal])
            next_state, reward, done, env_data = env.step(state, action)

            # 2.Stochastic
            a_sto = sto.choose_action()
            r_sto, env_sto = env.step(state, a_sto, is_update=False)
            # 3.Greedy Trans Delay
            a_gtd = greed.choose_action(state, trans_delay=True)
            r_gtd, env_gtd = env.step(state, a_gtd, is_update=False)
            # 4.Greedy Calcul Delay
            a_gcd = greed.choose_action(state, cal_delay=True)
            r_gcd, env_gcd = env.step(state, a_gcd, is_update=False)
            # 5.Greedy Accuracy
            a_gac = greed.choose_action(state, accuracy=True)
            r_gac, env_gac = env.step(state, a_gac, is_update=False)

            # history record
            # his_dqn.append([reward] + env_data)
            his_hdqn.append([reward] + env_data)
            his_sto.append([r_sto] + env_sto)
            his_gtd.append([r_gtd] + env_gtd)
            his_gcd.append([r_gcd] + env_gcd)
            his_gac.append([r_gac] + env_gac)

            if done:
                plt_hdqn.record_his(his_hdqn)
                plt_sto.record_his(his_sto)
                plt_gtd.record_his(his_gtd)
                plt_gcd.record_his(his_gcd)
                plt_gac.record_his(his_gac)
                if episode % 500 == 0 and episode > 0:
                    # fig1. x:episode, y:Reward|Time|Accuracy, compare with baseline
                    datas = [plt_hdqn.his, plt_sto.his, plt_gtd.his, plt_gcd.his, plt_gac.his]
                    Plot.multi_plot(datas, index=0, episode=episode, title="Reward", ylabel="Reward")
                    Plot.multi_plot(datas, index=1, episode=episode, title="Time", ylabel="Delay(s)")
                    Plot.multi_plot(datas, index=3, episode=episode, title="Accuracy", ylabel="Accuracy", loc=4)

                    # fig2. x:episode, left_y:delay, right_y:accuracy
                    his = np.array(plt_hdqn.his)
                    y_time = his[:, 1]
                    y_acc = his[:, 3]
                    Plot.plot_twinx(None, y_time, y_acc, "Episode", "Delay(s)", "Accuracy", episode)

                    # fig3. save datas for compare with diff env, accuracy_with_diff_edges_num() in plot.py
                    # if episode == 2000:
                    #     stool.save_data_in_dict(datas, "all")
                    #     fn = "data\\t_" + str(env.tasks_num)+"_"+str(env.edges_num) + "_"
                    #     stool.output_npy(fn)

                    # # fig5. save datas for compare with diff weight, plot in plot.py
                    if episode == 2000:
                        stool.save_data_in_dict(datas, "all")
                        fn = "data\\w_" + str(4) + "_"
                        stool.output_npy(dir=fn)
                break
            state = next_state
            step += 1