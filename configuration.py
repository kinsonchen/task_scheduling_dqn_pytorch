TASKS_NUM, EDGES_NUM = 5, 10
MAX_EPISODES = 2000

CONSIDER_TASKS = 3
BIRATE_TYPE, RESOLUTION_TYPE = 3, 3
BITRATES = [3, 3.5, 5]                          # (Mb/s)
RESOLUTIONS = [1280*720, 1280*960, 1920*1080]   # (p)


LR = 0.001                      # learning rate
DISCOUNT = 0.9                  # reward discount
TARGET_REPLACE_ITER = 100       # target update frequency
BATCH_SIZE = 20
MEMORY_SIZE = 5000              # ER memory size

''' 
Matrix :
state[TASK][task_index][TIME_LIMIT | ACCURACY_DEMAND]
     [NETWORK][edge_index][BAND_WIDTH | PROPAGATION_DELAY]
     [EDGE][edge_index][COMPUTING_POWER | STORAGE]
action[BIT_RATE | RESOLUTION | TARGET_EDGE] = [br_index, re_index, edge_index]
'''
# to confirm the matrix
TASK, NETWORK, EDGE = 0, 1, 2
TIME_LIMIT,  ACCURACY_DEMAND = 0, 1
BAND_WIDTH, PROPAGATION_DELAY = 0, 1
COMPUTING_POWER, STORAGE = 0, 1
BIT_RATE, RESOLUTION, TARGET_EDGE = 0, 1, 2

