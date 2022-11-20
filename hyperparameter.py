# Hyper Parameters 超参数
class Hyperparameter:
    def __init__(self):
        self.TOTAL_ITEMS_SUM = 1200
        #数据文件名
        self.ITEMS_FILE_RANDOM_PROFIT = 'goods_data_random_profit.pt'
        self.ITEMS_FILE_PROFIT = 'goods_data.pt'
        self.asu_b = 0.7
        self.gini_b = 0.25


        self.GAMMA = 0.99  # discount factor
        self.LR = 0.01  # learning rate
        self.EPISODE = 2000  # Episode limitation
        self.STEP = 300  # Step limitation in an episode
        self.TEST = 10  # The number of experiment test every 100 episode
        # Q 训练过程参数，部分参数与Policy过程中的参数值相等
        self.EPSILON = 0.01  #贪心算法的动作取样概率
        self.Q_EPISODE = 500 # Q循环次数，即取到的  迹  的条数
        self.TARGET_UPDATE = 10  #Q网络的目标网络间隔Q网络多少次后进行更新的次数
        self.REPLAY_BUFFER_MAXSIZE = 10000 #经验回放池队列大小
        self.REPLAY_BUFFER_SAMPLE_BATCHSIZE = 5
        self.REPLAY_BUFFER_SAMPLE_MINIMALSIZE = 20
        #货箱数量的上限
        self.BinsMunUpperLimit = 8000