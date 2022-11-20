import torch
import numpy as np
import time
import os

from tqdm import tqdm

#自定义文件导入
import env
from env import Creator
from PolicyModelTrain import ASUPG
from network import Policy_net
import bpUtils
from hyperparameter import Hyperparameter

#定义全局变量
hp = Hyperparameter()
creator = Creator()
b_all = 0.6

x12_m0 = []
y12_m0 = []
z12_m0 = []
loss_list12_m0 = []
PolicyNet = Policy_net()
model0 = PolicyNet.policy_net
car_list12, car_list16, car_list20,car_list24 = creator.car_list_create()



def run(items_file,car_list):
    # 初始化
    # 设定一个时间T
    # 模型稳定时间
    steady = 0
    for i in range(1):
        print('第', i + 1, '次：')
        # print('steady:',steady)
        agent = ASUPG(model0,items_file)  # 初始化
        time_start = time.time()
        temp = 0
        flag = 0  # 这个flag是当前的的基准值B与平均的值相等时，停止计算并进行标记
        b_temp = []
        flag_ = 0
        b =0.6 #
        # 这个flag_是一旦达到稳定值时，将当前时间进行记录
        steady_start = 0  # 局部变量之前没有定义
        gini_min = 1


        for episode in range(hp.EPISODE):  # 进行一次
            # initialize task
            #         agent.env.reset() #环境状态初始化
            #         items = data_create() # # 初始化数据
            #         index = [i for i in range(1200)] #  初始化自然装载
            #         index =torch.tensor(index).view(-1,1)
            # #         # print(episode)
            #         b,il = calculate_rate(car_list12, index)

            r, b_, loss = agent.learn(b, car_list)
            # gini_ = bpUtils.computer_gini_on_one_step(agent, car_list, items_file)
            # if gini_ < gini_min:
            #     gini_min = gini_
            b_temp.append(b_)  # 加一个 变量
            b = b_
            if flag_ == 0:
                # print('steady',steady,'b_',b_,'flag_',flag_)
                if steady == float(format(b_.numpy(), '.4f')):
                    steady_start = time.time()
                    # print('收敛时间：', steady_start)
                    flag_ = 1
            # 当第一次稳定值等于基准值时， 进行时间统计，以后将不再进行统计

            if flag == 0:
                if len(b_temp) > 174:
                    v = float(format(b_.numpy(), '.4f'))
                    t = float(format(np.mean(b_temp[-174:]), '.4f'))
                    # print('avg: ',t, ' b:' , v )
                    if t == v:  # 判断收敛

                        time_end = time.time()
                        flag = 1
                        print('The run time is', time_end - time_start)
                        if steady != 0:  # 这里应该进行一个判断，如果第一次的时候，当steady = 0的时候，steady_start的时间是不准确的，他的时间应该是最后的时间，当不为0 的时候，则会进行正确统计
                            # print('steady_start:,',steady_start)
                            print('The convergence time is', steady_start - time_start)
                            print('b:', b_, 'loss:', loss)
                        if steady == 0:  # 如果 初始稳定值为0， 则将第一次的稳定值进行传递， 如果不为0， 则以后稳定都会按照这个稳定时间进行统计
                            steady = t
                        break
                        # print(loss)
            if i == 1:
                x12_m0.append(episode)
                y12_m0.append(r)
                z12_m0.append(b_)
                loss_list12_m0.append(loss)

            if episode % 10 == 0:
               # print('EPI:', episode, 'loss:', loss, 'gini updated', r, 'B:', b_)#'Gini:', gini_
                print('EPI:', episode, 'loss:', loss, 'space ratio updated', r, 'B:', b_)#空间利用率

        item00 = torch.load(items_file)#
        item11 = torch.tensor(item00).detach()
        pro, ind = agent.network(item11)
        ras, ils, pls, rl = bpUtils.calculate_rate_drl_in_detail(car_list, item11, ind, pro)
        sum_pls_num=0
        for i in pls:
            #print('货物数量：',len(i))
            sum_pls_num+=len(i)
        print('需装载的货物总数：',len(item00))
        print('装载的货物总数：',sum_pls_num)
        print('Finally rate:', ras)
        print('rate_list:', np.array(rl), len(rl))


if __name__ == '__main__':
    #     time_start = time.time()
    if not os.path.exists('goods_data_random_profit.pt'):
        creator.data_create_save_random_profit()
    if not os.path.exists('goods_data.pt'):
        creator.data_create_save('goods_data.pt')
    if not os.path.exists('goods_data_1.pt'):
        creator.data_create_save('goods_data_1.pt')
    if not os.path.exists('goods_data_2.pt'):
        creator.data_create_save('goods_data_2.pt')
    if not os.path.exists('goods_data_3.pt'):
        creator.data_create_save('goods_data_3.pt')
    if not os.path.exists('goods_data_4.pt'):
        creator.data_create_save('goods_data_4.pt')
    if not os.path.exists('goods_data_5.pt'):
        creator.data_create_save('goods_data_5.pt')
    if not os.path.exists('goods_data_class_2.pt'):
        creator.data_create_save('goods_data_class_2.pt')
    if not os.path.exists('goods_data_class_3.pt'):
        creator.data_create_save('goods_data_class_3.pt')

    if not os.path.exists('goods_data_long_1.pt'):
        creator.data_create_save('goods_data_long_1.pt')

    run('goods_data_class_3.pt', car_list16)
    print("")


