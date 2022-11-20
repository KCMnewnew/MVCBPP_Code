import torch
import numpy as np
import pandas as pd
import collections
import random
#自定义文件导入
from env import Creator, Packer
from hyperparameter import Hyperparameter
#创建全局变量
creator = Creator() #Creator不用初始化
hy = Hyperparameter()
def trapezoidalIntegration(tens):
    v_sort = torch.sort(tens).values
    m1=0
    m2=0
    all_s=torch.sum(v_sort)
    for i in range(len(v_sort)):
        aaa=0
        for j in range(0,i):
            aaa=aaa+v_sort[j]
        m2=aaa/all_s
        if i == 0 or  i==len(v_sort)-1:
            m1=m1+m2/2
        m1=m1+m2
    value=m1/len(v_sort)
    return value
def gini_computer(wls):
    weights_list_binsnumber = []
    for i in wls:
        sum_w = 0
        for j in i:
            sum_w = sum_w + j
        weights_list_binsnumber.append(sum_w)
    wl_tensor = torch.tensor(weights_list_binsnumber)
    # 积分
    trapezoidal_B = trapezoidalIntegration(wl_tensor)
    gini = 1 - 2 * trapezoidal_B
    return gini

def save_loss(loss_value,csv_name):
    loss_ = []
    x = []
    for i in range(len(loss_value)):
        x.append(i)
        loss_.append(loss_value[i].detach().numpy())
    data = np.vstack((x, loss_)).T
    column_name=['Episode','Loss']
    dff = pd.DataFrame(data, columns=column_name)
    dff.to_csv(csv_name)
def save_model_parameter_loss(model_name,loss_value,csv_name,plk_name,baseline_value,baseline_csv_name):
    torch.save(model_name.state_dict(),plk_name)
    save_loss(loss_value,csv_name) #loss_12_m0-m5,存储文件名
    save_loss(baseline_value,baseline_csv_name)
def getMaxMin_2dim_list(weight_list):
    i_sum=[]
    for i in range(len(weight_list)):
        i_sum.append(np.sum(np.array(weight_list[i])))
    nd_i_sum=np.array(i_sum)
    return np.max(nd_i_sum),np.min(nd_i_sum)

def computer_gini_on_one_step(agent, car_list, items_file):
    item_temp = torch.load(items_file)  #
    item_temp = torch.tensor(item_temp).clone().detach()
    pro_, ind_ = agent.network(item_temp)
    ras_, ils_, pls_, rl_, wls_ = calculate_rate_drl_in_detail(car_list, item_temp, ind_,pro_)
    return gini_computer(wls_)

def calculate_rate_drl_in_detail(carlist,items, index, prob):
    i = index
    rate_list = []
    items = items # 1200,1,3
    items = items.squeeze()
    num_list = []
    temp = 0
    items_list = []
    prob_list = []
    p_list = []
    p = list(prob)
    load_items = creator.bin_create(items, i) # 装箱
    # print(len(load_items))
    for i in range(8000):
        if len(rate_list) == 0:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            temp = num
            # print(num,len(load_items))
            rate_list.append(rate)
            num_list.append(num)
            items_list.append(item_list)

            for k in range(num):
                prob_list.append(p[k])
            for k in range(num):
                p.pop(0)
#             for i in range(num):
#                 load_items.pop(0)
        else:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            # print(num,len(load_items))
            if num >= temp*0.1 : # 如果当前的装载数量达到了上次装载的80%
                rate_list.append(rate)
                num_list.append(num)
                items_list.append(item_list)
                temp = num
                for k in range(num):
                    prob_list.append(p[k])
                for k in range(num):
                    p.pop(0)
#                 for i in range(num):
#                     load_items.pop(0)
            else:
                break
        p_list.append(prob_list)
        prob_list = []
    try:
        avg_rate=torch.sum(torch.tensor(rate_list))/len(rate_list)
    except:
        raise ZeroDivisionError
    return avg_rate,items_list,p_list,rate_list

#---------------------------------------------------------------------------------------
def calculate_rate_drl_in_detail(carlist,items, index, prob):
    i = index
    rate_list = []
    items = items # 1200,1,3
    items = items.squeeze()
    num_list = []
    temp = 0
    items_list = []
    prob_list = []
    p_list = []
    p = list(prob)
    load_items = creator.bin_create(items, i) # 装箱
    # print(len(load_items))
    load_items_list = load_items.copy() #packer_own函数将在pop时剔除一个货物
    for i in range(hy.BinsMunUpperLimit): #货箱数量的上限
        if len(rate_list) == 0:
            # print(len(load_items))
            rate, num, item_list, peritem_list = Packer().packer_own(carlist, load_items_list)
            temp = num
            #print('已经装了的货物数量：',num,'所有货物数量:',len(load_items))
            rate_list.append(rate)
            num_list.append(num)
            items_list.append(item_list)

            for k in range(num):
                prob_list.append(p[k])
            for k in range(num):
                p.pop(0)
#             for i in range(num):
#                 load_items.pop(0)
        else:
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items_list)
            #print('已经装了的货物数量：',num,'所有货物数量:',len(load_items))
            if num >= temp * 0.1:

                rate_list.append(rate)
                num_list.append(num)
                items_list.append(item_list)
                temp = num
                for k in range(num):
                    prob_list.append(p[k])
                for k in range(num):
                    p.pop(0)

            else:
                break
        p_list.append(prob_list)
        prob_list = []
    try:
        avg_rate=torch.sum(torch.tensor(rate_list))/len(rate_list)
    except:
        raise ZeroDivisionError

    return avg_rate,items_list, p_list,rate_list



#**********************************************************************************




















def calculate_rate_drl_by_gini_in_detail(carlist,items, index, prob):
    i = index
    rate_list = []
    items = items # 1200,1,3
    items = items.squeeze()
    num_list = []
    temp = 0
    items_list = []
    prob_list = []
    p_list = []
    weights_list=[]
    p = list(prob)
    load_items = creator.bin_create(items, i) # 装箱
    # print(len(load_items))
    for i in range(80):
        if len(rate_list) == 0:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            temp = num
            # print(num,len(load_items))
            rate_list.append(rate)
            num_list.append(num)
            items_list.append(item_list)
            weight_list=[]
            for i in item_list:
                weight_list.append(i.weight)
            weights_list.append(weight_list)
            for k in range(num):
                prob_list.append(p[k])
            for k in range(num):
                p.pop(0)
#             for i in range(num):
#                 load_items.pop(0)
        else:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            print('已经装了的货物数量：',num,'所有货物数量:',len(load_items))
            if num >= temp*0.1 : # 如果当前的装载数量达到了上次装载的80%
                rate_list.append(rate)
                num_list.append(num)
                items_list.append(item_list)
                weight_list = []
                for i in item_list:
                    weight_list.append(i.weight)

                weights_list.append(weight_list)
                temp = num
                for k in range(num):
                    prob_list.append(p[k])
                for k in range(num):
                    p.pop(0)
#                 for i in range(num):
#                     load_items.pop(0)
            else:
                break
        p_list.append(prob_list)
        prob_list = []
    try:
        avg_rate=torch.sum(torch.tensor(rate_list))/len(rate_list)
    except:
        raise ZeroDivisionError
    return avg_rate,items_list,p_list,rate_list,weights_list

def calculate_rate_drl_in_detail_size(carlist,items, index, prob):
    i = index
    rate_list = []
    items = items # 1200,1,3
    items = items.squeeze()
    num_list = []
    temp = 0
    items_list = []
    prob_list = []
    p_list = []
    weights_list=[]
    p = list(prob)
    load_items = creator.bin_create(items, i) # 装箱
    # print(len(load_items))
    for i in range(80):
        if len(rate_list) == 0:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            temp = num
            # print(num,len(load_items))
            rate_list.append(rate)
            num_list.append(num)
            items_list.append(item_list)
            weight_list=[]
            for i in item_list:
                weight_list.append(i.weight)
            weights_list.append(weight_list)
            for k in range(num):
                prob_list.append(p[k])
            for k in range(num):
                p.pop(0)
#             for i in range(num):
#                 load_items.pop(0)
        else:
            # print(len(load_items))
            rate, num, item_list, peritem_list= Packer().packer_own(carlist, load_items)
            # print(num,len(load_items))
            if num >= temp*0.1 : # 如果当前的装载数量达到了上次装载的80%
                rate_list.append(rate)
                num_list.append(num)
                items_list.append(item_list)
                weight_list = []
                for i in item_list:
                    weight_list.append(i.weight)

                weights_list.append(weight_list)
                temp = num
                for k in range(num):
                    prob_list.append(p[k])
                for k in range(num):
                    p.pop(0)
#                 for i in range(num):
#                     load_items.pop(0)
            else:
                break
        p_list.append(prob_list)
        prob_list = []
    try:
        avg_rate=torch.sum(torch.tensor(rate_list))/len(rate_list)
    except:
        raise ZeroDivisionError
    items_size_list = []
    for i in items_list:
        for j in i:
            items_size_list.append([j.length.item(),j.width.item(),j.height.item(),j.weight.item()])
    return avg_rate,items_list, items_size_list, p_list,rate_list,weights_list

def il2value(ilz,carlist):
    listsa = []
    for i in ilz:
        vol = 0
        vol_list = []
        for j in i:
            vol_list.append(j.volume/ carlist[0].volume)
        listsa.append(vol_list)
    return listsa
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class ReplayBuffer_MORL:
    def __init__(self, capacity):
        self.buffer_1 = collections.deque(maxlen=capacity)
        self.buffer_2 = collections.deque(maxlen=capacity)

    def add_1(self, state, action, reward, next_state, done):
        self.buffer_1.append((state, action, reward, next_state, done))
    def add_2(self, state, action, reward, next_state, done):
        self.buffer_2.append((state, action, reward, next_state, done))
    #从两个里main随机取样才对，而不是分别从两个里面取样
    def sample(self, batch_size):
        batch_size_1 = random.randint(1, batch_size-1)# batch_size里挑出一部分给buffer_1,另一部分名额给buffer_2
        batch_size_2 = batch_size - batch_size_1
        transitions_1 = random.sample(self.buffer_1, batch_size_1)
        state_1, action_1, reward_1, next_state_1, done_1 = zip(*transitions_1)
        transitions_2 = random.sample(self.buffer_2, batch_size_2)
        state_2, action_2, reward_2, next_state_2, done_2 = zip(*transitions_2)
        #将两部分结合成一个
        #print(type(state_2),type(action_2),type(reward_2)) 数据类型: python的元组：tuple
        state = state_1+state_2
        action = action_1 + action_2
        reward = reward_1 + reward_2
        next_state = next_state_1 + next_state_2
        return np.array(state), action, reward, np.array(next_state), True

    def size(self):
        return len(self.buffer_1),len(self.buffer_2)
    def pop(self):
        self.buffer_1.pop()
        self.buffer_2.pop()
#庄家法寻找帕累托最优解的方法,Q是所有解的集合

def Pareto_MakersAMethod(Q_tuple):#Q是Tensor
    #元组转成list
    Q = []
    for i in Q_tuple:
        Q.append(i)
    NDset = []
    while len(Q) > 0:
        x = Q[0]
        #x是其中一个值
        x = Q[0]
        #删掉x
        Q =Q.pop(x)
        #寻找非支配解
        sign = True
        P = Q + NDset
        for i in range(len((P))):
            if x > P[i]:
               Q.remove(P[i])
            elif P[i] > x:
                sign = False
        if sign:
            NDset = NDset + x




#
# def Pareto_PaiChuFaAMethod(Q):
#     NDset = []
#     D = []
#     while len(Q) > 0:
#         #x是其中一个值
#         x = Q[0]
#         #删掉x
#         Q =Q.pop(x)
#         #寻找非支配解
#         sign = True
#         for i in (Q + NDset):
#             if x<i:
#                Q.pop(i)
#             elif i<x:
#                 sign = False
#         if sign:
#             NDset = NDset + x
#         NDset = NDset + Q





