import torch
import numpy as np
import pandas as pd
#自定义文件导入
from env import Creator, Packer
from hyperparameter import Hyperparameter
import bpUtils

#全局变量 超参
hp = Hyperparameter() #因为是类，所以需要创建实例
creator = Creator() #不用初始化,只调用方法，属于方法工具类，因为是类，所以需要创建实例
#----------------------------------  优化基尼系数的智能体策略梯度更新方法  ----------------------
class ASUPG():
    # DQN Agent
    def __init__(self, model, items_file):  # 初始化
        # 状态空间和动作空间的维度
        #         self.state_dim = env.observation_space.shape[0]
        #         self.action_dim = env.action_space.n
        # Packer().packer_own()
        self.b = 0.6
        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_po, self.ep_perpo = [], [], [], [], []
        # init network parameters
        self.network = model
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hp.LR)
        # require index, prob
        itemz = creator.data_create()  # 必须重新生成数据 否则会报 index  out of bound self的bug
        #
        item0 = itemz
        item1 = itemz
        #         print(itemz,item0,item1)
        self.prob, self.index = 0, 0
        self.env = 0
        # init some parameters
        self.time_step = 0

        self.item0 = torch.load(items_file)  # data_create()

    def choose_action(self, observation):
        # observation = torch.IntTensor(observation)
        #         network_output = self.network.forward(observation)
        #         with torch.no_grad():
        #             prob_weights = F.softmax(network_output, dim=0).numpy()
        #         # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        #         action = np.random.choice(range(prob_weights.shape[0]),
        #                                   p=prob_weights)  # select action w.r.t the actions prob
        action = self.env.Index_list[observation]  # 选取对应状态的序列作为动作
        return action

    #     # 将状态，动作，奖励这一个transition保存到三个列表中
    #     def store_transition(self, s, a, r): # 存储当前的动作，状态，奖励
    #         self.ep_obs.append(s) # 状态 i:1,2,3,4,5,6,7.....
    #         self.ep_as.append(a) # 动作 Action:当前的序列
    #         self.ep_rs.append(r) # 回报 R：当前的体积


    def learn(self, b, carlist):
        # b 初始状态下装载率，基值
        # self.time_step += 1
        if self.b == 0:
            self.b = b
        else:
            pass

        # Step 1: 计算每一步的状态价值
        #         discounted_ep_rs = np.zeros_like(self.ep_rs)
        #         running_add = 0
        #         # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        #         # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        #         for t in reversed(range(0, len(self.ep_rs))):
        #             running_add = running_add * GAMMA + self.ep_rs[t]
        #             discounted_ep_rs[t] = running_add
        #         print(discounted_ep_rs)
        #         discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)  # 减均值
        #         discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)  # 除以标准差
        #         discounted_ep_rs = torch.FloatTensor(discounted_ep_rs)
        #         print(discounted_ep_rs)
        # Step 2: 前向传播

        item0 = torch.tensor(self.item0).clone().detach()
        # print(item0)
        # item0 = item0.to(device)
        #         item0.requires_grad_(True)
        item0 = item0.requires_grad_(True)
        # it = item0.int()
        pro, ind = self.network.forward(item0)# 结果为Tensor
        #---------------- self.itemo按照序列重排序 ----------------  实验已验证，效果并不好
        #按照ind将itemo重排序
        #v_ind=ind.squeeze(1)
        #items_list = item0.index_select(dim=0,index=v_ind)
        #self.item0 = items_list
        #---------------- self.itemo按照序列重排序 完 ----------------
        r, ils, pls, rl = bpUtils.calculate_rate_drl_in_detail(carlist,item0.requires_grad_(False), ind, pro)
        # print(r)
        # ils 物品列表
        # pls 装载物品对应的概率
        # self.run(prob,i,car_list12)
        rs_list = bpUtils.il2value(ils,carlist)  # 列表转化价值表
        #***********基尼系数**********
        #对每个箱子里的货物的价值求和，变成一个一维列表

        #         print('rslist',rs_list)
        dis_list = []
        for i in rs_list:
            discounted_ep_rs = np.zeros_like(i)
            running_add = 0
            for t in reversed(range(0, len(i))):
                running_add = running_add * hp.GAMMA + i[t]#*(1/len(rs_list))
                discounted_ep_rs[t] = running_add
            discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)  # 减均值
            discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)  # 除以标准差
            discounted_ep_rs = torch.FloatTensor(discounted_ep_rs)
            dis_list.append(discounted_ep_rs)
        s2 = []
        s3 = []
        for i in range(len(dis_list)):
            s1 = []
            for j in range(len(dis_list[i])):
                s = (dis_list[i][j]) * torch.log(torch.tensor(pls[i][j]))#+ disincome_list[i][j] disnewreward_list #原版 dis_list
                s1.append(s)
            s2.append(s1)
            s3.append(
                torch.sum(torch.tensor(s2[i])) * (self.b - torch.sum(torch.tensor(rs_list[i])) ))#new_reward 原：rs_list #r_new_reward:原new_reward
        loss = torch.mean(torch.tensor(s3))
        self.optimizer.zero_grad()
        loss.requires_grad_(True)  # element 0 of tensors does not require grad and does not have a grad_fn
        loss.backward()
        self.optimizer.step()
        self.b = self.b + 0.02 * (r - self.b)
        # 每次学习完后清空数组
        # print(self.b)
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_perpo, self.ep_po = [], [], [], [], []
        return r, self.b, loss





