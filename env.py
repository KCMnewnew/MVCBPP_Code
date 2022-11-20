import torch
import numpy as np
import math
from  hyperparameter import Hyperparameter
hp = Hyperparameter()
class Creator:
    # def __init__(self):
    #     pass
    def data_create(self):
        Items = []
        size = hp.TOTAL_ITEMS_SUM

        # np.random.seed(0) # 随机固定
        # 数据生成变成了12000
    # torch.manual_seed(1)
        for i in range(size):#12000

            if i<0.30 * size:#180
                x,y,z = 4.,4.,3.
            elif i<0.60 * size:#360
                x,y,z = 6.,4.,3.
            elif i<  size:#540
                x,y,z = 6.,3.,3.

            Items.append([i, x, y, z])
        Items_noindex = list(Items.copy())
        Items_noindex = np.array(Items_noindex)
        Items_noindex = Items_noindex[:, 1:] # 原1:-1
        Items_noindex = torch.tensor(Items_noindex)
        Items_noindex = Items_noindex.unsqueeze(1)
        return Items_noindex
    #   shape[1200,1,3] batch,emb,length

    def data_create_random_profit(self):
        Items = []
        size = hp.TOTAL_ITEMS_SUM
        for i in range(size): #
            t=np.random.randint(20,150)
            if i< 0.15 * size:
                x,y,z = 4.,4.,3.
            elif i<0.3 * size:
                x,y,z = 6.,4.,3.
            elif i<0.45 * size:
                x,y,z = 6.,3.,3.
            elif i<0.6 * size:
                x,y,z = 6.,3.,2.
            elif i<0.75 * size:
                x,y,z = 4.,3.,2.
            elif i<0.9 * size:
                x,y,z = 4.,3.,3.
            elif i<size:
                x,y,z = 4.,2.,2.
            Items.append([i, x, y, z, t])
        Items_noindex = list(Items.copy())
        Items_noindex = np.array(Items_noindex)
        Items_noindex = Items_noindex[:, 1:] # 原1:-1
        Items_noindex = torch.tensor(Items_noindex)
        Items_noindex = Items_noindex.unsqueeze(1)
        return Items_noindex

    def data_create_save(self,file_name):
        bins=self.data_create()
        torch. save(bins,file_name)
    def data_create_save_random_profit(self):
        bins=self.data_create_random_profit()
        torch. save(bins,'goods_data_random_profit.pt')
    # 只为强化学习算法而生
    def bin_create(self,item_list,index_list): # item 为array,为得出的序列
        item_data = []
        for i in range(len(item_list)):
            index = index_list[i][0]
            item = Bin('0', item_list[index][0], item_list[index][1], item_list[index][2], 0) #货物的0长1宽2高和3价值
            item_data.append(item)
        return item_data
    def car_list_create(self):
        car2 = Car('3D Bin-20', 20, 20, 20, 3000, Attr.Position, 0)
        car0 = Car('3D Bin-12', 12, 12, 12, 3000, Attr.Position, 0)
        car1 = Car('3D Bin-16', 16, 16, 16, 3000, Attr.Position, 0)
        car3 = Car('3D Bin-24', 24, 24, 24, 3000, Attr.Position, 0)
        # car4 = Car('3t货车', 25, 25, 25, 3000, Attr.Position, 0)
        # car5 = Car('3t货车', 26, 26, 26, 3000, Attr.Position, 0)
        # car6 = Car('3t货车', 27, 27, 27, 3000, Attr.Position, 0)
        # car7 = Car('3t货车', 28, 28, 28, 3000, Attr.Position, 0)
        # car8 = Car('3t货车', 29, 29, 29, 3000, Attr.Position, 0)
        # car9 = Car('3t货车', 30, 30, 30, 3000, Attr.Position, 0)
        # car10 = Car('3t货车', 31, 31, 31, 3000, Attr.Position, 0)
        # car11 = Car('3t货车', 32, 32, 32, 3000, Attr.Position, 0)
        car_list20 = [car2]
        car_list12 = [car0]
        car_list16 = [car1]
        car_list24 = [car3]
        return car_list12,car_list16,car_list20,car_list24
        # car_list25 = [car4]
        # car_list26 = [car5]
        # car_list27 = [car6]
        # car_list28 = [car7]
        # carlist29 = [car8]
        # car_list3_0 = [car9]
        # car_list31 = [car10]
        # car_list32 = [car11]
    #************************  此方法需要重写 ******************
    # 功能设想：将多种启发式算法的结果作为列表输入，可以是所有大小类型的货箱的实验结果，得到每种货箱的均值或者中位数作为baseline
    def baseline_b(self,np_num):
        sum = 0
        for i in np_num:
            sum += i
        return float(format(sum / len(np_num), '.7f'))
    # b_12 = 0.6  # baseline_b(bin_12_16_20_24[0,:])
    # b_16 = 0.6  # baseline_b(bin_12_16_20_24[1,:])
    # b_20 = 0.6  # baseline_b(bin_12_16_20_24[2,:])
    # b_24 = 0.6  # baseline_b(bin_12_16_20_24[3,:])


# 生成100个2-4之间的随机长宽高组成的数据

#货箱属性类
# Car: name,length,width,height,Max_weight,position,temp_volume
class Car:
    def __init__(self,name,length,width,height,Max_weight,position,temp_volume):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        # self.weight = weight
        self.volume = length * width * height
        self.Max_weight = Max_weight
        self.position = position
        self.temp_volume = temp_volume
#货物属性类
# Bin: name,length,width,height,weight,pose
class Bin:
    def __init__(self,name,length,width,height,pose):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        #self.weight = weight
        self.volume = length * width * height
        self.pose = pose

    def get_dimension(self):
        if self.pose == Attr.Pose_wh_front:
            d = [self.length, self.width, self.height]  # 就是正面
        elif self.pose == Attr.Pose_hw_front:
            d = [self.length, self.height, self.width]
        elif self.pose == Attr.Pose_dh_front:
            d = [self.width, self.length, self.height]
        elif self.pose == Attr.Pose_hd_front:
            d = [self.width, self.height, self.length]
        elif self.pose == Attr.Pose_wd_front:
            d = [self.height, self.length, self.width]
        elif self.pose == Attr.Pose_dw_front:
            d = [self.height, self.width, self.length]
        else:
            d = []
        return d
#货物朝向类
class Attr:
    # 三个坐标轴
    Axis_x = 0
    Axis_y = 1
    Axis_z = 2
    # 原点坐标
    Position = [0, 0, 0]
    # 六个朝向
    Pose_wh_front = 0
    Pose_hw_front = 1
    Pose_hd_front = 2
    Pose_dh_front = 3
    Pose_wd_front = 4
    Pose_dw_front = 5

# 装载方式变换
class PackingMethod:
    def first_put(self):
        arg = 0
        for i in range(6):
            d = PackingMethod.set_pose(self, i).get_dimension()
            if d[Attr.Axis_x] > d[Attr.Axis_y] and d[Attr.Axis_y] > d[Attr.Axis_z]:
                arg = i
            else:
                pass
        return arg
    # 找到长是最长的 ，宽次要 高最短的那种形状填进去
    # 适合放进去的方式
    def pose_fit_arg(self, temp_L, temp_W, temp_H):  # 当前对象
        arg_ = 10000
        dv_ = 10000
        for i in range(6):
            d = PackingMethod.set_pose(self, i).get_dimension()
            #*******************************************          欧氏距离       ****************************************
            sr = math.sqrt((temp_L - d[Attr.Axis_x]) ** 2 + (temp_W - d[Attr.Axis_y]) ** 2 + (
                        temp_H - d[Attr.Axis_z]) ** 2)  # 我选择的判定方式是平方根
            # print(sr)
            if dv_ > sr:
                dv_ = sr
                arg_ = i
            else:
                continue
        return arg_
        # 返回的是 fanhuideshi

    def set_pose(self, num):
        if num >= 0 and num <= 5:
            self.pose = num
            return self
        else:
            print('数据异常')

class Sort:
    # 排序  车已经从大到小排序完了，这里车辆并没有付费，考虑到成本的话，需要加一下
    def Sort_Car(Car_list):
        Car_list.sort(key=lambda x: x.volume, reverse=True)
        return Car_list
    #货物按照体积大小排序，从小到大
    def Sort_Box(Car_list):
        Car_list.sort(key=lambda x: x.volume, reverse=False)
        return Car_list
class Packer:
    def packer_own(self,car_list_sort_fixed, box_sort_list):
        temp_l = 0    # 临时长，这是装入每个箱子的参照
        temp_w = 0
        temp_h = 0
        temp_ll = 0
        temp_ww = 0
        temp_hh = 0
        c = 0
        i_volume = 0
        b_volume = 0
        item_list = []
        peritem_list = []
        #for循环所有货箱 i:Car类
        for i in car_list_sort_fixed:
            # box_list_width = []
            # box_list_height = []
            while(True):
                # print(i.position)
                if i.position == [0,0,0]:  # 这里绝对不是i.position
                    if ((box_sort_list[0].height > i.height) or ( \
                                    box_sort_list[0].width > i.width) or ( \
                                    box_sort_list[0].length > i.length)):
                        break
                        # 这个意思就是说如果最开始就放不进去就不要放了
                    elif (
                            (box_sort_list[0].height <= i.height) and (\
                                    box_sort_list[0].width <= i.width) and (\
                                            box_sort_list[0].length <= i.length)
                    ):
                        d = PackingMethod.set_pose(box_sort_list[0],
                                                   PackingMethod.first_put(box_sort_list[0])).get_dimension()
                        temp_l = d[Attr.Axis_x]  # 初代放置作为模型开始对比
                        temp_w = d[Attr.Axis_y]
                        temp_h = d[Attr.Axis_z]
                        temp_ll = temp_l
                        temp_ww = temp_w
                        temp_hh = temp_h
                        i.position[Attr.Axis_x]+=d[Attr.Axis_x]    # 这里暂时相加
                        # i.position[attr.Attr.Axis_y]+=d[attr.Attr.Axis_y]
                        # i.position[attr.Attr.Axis_z]+=d[attr.Attr.Axis_z]
                        item_list.append(box_sort_list[0])
                        peritem_list.append(box_sort_list[0].volume/i.volume)
                        i_volume += box_sort_list[0].volume
                        # box_list_width.append(box_sort_list[0].width)
                        # box_list_height.append(box_sort_list[0].height)
                        c += 1
                        box_sort_list.pop(0)
                        continue
                    # 以上是空车第一次放置的时候
                else:
                    for j in box_sort_list[0:]: #原[1:]
                        temp_box = PackingMethod.set_pose(j, PackingMethod.pose_fit_arg(j, temp_l, temp_w,temp_h)).get_dimension()
                        if temp_box[Attr.Axis_z] > (i.height - i.position[Attr.Axis_z]):
                            temp_l = 0
                            temp_w = 0
                            temp_h = 0
                            temp_ll = 0
                            temp_hh = 0
                            temp_ww = 0
                            i.position[Attr.Axis_x] = 0
                            i.position[Attr.Axis_y] = 0
                            i.position[Attr.Axis_z] = 0
                            break
                            # 这句话意思即是后来的箱子已经放不进去了，换句话说就是已经装满了,然后变量全部归零
                        else:
                            if temp_box[Attr.Axis_y] <= (i.width - i.position[Attr.Axis_y]) and temp_box[Attr.Axis_x] <= ( i.length - i.position[Attr.Axis_x]) and temp_box[Attr.Axis_z] <= (i.height - i.position[Attr.Axis_z]):
                                i.position[Attr.Axis_x] += temp_box[Attr.Axis_x]
                                # print("7")
                                # global temp_ll, temp_ww, temp_hh
                                temp_ll = max(temp_ll, temp_box[Attr.Axis_x])  # 返回当前装箱体与之前装箱体的长宽高数据，并保留当前的最大的数值
                                temp_ww = max(temp_ww, temp_box[Attr.Axis_y])
                                temp_hh = max(temp_hh, temp_box[Attr.Axis_z])
                                item_list.append(j)
                                peritem_list.append(j.volume/i.volume)
                                i_volume += j.volume
                                box_sort_list.pop(0)
                                c += 1
                                Done = False
                                continue
                                # 都小于的时候，每一个块的长度相加 ，宽度 ，高度进行记录
                            elif temp_box[Attr.Axis_y] <= (i.width - i.position[Attr.Axis_y]) and temp_box[Attr.Axis_x] > (i.length - i.position[Attr.Axis_x]) and temp_box[Attr.Axis_z] <= (i.height - i.position[Attr.Axis_z]):
                                i.position[Attr.Axis_x] = 0
                                i.position[Attr.Axis_x] += temp_box[Attr.Axis_x]
                                i.position[Attr.Axis_y] += temp_ww
                                temp_ll = temp_box[Attr.Axis_x]  # 返回当前装箱体与之前装箱体的长宽高数据，并保留当前的最大的数值
                                temp_ww = temp_box[Attr.Axis_y]
                                temp_hh = max(temp_hh, temp_box[Attr.Axis_z])
                                item_list.append(j)
                                peritem_list.append(j.volume/i.volume)
                                i_volume += j.volume
                                box_sort_list.pop(0)
                                c += 1
                                continue
                                # 换到下一行，x重新置为0，将当前的最大W加上
                            else:
                                i.position[Attr.Axis_z] += temp_hh
                                i.position[Attr.Axis_x] = temp_box[Attr.Axis_x]
                                i.position[Attr.Axis_y] = temp_box[Attr.Axis_y]
                                temp_ll = temp_box[Attr.Axis_x]  # 返回当前装箱体与之前装箱体的长宽高数据，并保留当前的最大的数值
                                temp_ww = temp_box[Attr.Axis_y]
                                temp_hh = temp_box[Attr.Axis_z]
                                item_list.append(j)
                                peritem_list.append(j.volume/i.volume)
                                i_volume += j.volume
                                box_sort_list.pop(0)
                                c += 1
                                continue

                break
            b_volume = i.volume
        return i_volume / b_volume , c , item_list, peritem_list
        # i_volume / b_volume  体积占有率
        # c 装载个数
        # item_list # 装载对象列表
        # peritem_list 每一个占有的占有率的列表
            # print("11")

class Bin_env:
    def __init__(self, items, prob, index,
                 carlist):  # param:items:itemlist:data_create()的列表, problist, indexlist, carlist
        self.Reward_list = []
        self.Index_list = []
        self.State_list = []
        self.prob_list = []
        self.peritem_list = []
        self.carlist = carlist
        load_items = Creator.bin_create(items, index)  # 装箱
        # rate 占有率
        # num:
        # item_list
        #
        rate, num, item_list, peritem_list = Packer().packer_own(carlist, load_items)
        for z in item_list:
            self.Reward_list.append(z.volume)
        for x in range(len(item_list)):
            self.State_list.append(x)
            self.Index_list.append(index[x][0])
            self.prob_list.append(prob[x][0])
        self.peritem_list = peritem_list
        # print(peritem_list)

    # 定义方法：
    # 作用： 返回当前状态的利用率
    def render(self, state):
        pass
    def reset(self):
        pass

    def step(self, state):  # 这个动作就是当前的序列
        pass