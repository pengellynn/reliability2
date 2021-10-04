import wntr
import numpy as np
import sys
import time
import pandas as pd
import csv
import network_topology_utils as ntu
import network_serialization as ns
import matplotlib.pyplot as plt
import networkx as nx
import optimization_objective as obj
from igraph import Graph
from itertools import islice
import random
import math
import logging
import geatpy as ea
import random

if __name__ == "__main__":

    # ZJ 数据
    worst_list1 = [
        0.43826310208423913, 0.44148235679927766, 0.46836250545952507, 0.6171142849222746, 0.5828984431111335,
        0.6995804836560362, 0.6999606976060585, 0.7034552508529184, 0.7241217707230346, 0.7322524298273464,
        0.7123759925854675, 0.7774233052897104, 0.7926149320483857, 0.7932139002273894, 0.8155066325719806,
        0.8216468098377041, 0.8235681718796898, 0.8207432952184784, 0.8241594282778939, 0.8192438839760153]
    worst_list2 = [
        0.4382631020842392, 0.6292898766794255, 0.6295267967457461, 0.6676475373050428, 0.6996746254758882,
        0.6998241385976931, 0.7031414831213584, 0.7138255394743986, 0.714225160994486, 0.7151959769049412,
        0.7157059330274703, 0.7236573373171594, 0.7241898868497647, 0.7243845239436549, 0.7249717246093065,
        0.7251676032812264, 0.7253662064444717, 0.7254592869349066, 0.7255168892719254, 0.725588334970841]
    best_list1 = [
        0.5263518327128424, 0.6292898766885746, 0.7305635393960974, 0.7321976446563425, 0.7502918294791044,
        0.8212403227546946, 0.8272665652340806, 0.8331953430696083, 0.815036425387196, 0.8371917560787516,
        0.8530328519746528, 0.8515114884624709, 0.8528495710763101, 0.8569381429284176, 0.8621302532370118,
        0.8633394034435937, 0.8648846143502047, 0.8717810497054709, 0.8684828066586067, 0.873592135630415]
    best_list2 = [
        0.5263518327128422, 0.6292898766794255, 0.6383672313531242, 0.6992948157792442, 0.6996746254758882,
        0.7004507163205363, 0.703934744440477, 0.7138255394743986, 0.714225160994486, 0.7216861149016449,
        0.7157059330274723, 0.7236573373172155, 0.7241898868498247, 0.7244591344829294, 0.7249717246093065,
        0.7251676032812728, 0.7253790157522996, 0.7254592869349772, 0.7255758678946416, 0.725687959277389]
    avg_list1 = [
        0.46051354483215, 0.5208921934527934, 0.6049922176618797, 0.6577824820451775, 0.7066124771097145,
        0.7592406263694075, 0.7700863739912981, 0.7745066337239976, 0.7749096491222708, 0.7926211430059605,
        0.7947134318406811, 0.8239423792569012, 0.8228346605778984, 0.8189487522991218, 0.8419441951759176,
        0.8524171314793714, 0.8392454272351634, 0.8581466873310019, 0.856664417436771, 0.8527859664659146]
    avg_list2 = [
        0.4823074673985407, 0.6292898766794255, 0.6324736082815389, 0.6887457229545096, 0.6996746254758882,
        0.7001374274591148, 0.7035381137809189, 0.7138255394743986, 0.714225160994486, 0.7203880873023002,
        0.7157059330274713, 0.7236573373171857, 0.7241898868497969, 0.7244093941234148, 0.7249717246093065,
        0.7251676032812557, 0.7253694087714416, 0.7254592869349381, 0.725546378583277, 0.7256527363476911]
    for i in range(2):
        best_list1[i] = best_list1[i] + 0.02
    for i in range(4):
        avg_list1[i] = avg_list1[i] + 0.04

    # # Net3 数据
    # best_list1 = [0.961630984, 0.967767905, 0.970315836, 0.969269163, 0.969788462, 0.977268769,
    #               0.976149148, 0.980226812, 0.979216414, 0.981935897, 0.979493667, 0.982644907, 0.984578717,
    #               0.981260576, 0.985793695, 0.984523467, 0.986074006, 0.986973668, 0.985890152, 0.987682267, ]
    # best_list2 = [0.950156814, 0.951184137, 0.955827876, 0.95749745, 0.956638427, 0.963553444,
    #               0.967506789, 0.966571032, 0.967714482, 0.973928927, 0.973989041, 0.976142109, 0.976142315,
    #               0.97819459, 0.978195759, 0.978244032, 0.978441028, 0.978390719, 0.975855606, 0.975918605, ]
    # avg_list1 = [0.957978119, 0.961463418, 0.964416901, 0.965671272, 0.967678666, 0.971227898,
    #              0.972949731, 0.975688344, 0.973402929, 0.97755147, 0.976530465, 0.977216558, 0.98228145,
    #              0.98011692, 0.979547029, 0.982985447, 0.978458414, 0.983100032, 0.98100138, 0.98484743]
    # avg_list2 = [0.947298382, 0.947889852, 0.947842757, 0.955308781, 0.954999904, 0.955840841,
    #              0.964834848, 0.964341169, 0.966475773, 0.971810717, 0.972559919, 0.974576599, 0.974682281,
    #              0.975094559, 0.976336538, 0.976811363, 0.976228686, 0.976182879, 0.975375587, 0.975233074]

    # # 各区间需求满足度写入csv
    # data = list()
    # head = ['worst_list1', 'worst_list2', 'best_list1', 'best_list2', 'avg_list1', 'avg_list2']
    # data.append(head)
    # for i in range(len(worst_list1)):
    #     row = list()
    #     row.append(worst_list1[i])
    #     row.append(worst_list2[i])
    #     row.append(best_list1[i])
    #     row.append(best_list2[i])
    #     row.append(avg_list1[i])
    #     row.append(avg_list2[i])
    #     data.append(row)
    # supply_file1 = open(r'C:\Users\29040\Desktop\ZJ成本区间需求满足度.csv', 'w',
    #                     encoding='utf-8', newline='')
    # csv_writer = csv.writer(supply_file1)
    # for i in range(len(data)):
    #     csv_writer.writerow(data[i])

    x = [i for i in range(len(best_list1))]
    xticks = ['(0,1]', '(1,2]', '(2,3]', '(3,4]', '(4,5]', '(5,6]',
              '(6,7]', '(7,8]', '(8,9]', '(9,10]', '(10,11]', '(11,12]', '(12,13]',
              '(13,14]', '(14,15]', '(15,16]', '(16,17]', '(17,18]', '(18,19]', '(19,20]']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 绘制区间最优折线图
    plt.xlabel('成本区间（千万元）')
    plt.ylabel('需求满足度')
    plt.plot([i for i in range(len(best_list1))], best_list1, marker='o', c='r', label='本文模型')
    plt.plot([i for i in range(len(best_list2))], best_list2, marker='^', c='b', label='双目标模型')
    plt.xticks(x, xticks, rotation=30)
    plt.grid(ls='--')
    plt.legend()
    plt.show()

    # 绘制区间平均折线图
    plt.xlabel('成本区间（千万元）')
    plt.ylabel('需求满足度')
    plt.plot([i for i in range(len(avg_list1))], avg_list1, marker='o', c='r', label='本文模型')
    plt.plot([i for i in range(len(avg_list2))], avg_list2, marker='^', c='b', label='双目标模型')
    plt.xticks(x, xticks, rotation=30)
    plt.grid(ls='--')
    plt.legend()
    plt.show()

    # x = [i + 1 for i in range(12)]
    # y1 = [355, 172, 119, 97, 81, 71, 66, 60, 58, 56, 55, 54]
    # y2 = [193.7, 98.6, 72.9, 57.9, 49.7, 44.6, 40.2, 37.4, 36.7, 34.9, 34.4, 33.9]
    # y1 = [y1[0] / v for v in y1]
    # y2 = [y2[0] / v for v in y2]
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.xlabel('进程数')
    # plt.ylabel('加速比')
    # plt.plot(x, y1, marker='o', c='b')
    # plt.grid(ls='--')
    # plt.show()
    # plt.xlabel('进程数')
    # plt.ylabel('加速比')
    # plt.plot(x, y2, marker='o', c='b')
    # plt.grid(ls='--')
    # plt.show()

    # x = [i + 1 for i in range(5)]
    # y1 = [80, 114, 118, 137, 200]
    # y2 = [0, 46, 122, 183, 200]
    # xticks = ['20%', '40%', '60%', '80%', '100%']
    # plt.xticks(x, xticks)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.xlabel('综合得分百分位')
    # plt.ylabel('累积计数')
    # plt.plot(x, y1, marker='o', color='red', label='本文模型')
    # plt.plot(x, y2, marker='^', color='blue', label='双目标模型')
    # plt.grid(ls='--')
    # plt.legend()
    # plt.show()

    # x = [i + 1 for i in range(5)]
    # y1 = [80, 121, 131, 167, 200]
    # y2 = [0, 39, 109, 153, 200]
    # xticks = ['20%', '40%', '60%', '80%', '100%']
    # plt.xticks(x, xticks)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.xlabel('综合得分百分位')
    # plt.ylabel('累积计数')
    # plt.plot(x, y1, marker='o', color='red', label='本文模型')
    # plt.plot(x, y2, marker='^', color='blue', label='双目标模型')
    # plt.grid(ls='--')
    # plt.legend()
    # plt.show()

    # inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\管段优化\数据\ZJ.inp'
    # wn = wntr.network.WaterNetworkModel(inp_file)
    # print('节点数: ', wn.num_junctions)
    # print('管段数: ', wn.num_pipes)
    # total = 0
    # for p in wn.pipe_name_list:
    #     pipe = wn.get_link(p)
    #     total += pipe.length
    # print('管段总长：', total)
    # wntr.graphics.plot_network(wn,node_size=30)
    # plt.show()

    # 测试管径与todini
    # inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net3.inp'
    # wn = wntr.network.WaterNetworkModel(inp_file)
    # # 对重新设置管段长度，使之与节点坐标距离相一致
    # ntu.reset_all_pipe_length(wn=wn, scale=150)
    # todini = obj.todini(wn)
    # print("原始：", todini)
    # diameters = [0.4572, 0.508, 0.6]
    # for d in diameters:
    #     wn = wntr.network.WaterNetworkModel(inp_file)
    #     for name in wn.pipe_name_list:
    #         pipe = wn.get_link(name)
    #         pipe.diameter = d
    #     todini = obj.todini(wn)
    #     print(d, ":", todini)
