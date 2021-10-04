import numpy as np
import math
import wntr
import network_serialization as ns
import os
from itertools import islice
import csv
import analysis__
import optimization_objective as obj
import pipe_evaluation as pe
import network_topology_utils as ntu
import matplotlib.pyplot as plt


def topsis(matrix):
    # 目标归一化，构建标准化决策矩阵
    max_f1 = matrix.max(axis=0)[0]
    max_f2 = matrix.max(axis=0)[1]
    max_f3 = matrix.max(axis=0)[2]
    min_f1 = matrix.min(axis=0)[0]
    min_f2 = matrix.min(axis=0)[1]
    min_f3 = matrix.min(axis=0)[2]
    for i in range(len(matrix)):
        matrix[i][0] = (max_f1 - matrix[i][0]) / (max_f1 - min_f1)
        matrix[i][1] = (matrix[i][1] - min_f2) / (max_f2 - min_f2)
        matrix[i][2] = (matrix[i][2] - min_f3) / (max_f3 - min_f3)

    # 构造加权的标准化决策矩阵
    for i in range(len(matrix)):
        matrix[i][0] = matrix[i][0] * 0.333
        matrix[i][1] = matrix[i][1] * 0.333
        matrix[i][2] = matrix[i][2] * 0.333

    # 负理想方案
    z0 = [matrix.max(axis=0)[0], matrix.min(axis=0)[1], matrix.min(axis=0)[2]]
    # 理想方案
    z1 = [matrix.min(axis=0)[0], matrix.max(axis=0)[1], matrix.max(axis=0)[2]]

    scores = {}
    for i in range(len(matrix)):
        # 与负理想方案的欧氏距离
        d0 = 0
        for j in range(len(matrix[0])):
            d0 += pow(matrix[i][j] - z0[j], 2)
        d0 = math.sqrt(d0)
        # 与理想方案的欧氏距离
        d1 = 0
        for j in range(len(matrix[0])):
            d1 += pow(matrix[i][j] - z1[j], 2)
        d1 = math.sqrt(d1)
        # 贴近度
        score = d0 / (d0 + d1)
        scores[i] = score

    # 按综合评价排序
    sorted_res = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_res


def topsis2(matrix):
    # 目标归一化，构建标准化决策矩阵
    max_f1 = matrix.max(axis=0)[0]
    max_f2 = matrix.max(axis=0)[1]
    min_f1 = matrix.min(axis=0)[0]
    min_f2 = matrix.min(axis=0)[1]
    for i in range(len(matrix)):
        matrix[i][0] = (matrix[i][0] - min_f1) / (max_f1 - min_f1)
        matrix[i][1] = (matrix[i][1] - min_f2) / (max_f2 - min_f2)

    # 构造加权的标准化决策矩阵
    for i in range(len(matrix)):
        matrix[i][0] = matrix[i][0] * 0.5
        matrix[i][1] = matrix[i][1] * 0.5

    # 负理想方案
    z0 = [matrix.min(axis=0)[0], matrix.min(axis=0)[1]]
    # 理想方案
    z1 = [matrix.max(axis=0)[0], matrix.max(axis=0)[1]]

    scores = {}
    for i in range(len(matrix)):
        # 与负理想方案的欧氏距离
        d0 = 0
        for j in range(len(matrix[0])):
            d0 += pow(matrix[i][j] - z0[j], 2)
        d0 = math.sqrt(d0)
        # 与理想方案的欧氏距离
        d1 = 0
        for j in range(len(matrix[0])):
            d1 += pow(matrix[i][j] - z1[j], 2)
        d1 = math.sqrt(d1)
        # 贴近度
        score = d0 / (d0 + d1)
        scores[i] = score

    # 按综合评价排序
    sorted_res = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_res


def get_obj_list(objs):
    obj_list = list()
    for row in objs:
        temp = list()
        for obj in row:
            temp.append(float(obj))
        obj_list.append(temp)
    return obj_list


def get_cost_list(objs):
    cost_list = list()
    for row in objs:
        cost_list.append(float(row[0]))
    return cost_list


def get_cumulate_count(count_list):
    res = list()
    sum = 0
    for count in count_list:
        sum += count
        res.append(sum)
    return res


def write_score_to_csv(file, score_list):
    csv_writer = csv.writer(file)
    csv_writer.writerow(['sol_index', 'score'])
    for i in range(len(score_list)):
        row = list()
        row.append(score_list[i][0])
        row.append(score_list[i][1])
        csv_writer.writerow(row)


if __name__ == '__main__':
    inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\Net2.inp'
    # inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\Net3.inp'
    # inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\ZJ.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)

    # Net3 scale 800; Net2 scale 150; ZJ scale 700
    scale = 150
    ntu.reset_all_pipe_length(wn=wn, scale=scale)

    network_name = "network"
    ns.pickle_network(wn, network_name)

    sol_file1 = r'C:\Users\29040\Desktop\lastExp2\Net2\500\cost_todini_wbe\Result\solution.csv'
    obj_file1 = r'C:\Users\29040\Desktop\lastExp2\Net2\500\cost_todini_wbe\Result\ObjV.csv'
    sol_file2 = r'C:\Users\29040\Desktop\lastExp2\Net2\500\cost_todini\Result\solution.csv'
    obj_file2 = r'C:\Users\29040\Desktop\lastExp2\Net2\500\cost_todini\Result\ObjV.csv'

    k = None
    solutions1 = list(islice(csv.reader(open(sol_file1, 'r')), k))
    objs1 = islice(csv.reader(open(obj_file1, 'r')), k)
    objs1 = get_obj_list(objs1)
    cost1 = get_cost_list(objs1)

    solutions2 = list(islice(csv.reader(open(sol_file2)), k))
    objs2 = islice(csv.reader(open(obj_file2, 'r')), k)
    objs2 = get_obj_list(objs2)
    cost2 = get_cost_list(objs2)
    edges = solutions2[0]
    # 反算双目标模型的加权边介数熵
    for i in range(1, len(solutions2)):
        wn = ns.reload_network(network_name)
        analysis__.update_network(wn, scale, edges, solutions2[i])
        # 针对Net2的数据预处理
        sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))
        sources.add('1')
        targets = wn.junction_name_list
        targets.remove('1')
        betweenness_dict = pe.weighted_betweenness(wn, None, sources, targets)
        entropy = obj.weighted_betweenness_entropy(betweenness_dict)
        row = list()
        row.append(objs2[i - 1][0])
        row.append(entropy)
        row.append(objs2[i - 1][1])
        objs2[i - 1] = row

    # 选择不大于指定成本下的方案来进行综合评价
    cost_threshold = 20000000000
    objs1_idx = list()
    objs1_val = list()
    for i in range(len(objs1)):
        if objs1[i][0] <= cost_threshold:
            objs1_idx.append(i)
            objs1_val.append([objs1[i][j] for j in range(0, len(objs1[i]))])

    objs2_idx = list()
    objs2_val = list()
    for i in range(len(objs1)):
        if objs2[i][0] <= cost_threshold:
            objs2_idx.append(i)
            objs2_val.append([objs2[i][j] for j in range(0, len(objs2[i]))])

    sorted_res = topsis(np.vstack([objs1_val, objs2_val]))

    # 综合性能分布对比
    level = 5
    count_list1 = list()
    count_list2 = list()
    for i in range(level):
        count1 = 0
        count2 = 0
        m = int(len(sorted_res) / level)
        for j in range(m):
            index = i * m + j
            if index >= len(sorted_res):
                continue
            if sorted_res[index][0] < len(objs1_idx):
                count1 += 1
            else:
                count2 += 1
        count_list1.append(count1)
        count_list2.append(count2)
    x = [100 / level * (i + 1) for i in range(level)]
    y1 = get_cumulate_count(count_list1)
    y2 = get_cumulate_count(count_list2)
    print(count_list1)
    print(count_list2)
    xticks = ['20%', '40%', '60%', '80%', '100%']
    plt.xticks(x, xticks)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('综合得分百分位')
    plt.ylabel('累积计数')
    plt.plot(x, y1, marker='o', color='red', label='本文模型')
    plt.plot(x, y2, marker='^', color='blue', label='双目标模型')
    plt.grid(ls='--')
    plt.legend()
    plt.show()

    # # 最佳方案对比
    # count1 = 3
    # count2 = 3
    # idx1 = []
    # idx2 = []
    # for item in sorted_res:
    #     index = item[0]
    #     if count1 > 0 and index < len(objs1_idx):
    #         idx1.append(objs1_idx[index])
    #         count1 -= 1
    #     if count2 > 0 and index >= len(objs1_idx):
    #         idx2.append(objs2_idx[int(index - len(objs1_idx))])
    #         count2 -= 1
    #     if count1 <= 0 and count2 <= 0:
    #         break
    #
    # wn = ns.reload_network(network_name)
    # sols1 = [solutions1[0]]
    # for idx in idx1:
    #     sols1.append(solutions1[idx])
    #     print(objs1[idx])
    # avg_supply_ratio1 = result_analysis.satisfaction_for_pipe_break(wn, scale, sols1, 15)
    #
    # wn = ns.reload_network(network_name)
    # sols2 = [solutions2[0]]
    # for idx in idx2:
    #     sols2.append(solutions2[idx])
    #     print(objs2[idx])
    # avg_supply_ratio2 = result_analysis.satisfaction_for_pipe_break(wn, scale, sols2, 15)
    #
    # print(avg_supply_ratio1)
    # print(avg_supply_ratio2)

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.xlabel('方案排名')
    # plt.ylabel('供水满足度')
    # x = [i + 1 for i in range(level)]
    # # x1 = [objs1[idx][0] for idx in idx1]
    # # x2 = [objs2[idx][0] for idx in idx2]
    # y1 = [v for v in avg_supply_ratio1]
    # y2 = [v for v in avg_supply_ratio2]
    # plt.scatter(x, y1, marker='o', color='red', label='本文模型')
    # plt.scatter(x, y2, marker='^', color='blue', label='双目标模型')
    # plt.legend()
    # plt.show()

    # 清理缓存
    os.remove(network_name)
