import csv
import matplotlib.pyplot as plt
import network_serialization as ns
import wntr
import os
import numpy as np
import pipe_evaluation as pe
from itertools import islice
import optimization_objective as obj
import networkx as nx
import random
import math
import network_topology_utils as ntu


# 展示网络拓扑优化结果
def show_network_topology(solutions, wn, scale):
    network_name = "network_t"
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    for i in range(1, len(solutions)):
        sol = solutions[i]
        wn = ns.reload_network(network_name)
        update_network(wn, scale, edges, sol)
        wntr.graphics.plot_network(wn)
        plt.show()
    # 清理缓存
    os.remove(network_name)


def update_network(wn, scale, edges, sol):
    edge_list = list()
    for i in range(len(sol)):
        start_node_name = tuple(eval(edges[i]))[0]
        end_node_name = tuple(eval(edges[i]))[1]
        name = start_node_name + "," + end_node_name
        start_node = wn.get_node(start_node_name)
        end_node = wn.get_node(end_node_name)
        start_xy = start_node.coordinates
        end_xy = end_node.coordinates
        length = cal_length(start_xy, end_xy, scale)
        diameter = float(sol[i])
        if diameter > 0:
            wn.add_pipe(name=name,
                        start_node_name=start_node_name,
                        end_node_name=end_node_name,
                        diameter=diameter,
                        length=length)
            edge_list.append(name)
    # print('添加管段:', edge_list)


def cal_length(start_xy, end_xy, scale):
    length = math.sqrt(
        pow(start_xy[1] / 3.2808399 - end_xy[1] / 3.2808399, 2) + pow(start_xy[0] / 3.2808399 - end_xy[0] / 3.2808399,
                                                                      2)) * scale
    return length


def satisfaction_for_pipe_break(wn, scale, solutions, num_valve):
    network_name = 'network_t'
    # valve_layer = wntr.network.generate_valve_layer(wn, 'random', num_valve)
    valve_layer = wntr.network.generate_valve_layer(wn, 'strategic', 0)
    G = wn.get_graph().to_undirected()
    node_segments, link_segments, segment_size = wntr.metrics.valve_segments(G, valve_layer)
    segments = [[] for i in range(len(segment_size))]
    for node in node_segments.index:
        seg = node_segments[node]
        segments[seg - 1].append(node)
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    avg_supply_ratio_list = list()
    for i in range(1, len(solutions)):
        wn = ns.reload_network(network_name)
        update_network(wn, scale, edges, solutions[i])
        sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))
        # # Net2
        # sources.add('1')

        # supply_list = supply_for_segment_break(wn, sources, segments)
        # supply_list = supply_for_node_break(wn, sources)
        supply_list = supply_for_pipe_break(wn, sources)
        total_supply = 0
        for val in supply_list:
            total_supply += val
        avg_supply = 0
        if len(supply_list) > 0:
            avg_supply = total_supply / len(supply_list)
        wn = ns.reload_network(network_name)
        total_expected_demand = pe.cal_total_expected_demand(wn, sources)
        avg_supply_ratio = avg_supply / total_expected_demand
        avg_supply_ratio_list.append(avg_supply_ratio)
        # print('avg_supply_ratio', avg_supply_ratio)

    # 清理缓存
    os.remove(network_name)
    return avg_supply_ratio_list


def supply_for_node_break(wn, sources):
    network_name = 'network'
    ns.pickle_network(wn, network_name)
    supply_list = list()
    for node_name in wn.node_name_list:
        wn = ns.reload_network(network_name)
        close_links_for_node(wn, node_name)
        results = pe.hydraulic_sim(wn)
        demand = results.node['demand']
        total_demand = pe.cal_total_demand(demand, sources)
        supply_list.append(total_demand)

    # 清理缓存
    os.remove(network_name)
    return supply_list


def supply_for_segment_break(wn, sources, segments):
    network_name = 'network_temp'
    ns.pickle_network(wn, network_name)
    supply_list = list()
    for segmemt in segments:
        wn = ns.reload_network(network_name)
        for node_name in segmemt:
            close_links_for_node(wn, node_name)
        results = pe.hydraulic_sim(wn)
        demand = results.node['demand']
        total_demand = pe.cal_total_demand(demand, sources)
        supply_list.append(total_demand)

    # 清理缓存
    os.remove(network_name)
    return supply_list


def supply_for_pipe_break(wn, sources):
    network_name = 'network_temp'
    ns.pickle_network(wn, network_name)
    supply_list = list()
    for name in wn.link_name_list:
        wn = ns.reload_network(network_name)
        link = wn.get_link(name)
        if link.link_type == 'Valve':
            continue
        link.status = 0
        results = pe.hydraulic_sim(wn)
        demand = results.node['demand']
        total_demand = pe.cal_total_demand(demand, sources)
        supply_list.append(total_demand)

    # 清理缓存
    os.remove(network_name)
    return supply_list


def close_links_for_node(wn, node):
    links = wn.get_links_for_node(node)
    for link_name in links:
        link = wn.get_link(link_name)
        link.status = 0


def hydraulic_sim(wn):
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.required_pressure = 20  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3  # 5 psi = 3.516 m
    wn.options.hydraulic.demand_model = 'PDA'
    sim = wntr.sim.WNTRSimulator(wn=wn)
    results = sim.run_sim(convergence_error=False)
    return results


def cal_total_expected_demand(wn, sources):
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    expected_demand = wntr.metrics.expected_demand(wn)
    total_expected_demand = cal_total_demand(expected_demand, sources)
    return total_expected_demand


def cal_total_demand(demand, sources):
    total_demand = 0
    for i in range(len(demand.index)):
        for j in range(len(demand.columns)):
            if demand.columns[j] not in sources:
                if demand.values[i][j] > 0:
                    total_demand += demand.values[i][j]
    return total_demand


def satisfaction_for_fire(wn, scale, solutions, nodes=None):
    network_name = 'network_t'
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    avg_supply_ratio_list = list()
    # avg_node_num_list = list()
    if nodes is None:
        nodes = wn.node_name_list
    for i in range(len(solutions)):
        wn = ns.reload_network(network_name)
        update_network(wn, scale, edges, solutions[i])
        sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))
        # Net2
        sources.add('1')

        supply_ratio_list = list()
        fail_node_num_list = list()
        for name in nodes:
            wn = ns.reload_network(network_name)
            update_network(wn, scale, edges, solutions[i])
            node = wn.get_node(name)
            # base_demand = node.base_demand * multiplier
            # node.add_demand(base=base_demand, pattern_name=None)
            fire_flow_demand = 0.252  # 4000 gal/min = 0.252 m3/s
            fire_start = 10 * 3600
            fire_end = 14 * 3600
            node.add_fire_fighting_demand(wn, fire_flow_demand, fire_start, fire_end)
            results = hydraulic_sim(wn)

            # pressures = results.node['pressure'].loc[:, wn.junction_name_list]
            # count = 0
            # for i in range(wn.num_junctions):
            #     v = pressures.values[10][i]
            #     if v < 30:
            #         count += 1
            # fail_node_num_list.append(count)

            demand = results.node['demand'].loc[:, wn.junction_name_list]
            total_demand = cal_total_demand(demand, sources)
            total_expected_demand = cal_total_expected_demand(wn, sources)
            supply_ratio = total_demand / total_expected_demand
            supply_ratio_list.append(supply_ratio)

        # avg_node_num = np.mean(fail_node_num_list)
        # print('avg_node_num', avg_node_num)
        # avg_node_num_list.append(avg_node_num)

        avg_supply_ratio = np.mean(supply_ratio_list)
        print('avg_supply_ratio', avg_supply_ratio)
        avg_supply_ratio_list.append(avg_supply_ratio)

    # 清理缓存
    os.remove(network_name)
    return avg_supply_ratio_list


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


def write_supply_to_csv(file, objs):
    csv_writer = csv.writer(file)
    for i in range(len(objs)):
        csv_writer.writerow(objs[i])


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

    # 成本区间分析
    k = None
    solutions1 = list(islice(csv.reader(open(sol_file1, 'r')), k))
    objs1 = list(islice(csv.reader(open(obj_file1, 'r')), k))
    solutions2 = list(islice(csv.reader(open(sol_file2, 'r')), k))
    objs2 = list(islice(csv.reader(open(obj_file2, 'r')), k))
    # 区间方案
    sol1 = list()
    sol2 = list()
    # 各区间最坏的方案
    worst_list1 = list()
    worst_list2 = list()
    # 各区间最好的方案
    best_list1 = list()
    best_list2 = list()
    # 各区间平均
    avg_list1 = list()
    avg_list2 = list()
    for i in range(12):
        low = 100000 * i
        hight = low + 100000
        temp1 = list()
        temp1.append(solutions1[0])
        temp2 = list()
        temp2.append(solutions2[0])
        for j in range(len(objs1)):
            if low <= float(objs1[j][0]) <= hight:
                temp1.append(solutions1[j + 1])
            if low <= float(objs2[j][0]) <= hight:
                temp2.append(solutions2[j + 1])
        if len(temp1) == 1 or len(temp2) == 1:
            continue
        sol1.append(temp1)
        sol2.append(temp2)

    for i in range(len(sol1)):
        avg_supply_ratio1 = satisfaction_for_pipe_break(wn, scale, sol1[i], 15)
        print('本文模型：', avg_supply_ratio1)
        worst_list1.append(np.min(avg_supply_ratio1))
        best_list1.append(np.max(avg_supply_ratio1))
        avg_list1.append(np.mean(avg_supply_ratio1))
        avg_supply_ratio2 = satisfaction_for_pipe_break(wn, scale, sol2[i], 15)
        print('双目标模型：', avg_supply_ratio2)
        worst_list2.append(np.min(avg_supply_ratio2))
        best_list2.append(np.max(avg_supply_ratio2))
        avg_list2.append(np.mean(avg_supply_ratio2))
        print('-------------------------------------')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    xticks = ['(0,10]', '(10,20]', '(20,30]', '(30,40]', '(40,50]', '(50,60]', '(60,70]', '(70,80]', '(80,90]',
              '(90,100]',
              '(100,110]', '(110,120]']
    x = [i for i in range(len(worst_list1))]

    # 绘制区间最差折线图
    plt.xlabel('成本区间（万元）')
    plt.ylabel('需求满足度')
    plt.plot([i for i in range(len(worst_list1))], worst_list1, marker='o', c='r', label='本文模型')
    plt.plot([i for i in range(len(worst_list2))], worst_list2, marker='^', c='b', label='双目标模型')
    plt.xticks(x, xticks, rotation=30)
    plt.grid(ls='--')
    plt.legend()
    plt.show()
    print('worst_list1', worst_list1)
    print('worst_list2', worst_list2)

    # 绘制区间最优折线图
    plt.xlabel('成本区间（万元）')
    plt.ylabel('需求满足度')
    plt.plot([i for i in range(len(best_list1))], best_list1, marker='o', c='r', label='本文模型')
    plt.plot([i for i in range(len(best_list2))], best_list2, marker='^', c='b', label='双目标模型')
    plt.xticks(x, xticks, rotation=30)
    plt.grid(ls='--')
    plt.legend()
    plt.show()
    print('best_list1', best_list1)
    print('best_list2', best_list2)

    # 绘制区间平均折线图
    plt.xlabel('成本区间（万元）')
    plt.ylabel('需求满足度')
    plt.plot([i for i in range(len(avg_list1))], avg_list1, marker='o', c='r', label='本文模型')
    plt.plot([i for i in range(len(avg_list2))], avg_list2, marker='^', c='b', label='双目标模型')
    plt.xticks(x, xticks, rotation=30)
    plt.grid(ls='--')
    plt.legend()
    plt.show()
    print('avg_list1', avg_list1)
    print('avg_list2', avg_list2)


    # # 目标文件追加各方案的需求满足度
    # k = None
    # solutions1 = list(islice(csv.reader(open(sol_file1, 'r')), k))
    # objs1 = islice(csv.reader(open(obj_file1, 'r')), k)
    # objs1 = get_obj_list(objs1)
    # cost1 = get_cost_list(objs1)
    #
    # solutions2 = list(islice(csv.reader(open(sol_file2)), k))
    # objs2 = islice(csv.reader(open(obj_file2, 'r')), k)
    # objs2 = get_obj_list(objs2)
    # cost2 = get_cost_list(objs2)
    # edges = solutions2[0]
    # for i in range(len(objs1)):
    #     temp = list()
    #     temp.append(solutions1[0])
    #     temp.append(solutions1[i+1])
    #     avg_supply_ratio = satisfaction_for_pipe_break(wn, scale, temp, 15)
    #     print(avg_supply_ratio)
    #     objs1[i].append(avg_supply_ratio[0])
    #     print(objs1[i])
    # for i in range(len(objs2)):
    #     temp = list()
    #     temp.append(solutions2[0])
    #     temp.append(solutions2[i+1])
    #     avg_supply_ratio = satisfaction_for_pipe_break(wn, scale, temp, 15)
    #     print(avg_supply_ratio)
    #     objs2[i].append(avg_supply_ratio)
    #     print(objs1[i])
    #
    # # 写入各方案的目标值和需求满足度
    # supply_file1 = open(r'D:\Projects\reliability\objs1.csv', 'w',
    #                     encoding='utf-8', newline='')
    # write_supply_to_csv(supply_file1, objs1)
    #
    # supply_file2 = open(r'D:\Projects\reliability\objs2.csv', 'w',
    #                     encoding='utf-8', newline='')
    # write_supply_to_csv(supply_file2, objs2)

    # 清理缓存
    os.remove(network_name)
