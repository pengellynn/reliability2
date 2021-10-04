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
    # valve_layer = wntr.network.generate_valve_layer(wn, 'strategic', 0)
    # valve_layer = wntr.network.generate_valve_layer(wn, 'random', num_valve)
    # G = wn.get_graph().to_undirected()
    # node_segments, link_segments, segment_size = wntr.metrics.valve_segments(G, valve_layer)
    # segments = [[] for i in range(len(segment_size))]
    # for node in node_segments.index:
    #     seg = node_segments[node]
    #     segments[seg - 1].append(node)
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    avg_supply_ratio_list = list()
    for i in range(1, len(solutions)):
        wn = ns.reload_network(network_name)
        update_network(wn, scale, edges, solutions[i])
        sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))
        # Net2
        sources.add('1')

        # supply_list = supply_for_segment_break(wn, sources, segments)
        supply_list = supply_for_node_break(wn, sources)
        # supply_list = supply_for_pipe_break(wn, sources)

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
        print('avg_supply_ratio', avg_supply_ratio)

    # 清理缓存
    os.remove(network_name)
    return avg_supply_ratio_list


def supply_for_node_break(wn, sources):
    network_name = 'network_temp'
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


def close_links_for_node(wn, node):
    links = wn.get_links_for_node(node)
    for link_name in links:
        link = wn.get_link(link_name)
        link.status = 0


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

        # if link.link_type == 'Pump':
        #     continue
        # wn = wntr.morph.split_pipe(wn, name, name + 'B', name + 'node')
        # leak_node = wn.get_node(name + 'node')
        # leak_node.add_leak(wn, area=0.05, start_time=0 * 3600, end_time=12 * 3600)

        results = pe.hydraulic_sim(wn)
        demand = results.node['demand']
        total_demand = pe.cal_total_demand(demand, sources)
        supply_list.append(total_demand)
    # 清理缓存
    os.remove(network_name)
    return supply_list


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


def satisfaction_for_fire(wn, scale, solutions, nodes):
    network_name = 'network_t'
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    avg_supply_ratio_list = list()
    avg_node_num_list = list()
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
            pressures = results.node['pressure'].loc[:, wn.junction_name_list]
            count = 0
            for i in range(wn.num_junctions):
                v = pressures.values[10][i]
                if v < 30:
                    count += 1
            fail_node_num_list.append(count)
            # demand = results.node['demand'].loc[:, wn.junction_name_list]
            # total_demand = cal_total_demand(demand, sources)
            # total_expected_demand = cal_total_expected_demand(wn, sources)
            # supply_ratio = total_demand / total_expected_demand
            # supply_ratio_list.append(supply_ratio)
        avg_node_num = np.mean(fail_node_num_list)
        print('avg_node_num', avg_node_num)
        avg_node_num_list.append(avg_node_num)
        # avg_supply_ratio = np.mean(supply_ratio_list)
        # print('avg_supply_ratio', avg_supply_ratio)
        # avg_supply_ratio_list.append(avg_supply_ratio)

    # 清理缓存
    os.remove(network_name)
    return avg_node_num_list


def write_supply_to_csv(file, cost, avg_supply_ratio):
    csv_writer = csv.writer(file)
    csv_writer.writerow(['cost', 'avg_supply_ratio'])
    for i in range(len(avg_supply_ratio)):
        row = list()
        row.append(cost[i])
        row.append(avg_supply_ratio[i])
        csv_writer.writerow(row)


def get_cost_list(obj_val):
    cost_list = list()
    for row in obj_val:
        cost_list.append(float(row[0]))
    return cost_list


def cal_cost(solutions, wn, scale):
    network_name = "network_t"
    ns.pickle_network(wn, network_name)
    edges = solutions[0]
    prices = {0: 0, 0.2032: 433.93, 0.3048: 720.14, 0.4064: 962.99, 0.508: 1197.29}
    cost_sol_dict = dict()
    for i in range(1, len(solutions)):
        sol = solutions[i]
        wn = ns.reload_network(network_name)
        update_network(wn, scale, edges, sol)
        cost = 0
        new_pipes = list()
        for j in range(len(sol)):
            d = float(sol[j])
            if d > 0:
                start_node_name = tuple(eval(edges[j]))[0]
                end_node_name = tuple(eval(edges[j]))[1]
                name = start_node_name + "," + end_node_name
                pipe = wn.get_link(name)
                cost += prices[d] * pipe.length
                new_pipes.append(name)
        if cost not in cost_sol_dict.keys():
            pipes = list()
        else:
            pipes = cost_sol_dict[cost]
        pipes.append(new_pipes)
        cost_sol_dict.update({cost: pipes})

    for k, v in cost_sol_dict.items():
        print(k)
        for pipes in v:
            print(pipes)
            wn = ns.reload_network(network_name)
            len_list = list()
            for p in pipes:
                nodes = p.split(",")
                wn.add_pipe(name=p,
                            start_node_name=nodes[0],
                            end_node_name=nodes[1])
                len_list.append(wn.get_link(p).length)
            print(len_list)
            wntr.graphics.plot_network(wn, link_attribute='length', link_width=3)
            plt.show()
        print('-------------------')

    # 清理缓存
    os.remove(network_name)


if __name__ == '__main__':
    inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net2.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)

    scale = 150
    ntu.reset_all_pipe_length(wn=wn, scale=scale)

    network_name = "network"
    ns.pickle_network(wn, network_name)
    sol_file1 = r'C:\Users\29040\Desktop\exp2\Net2\200_500_200_2\cost_todini_flowrate\Result\solution.csv'
    obj_file1 = r'C:\Users\29040\Desktop\exp2\Net2\200_500_200_2\cost_todini_flowrate\Result\ObjV.csv'
    sol_file2 = r'C:\Users\29040\Desktop\exp2\Net2\200_500_200_2\cost_todini_wbe\Result\solution.csv'
    obj_file2 = r'C:\Users\29040\Desktop\exp2\Net2\200_500_200_2\cost_todini_wbe\Result\ObjV.csv'

    # inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net3.inp'
    # wn = wntr.network.WaterNetworkModel(inp_file)
    # sol_file1 = r'C:\Users\29040\Desktop\exp0\Net3\todini\100_1000_100_2\Result\solution.csv'
    # obj_file1 = r'C:\Users\29040\Desktop\exp0\Net3\todini\100_1000_100_2\Result\ObjV.csv'
    # sol_file2 = r'C:\Users\29040\Desktop\exp0\Net3\3obj\200_500_200_2\Result\solution.csv'
    # obj_file2 = r'C:\Users\29040\Desktop\exp0\Net3\3obj\200_500_200_2\Result\ObjV.csv'

    k = 5
    solutions1 = list(islice(csv.reader(open(sol_file1, 'r')), k))
    avg_supply_ratio1 = satisfaction_for_pipe_break(wn, scale, solutions1, 15)
    # avg_supply_ratio1 = np.array([0 for i in range(k)], dtype='float64')
    # for i in range(1):
    #     avg_supply_ratio1 += np.array(satisfaction_for_pipe_break(wn, scale, solutions1, 15))

    objs1 = islice(csv.reader(open(obj_file1, 'r')), k)
    cost1 = get_cost_list(objs1)[:k]

    solutions2 = list(islice(csv.reader(open(sol_file2)), k))
    avg_supply_ratio2 = satisfaction_for_pipe_break(wn, scale, solutions2, 15)
    # avg_supply_ratio2 = np.array([0 for i in range(k)], dtype='float64')
    # for i in range(1):
    #     avg_supply_ratio2 += np.array(satisfaction_for_pipe_break(wn, scale, solutions2, 15))
    objs2 = islice(csv.reader(open(obj_file2, 'r')), k)
    cost2 = get_cost_list(objs2)[:k]

    # nodes = set()
    # for i in range(5):
    #     name = random.choice(wn.junction_name_list)
    #     node = wn.get_node(name)
    #     if node.base_demand > 0:
    #         nodes.add(name)
    #
    # solutions1 = islice(csv.reader(open(sol_file1, 'r')), k)
    # avg_supply_ratio1 = satisfaction_for_fire(wn, solutions1, nodes)
    # objs1 = islice(csv.reader(open(obj_file1, 'r')), k)
    # cost1 = get_cost_list(objs1)[:k]
    #
    # wn = ns.reload_network(network_name)
    # solutions2 = islice(csv.reader(open(sol_file2)), k)
    # avg_supply_ratio2 = satisfaction_for_fire(wn, solutions2, nodes)
    # objs2 = islice(csv.reader(open(obj_file2, 'r')), k)
    # cost2 = get_cost_list(objs2)[:k]

    # 写入各方案的成本及其对应供应量
    supply_file1 = open(r'D:\Projects\reliability\cost_todini_supply.csv', 'w',
                        encoding='utf-8', newline='')
    write_supply_to_csv(supply_file1, cost1, avg_supply_ratio1)

    supply_file2 = open(r'D:\Projects\reliability\cost_wbe_todini_supply.csv', 'w',
                        encoding='utf-8', newline='')
    write_supply_to_csv(supply_file2, cost2, avg_supply_ratio2)

    # obj_val = csv.reader(open(r'D:\Projects\reliability\cost_todini_supply.csv', 'r'))
    # next(obj_val)
    # cost1 = list()
    # avg_supply1 = list()
    # avg_supply_ratio1 = list()
    # for row in obj_val:
    #     cost1.append(float(row[0]))
    #     avg_supply1.append(float(row[1]))
    #     avg_supply_ratio1.append(float(row[2]))
    #
    # obj_val = csv.reader(open(r'D:\Projects\reliability\cost_wbe_todini_supply.csv', 'r'))
    # next(obj_val)
    # cost2 = list()
    # avg_supply2 = list()
    # avg_supply_ratio2 = list()
    # for row in obj_val:
    #     cost2.append(float(row[0]))
    #     avg_supply2.append(float(row[1]))
    #     avg_supply_ratio2.append(float(row[2]))

    # 绘制cost-avg_supply_ratio散点图
    plt.xlabel('cost')
    plt.ylabel('avg_supply_ratio')
    plt.scatter(cost1, avg_supply_ratio1, marker='o', c='r')
    plt.scatter(cost2, avg_supply_ratio2, marker='o', c='b')
    plt.show()

    # show_network_topology(solutions, wn)

    # file = r'C:\Users\29040\Desktop\ret\2\Net2\Result\ObjV.csv'
    # obj_val = csv.reader(open(file, 'r'))
    # f1 = list()
    # f2 = list()
    # f3 = list()
    # for row in obj_val:
    #     f1.append(float(row[0]))
    #     f2.append(float(row[1]))
    #     f3.append(float(row[2]))
    # show_obj_3d(f1, f2, f3)

    # 清理缓存
    os.remove(network_name)
