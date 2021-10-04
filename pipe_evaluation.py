import wntr
import networkx as nx
import pandas as pd
from itertools import islice
import network_serialization as ns
import os
import matplotlib.pyplot as plt
import network_topology_utils as ntu
import customize_drawing
import time


# 计算各管段介数
def edge_betweenness(wn):
    g = nx.Graph(wn.get_graph().to_undirected())
    betweenness_list = nx.edge_betweenness_centrality(g)
    betweenness_dict = dict()
    for k, v in betweenness_list.items():
        links1 = wn.get_links_for_node(k[0])
        links2 = wn.get_links_for_node(k[1])
        link_name = set(links1).intersection(set(links2)).pop()
        betweenness_dict.update({link_name: v})
    return betweenness_dict


# 计算各管段加权介数
def weighted_betweenness(wn, k=None, sources=None, targets=None):
    # 构建加权无向图
    length = wn.query_link_attribute('length')
    diameter = wn.query_link_attribute('diameter')
    roughness = wn.query_link_attribute('roughness')
    w = roughness * length / pow(diameter, 5)
    # w = roughness * length / diameter

    # 两个节点只有阀门连接时，权值设为0
    for valve in wn.valve_name_list:
        w[valve] = 0

    # pump权值设为0, 即阻力为0
    index = list(w.index)
    values = list(w.values)
    for pump in wn.pump_name_list:
        index.append(pump)
        values.append(0)

    weight = pd.Series(data=values, index=index)
    wG = nx.Graph(wn.get_graph(link_weight=weight).to_undirected())

    if sources is None:
        sources = set(wn.reservoir_name_list)
    if targets is None:
        targets = set(wn.junction_name_list)

    # 存储所有边的端点和权值
    edges_weight_dict = calculate_edges_weight(wG)

    # 计算每个需求节点与各个水源的K条最短路径
    if k is None:
        k = int(wn.num_nodes * 0.15)
    all_paths = list()
    for t in targets:
        for s in sources:
            k_shortest_paths = list(islice(nx.shortest_simple_paths(wG, s, t), k))
            # 如果没有K条最短路径，用最后一条路径填充满K条
            if len(k_shortest_paths) < k:
                for i in range(k - len(k_shortest_paths)):
                    k_shortest_paths.append(k_shortest_paths[len(k_shortest_paths) - 1])
            all_paths.append(k_shortest_paths)

    # Net3
    supply_ratio_dict = {'River': 0.667, 'Lake': 0.333}
    total_demand = 0
    avg_demand = wntr.metrics.average_expected_demand(wn)
    for demand in avg_demand.values:
        if demand > 0:
            total_demand += demand

    betweenness_dict = dict()
    for k_paths in all_paths:
        target_node = k_paths[0][len(k_paths[0]) - 1]
        demand_ratio = avg_demand[target_node] / total_demand

        k_total_weight = 0
        for path in k_paths:
            resistance = calculate_path_resistance(path, edges_weight_dict)
            if resistance != 0:
                k_total_weight += 1.0 / resistance

        for path in k_paths:
            resistance = calculate_path_resistance(path, edges_weight_dict)
            path_ratio = 1
            if resistance != 0:
                path_ratio = (1.0 / resistance) / k_total_weight

            supply_ratio = 1 / len(sources)
            for source, ratio in supply_ratio_dict.items():
                if path[0] == source or path[len(path) - 1] == source:
                    supply_ratio = ratio

            path_weight = supply_ratio * demand_ratio * path_ratio
            for i in range(len(path) - 1):
                links1 = wn.get_links_for_node(path[i])
                links2 = wn.get_links_for_node(path[i + 1])
                link_name = set(links1).intersection(set(links2)).pop()
                if link_name in betweenness_dict.keys():
                    betweenness_dict.update(
                        {link_name: betweenness_dict[link_name] + path_weight})
                else:
                    betweenness_dict.update({link_name: path_weight})

    for p in wn.pipe_name_list:
        if p not in betweenness_dict:
            betweenness_dict.update({p: 0})

    return betweenness_dict


# 计算路径的阻力
def calculate_path_resistance(path, edges_weight_dict):
    resistance = 0
    for i in range(len(path) - 1):
        edge = path[i] + '_' + path[i + 1]
        if edge not in edges_weight_dict:
            edge = path[i + 1] + '_' + path[i]
        resistance += edges_weight_dict[edge]
    return resistance


# 计算所有边的权重
def calculate_edges_weight(G):
    edges_weight_dict = dict()
    for edge in G.edges.data('weight'):
        edge_name = edge[0] + '_' + edge[1]
        weight = 0
        if edge[2] is not None:
            weight = edge[2]
        edges_weight_dict.update({edge_name: weight})
    return edges_weight_dict


# 计算各管段故障引发的管网欠缺供应比例
def water_shortfall(wn, sources=None):
    cache_file = 'wn.pickle'
    ns.pickle_network(wn, cache_file)

    if sources is None:
        sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))

    total_expected_demand = cal_total_expected_demand(wn, sources)

    wsf_dict = dict()
    for name in wn.link_name_list:
        wn = ns.reload_network(cache_file)
        link = wn.get_link(name)
        if link.link_type == 'Valve':
            continue
        link.status = 0

        # # Reset the water network model
        # wn.reset_initial_values()
        #
        # # Add a control to close the pipe
        # act = wntr.network.controls.ControlAction(link, 'status',
        #                                           wntr.network.LinkStatus.Closed)
        # cond = wntr.network.controls.SimTimeCondition(wn, '=', '00:00:00')
        # ctrl = wntr.network.controls.Control(cond, act)
        # wn.add_control('close pipe ' + name, ctrl)

        # if link.link_type == 'Pump':
        #     continue
        # wn = wntr.morph.split_pipe(wn, name, name + 'B', name + 'node')
        # leak_node = wn.get_node(name + 'node')
        # leak_node.add_leak(wn, area=0.05, start_time=0 * 3600, end_time=12 * 3600)

        results = hydraulic_sim(wn)
        demand = results.node['demand']
        total_demand = cal_total_demand(demand, sources)

        # 24小时欠缺供应量比例
        wsf = 0
        if total_demand < total_expected_demand:
            wsf = 1 - total_demand / total_expected_demand
            print(name, wsf)
        wsf_dict.update({name: wsf})

    # 清理缓存
    os.remove(cache_file)
    return wsf_dict


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


# 计算各管段流量比例
def water_flowrate(wn):
    results = hydraulic_sim(wn)
    flowrates = results.link['flowrate']
    flowrate_dict = dict()

    # 计算24小时平均总流量
    total_flow = 0
    for c in flowrates.columns:
        total_flow += flowrates[c].abs().mean()

    # 各管段24小时平均流量比例
    for c in flowrates.columns:
        v = flowrates[c].abs().mean() / total_flow
        flowrate_dict.update({c: v})

    return flowrate_dict


# 计算各管段接近中心性
def edge_closeness(wn):
    G = nx.Graph(wn.get_graph().to_undirected())
    closeness_list = nx.closeness_centrality(nx.line_graph(G))
    closeness_dict = dict()
    for k, v in closeness_list.items():
        links1 = wn.get_links_for_node(k[0])
        links2 = wn.get_links_for_node(k[1])
        link_name = set(links1).intersection(set(links2)).pop()
        closeness_dict.update({link_name: v})
    return closeness_dict


# 计算边度
def edge_degree(wn):
    G = nx.Graph(wn.get_graph().to_undirected())
    degree_list = nx.degree_centrality(nx.line_graph(G))
    degree_dict = dict()
    for k, v in degree_list.items():
        links1 = wn.get_links_for_node(k[0])
        links2 = wn.get_links_for_node(k[1])
        link_name = set(links1).intersection(set(links2)).pop()
        degree_dict.update({link_name: v})
    return degree_dict


def water_quality(wn, sources=None):
    cache_file = 'wn.pickle'
    ns.pickle_network(wn, cache_file)
    age_inc_ratio_dict = dict()
    init_age_dict = water_quality_sim(wn)
    for link_name in wn.link_name_list:
        wn = ns.reload_network(cache_file)
        link = wn.get_link(link_name)
        if link.link_type == 'Valve':
            continue

        # Reset the water network model
        wn.reset_initial_values()

        # Add a control to close the pipe
        act = wntr.network.controls.ControlAction(link, 'status',
                                                  wntr.network.LinkStatus.Closed)
        cond = wntr.network.controls.SimTimeCondition(wn, '=', '00:00:00')
        ctrl = wntr.network.controls.Control(cond, act)
        wn.add_control('close pipe ' + link_name, ctrl)
        print('close pipe ' + link_name)
        age_dict = water_quality_sim(wn)
        expected_total_water_age = 0
        total_water_age = 0
        if sources is None:
            sources = set(wn.reservoir_name_list)
        for node in age_dict.keys():
            if node not in sources:
                if age_dict[node] > 0:
                    total_water_age += age_dict[node]
                    expected_total_water_age += init_age_dict[node]
        print(expected_total_water_age)
        print(total_water_age)
        age_inc_ratio = 0
        if total_water_age > expected_total_water_age:
            age_inc_ratio = total_water_age / expected_total_water_age - 1
        age_inc_ratio_dict.update({link_name: age_inc_ratio})
    print(age_inc_ratio_dict)
    return age_inc_ratio_dict


def water_quality_sim(wn):
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.quality.parameter = 'AGE'
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    age_24h = results.node['quality'].loc[:24 * 3600]
    average_age = age_24h.mean()  # /3600 convert to hours
    age_dict = {}
    for node, avg_age in average_age.items():
        age_dict[node] = avg_age
    return age_dict


if __name__ == '__main__':
    # inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\Net2.inp'
    # inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\Net3.inp'
    inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\数据\ZJ.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)

    # # 对重新设置管段长度，使之与节点坐标距离相一致
    # ntu.reset_all_pipe_length(wn=wn, scale=1000)
    # wntr.graphics.plot_network(wn)
    # plt.show()

    # # 针对Net2的数据预处理, 计算Net2各指标需要指定sources，targets
    # sources = set(wn.tank_name_list)
    # sources.add('1')
    # targets = set(wn.junction_name_list)
    # targets.remove('1')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # # 水龄增长比
    # water_age_inc_dict = water_quality(wn)
    # customize_drawing.plot_network(wn=wn,
    #                                node_size=20,
    #                                node_labels=True,
    #                                link_attribute=water_age_inc_dict,
    #                                link_width=5,
    #                                link_colorbar_label='water age increment ratio')
    # plt.show()

    # 欠缺供应量比
    # Net2
    # shortfall_dict = water_shortfall(wn, sources)

    start = time.time()
    shortfall_dict = water_shortfall(wn)
    end = time.time()
    print('shortfall', end - start)
    customize_drawing.plot_network(wn=wn,
                                   node_size=20,
                                   node_labels=True,
                                   link_attribute=shortfall_dict,
                                   link_width=5,
                                   link_colorbar_label='欠缺供应量比')
    plt.show()

    # 边介数
    betweenness_dict = edge_betweenness(wn)
    customize_drawing.plot_network(wn=wn,
                                   node_size=20,
                                   node_labels=True,
                                   link_attribute=betweenness_dict,
                                   link_width=5,
                                   link_colorbar_label='边介数')
    plt.show()

    # 边接近中心性
    closeness_dict = edge_closeness(wn)
    customize_drawing.plot_network(wn=wn,
                                   node_size=20,
                                   node_labels=True,
                                   link_attribute=closeness_dict,
                                   link_width=5,
                                   link_colorbar_label='边接近中心性')
    plt.show()

    # 边度
    degree_dict = edge_degree(wn)
    customize_drawing.plot_network(wn=wn,
                                   node_size=20,
                                   node_labels=True,
                                   link_attribute=degree_dict,
                                   link_width=5,
                                   link_colorbar_label='边度')
    plt.show()

    # 加权边介数
    k = 10
    start = time.time()
    weighted_betweenness_dict = weighted_betweenness(wn, k=k)
    end = time.time()
    print('weighted edge betweenness', end - start)
    customize_drawing.plot_network(wn=wn,
                                   node_size=20,
                                   node_labels=True,
                                   link_attribute=weighted_betweenness_dict,
                                   link_width=5,
                                   link_colorbar_label='加权边介数')
    plt.show()

    # 选择脆弱性最高的管段添加平行管段及可视化
    # link_range = list()
    # sorted_res = sorted(weighted_betweenness_dict.items(), key=lambda item: item[1])
    # link_range.append(sorted_res[0][1])
    # link_range.append(sorted_res[-1][1])
    # top_edge_dict = dict()
    # print(sorted_res)
    # num = int(len(sorted_res) * 0.05)
    # for item in sorted_res[-num:]:
    #     link = wn.get_link(item[0])
    #     start_node_name = link.start_node_name
    #     end_node_name = link.end_node_name
    #     top_edge_dict[(start_node_name, end_node_name)] = item[1] / 2
    #     weighted_betweenness_dict[item[0]] = item[1] / 2
    # print(top_edge_dict)
    # customize_drawing.plot_network(wn=wn,
    #                                node_size=20,
    #                                node_labels=True,
    #                                link_attribute=weighted_betweenness_dict,
    #                                link_width=3,
    #                                link_colorbar_label='加权边介数', link_range=link_range,
    #                                curved_links=top_edge_dict.keys(),
    #                                curved_links_attribute=top_edge_dict.values())
    # plt.show()
