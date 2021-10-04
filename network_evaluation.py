import wntr
import networkx as nx
import pandas as pd
import numpy as np
from itertools import islice


class Evaluation:
    max_path_distance = 0

    def __init__(self, wn):
        self.wn = wn
        self.source_list = self.get_source_list()
        self.target_list = self.get_target_list()
        self.wG = self.build_weighted_graph()
        self.edges_weight_dict = self.get_all_edge_weight()
        self.k = 10
        self.k_paths_dict = self.get_all_k_paths()
        # self.supply_ratio_dict = {'River': 0.667, 'Lake': 0.333}
        self.supply_ratio_dict = {}
        self.avg_demand_dict = wntr.metrics.average_expected_demand(wn)
        self.total_demand = self.calculate_total_demand()
        if Evaluation.max_path_distance == 0:
            Evaluation.max_path_distance = self.get_max_path_distance()

    # 网络最大路径距离
    def get_max_path_distance(self):
        max_distance = 0
        for paths in self.k_paths_dict.values():
            for path in paths:
                distance = self.calculate_path_distance(path)
                if distance > max_distance:
                    max_distance = distance
        print('max_path_distance', max_distance * 2)
        return max_distance * 2

    # 获取所有源节点
    def get_source_list(self):
        source_list = list()
        for r in self.wn.reservoir_name_list:
            source_list.append(r)
        # for t in self.wn.tank_name_list:
        #     source_list.append(t)
        return source_list

    # 获取所有非源节点（junction）
    def get_target_list(self):
        # target_list = list()
        # for name in self.wn.junction_name_list:
        #     node = self.wn.get_node(name)
        #     if node.base_demand > 0:
        #         target_list.append(name)
        # return target_list
        return self.wn.junction_name_list

    # 构建加权无向图，以边的阻力为权
    def build_weighted_graph(self):
        # 移除所有control
        for ctl in self.wn.control_name_list:
            self.wn.remove_control(ctl)
        length = self.wn.query_link_attribute('length')
        diameter = self.wn.query_link_attribute('diameter')
        roughness = self.wn.query_link_attribute('roughness')
        w = roughness * length / pow(diameter, 1)
        for valve in self.wn.valve_name_list:
            w[valve] = 0
        # pump权值设为0, 即阻力为0
        index = list(w.index)
        values = list(w.values)
        for pump in self.wn.pump_name_list:
            index.append(pump)
            values.append(0)
        weight = pd.Series(data=values, index=index)
        return nx.Graph(self.wn.get_graph(link_weight=weight).to_undirected())

    # 计算所有节点的k最短路径
    def get_all_k_paths(self):
        k_paths_dict = dict()
        for t in self.target_list:
            k_paths = list()
            for s in self.source_list:
                if nx.has_path(G=self.wG, source=s, target=t):
                    try:
                        k_shortest_paths = list(
                            islice(
                                nx.shortest_simple_paths(G=self.wG,
                                                         source=s,
                                                         target=t,
                                                         weight='weight'),
                                self.k))
                        for path in k_shortest_paths:
                            k_paths.append(path)
                        # 如果没有K条最短路径，用最后一条路径填充满K条
                        n = len(k_shortest_paths)
                        if n < self.k:
                            for i in range(self.k - n):
                                k_paths.append(k_shortest_paths[n - 1])
                    except Exception as e:
                        print(e)
            k_paths_dict.update({t: k_paths})
        return k_paths_dict

    # 节点的平均效率
    def source_avg_efficiency(self):
        nodes = self.wn.junction_name_list
        efficiency_list = list()
        for node in nodes:
            min_distance = Evaluation.max_path_distance
            for s in self.source_list:
                if nx.has_path(G=self.wG, source=s, target=node):
                    distance = nx.shortest_path_length(G=self.wG,
                                                       source=s,
                                                       target=node,
                                                       weight='weight')
                    if distance < min_distance:
                        min_distance = distance
            if min_distance != 0:
                efficiency_list.append(1 / min_distance)
        avg_efficiency = 1 / Evaluation.max_path_distance
        if len(efficiency_list) != 0:
            avg_efficiency = np.average(efficiency_list)
        else:
            print('efficiency_list == null')
        return avg_efficiency

    # 包含水源的最大连通子图大小
    def source_connected_component(self):
        max_node_num = 0
        for s in self.source_list:
            component = nx.node_connected_component(G=self.wG, n=s)
            node_num = len(component)
            if node_num > max_node_num:
                max_node_num = node_num
        return max_node_num

    # 获取所有边的权重
    def get_all_edge_weight(self):
        weight_dict = dict()
        for edge in self.wG.edges.data('weight'):
            edge_name = edge[0] + '_' + edge[1]
            weight = 0
            if edge[2] is not None:
                weight = edge[2]
            weight_dict.update({edge_name: weight})
        return weight_dict

    # 计算路径距离（阻力）
    def calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            edge = path[i] + '_' + path[i + 1]
            if edge not in self.edges_weight_dict:
                edge = path[i + 1] + '_' + path[i]
            distance += self.edges_weight_dict[edge]
        return distance

    # 计算总需求
    def calculate_total_demand(self):
        total_demand = 0
        for demand in self.avg_demand_dict.values:
            total_demand += demand
        return total_demand
