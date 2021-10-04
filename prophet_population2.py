import random
import network_serialization as ns
import numpy as np


def get_prophet_population(problem, size):
    max_degree = 4
    candidate_edges = problem.edges
    var_range = [i for i in range(len(problem.diameters))]
    population = list()
    # 候选节点的候选邻接边集
    edges_for_node = dict()
    for edge in candidate_edges:
        for node in edge:
            if node in edges_for_node.keys():
                edges = edges_for_node[node]
                edges.append(edge)
                edges_for_node.update({node: edges})
            else:
                edges = list()
                edges.append(edge)
                edges_for_node.update({node: edges})
    # 初始化size个个体
    for i in range(size):
        wn = ns.reload_network(problem.network_name)
        G = wn.get_graph().to_undirected()
        degree_dict = dict(G.degree())
        nodes = list(edges_for_node.keys())
        # 候选节点乱序排列
        random.shuffle(nodes)
        # 染色体初始化0值
        chromosome = [0 for i in range(len(candidate_edges))]
        # 每个节点尝试随机增加一定数量的边，增加后各节点度要仍不大于4
        for j in range(len(nodes)):
            start_node_name = nodes[j]
            begin = degree_dict[start_node_name]
            if begin >= max_degree:
                continue
            end = random.randint(begin, max_degree)
            # 如过节点随机增加的边数大于其总候选邻接边数， 以总候选邻接边数为准
            max_edge_num = len(edges_for_node[start_node_name])
            if end > begin + max_edge_num:
                end = begin + max_edge_num
            # 尝试增加边
            for k in range(begin, end):
                # 尝试抽取一个度小于4的节点，最多尝试次数等于总候选邻接边数
                end_node_name = random_choice_end_node(start_node_name, edges_for_node, degree_dict, max_degree,
                                                       max_edge_num*2)
                # 获取边在染色体中的索引位置
                index = locate_in_chromosome(start_node_name, end_node_name, candidate_edges)
                # 两个节点还未有边直接相连则添加
                if index != -1 and G.has_edge(start_node_name, end_node_name) is False:
                    # 随机选择边的管径并更新染色体对应基因位
                    chromosome[index] = random.choice(var_range)
                    # 更新节点度
                    degree_dict.update({start_node_name: degree_dict[start_node_name] + 1})
                    degree_dict.update({end_node_name: degree_dict[end_node_name] + 1})
                    # 更新网络拓扑
                    name = start_node_name + "," + end_node_name
                    wn.add_pipe(name=name, start_node_name=start_node_name, end_node_name=end_node_name)
        print(np.count_nonzero(chromosome))
        population.append(chromosome)
    return np.vstack(population)


# 随机选择一个可行终点
def random_choice_end_node(start_node_name, edges_for_node, degree_dict, max_degree, max_iter):
    end_node_name = ''
    for i in range(max_iter):
        edge = random.choice(edges_for_node[start_node_name])
        if edge[0] == start_node_name:
            node_name = edge[1]
        else:
            node_name = edge[0]
        if degree_dict[node_name] < max_degree:
            end_node_name = node_name
            break
    return end_node_name


# 根据两个端点获取边在染色体中的索引
def locate_in_chromosome(start_node_name, end_node_name, edges):
    index = -1
    if start_node_name == '' or end_node_name == '':
        return index
    for i in range(len(edges)):
        if start_node_name in edges[i] and end_node_name in edges[i]:
            index = i
            break
    return index
