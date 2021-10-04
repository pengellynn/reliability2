import random
import network_serialization as ns
import numpy as np


def get_prophet_population(problem, size):
    max_degree = 4
    candidate_edges = problem.edges
    var_range = [i for i in range(len(problem.diameters))]
    population = list()
    # 初始化size个个体
    for i in range(size):
        wn = ns.reload_network(problem.network_name)
        graph = wn.get_graph().to_undirected()
        nodes = list()
        degree_dict = dict(graph.degree())
        for name, degree in degree_dict.items():
            if degree < max_degree:
                nodes.append(name)
        # 度小于4的节点乱序排列
        random.shuffle(nodes)
        # 染色体初始化0值
        chromosome = [0 for i in range(len(candidate_edges))]
        total_exp = 0
        total_loc = 0
        total_rel = 0
        # 每个节点尝试与其后面节点随机增加一定数量的边
        for j in range(len(nodes)-1):
            start_node_name = nodes[j]
            begin = degree_dict[start_node_name]
            if begin >= max_degree:
                continue
            # 度小于4的节点随机增加一定数量的边，增加后各节点度要仍不大于4
            end = random.randint(begin, max_degree)
            count = 0
            for k in range(begin, end):
                # print(end - begin)
                # 尝试抽取一个度小于4的节点，最多尝试n次
                end_node_name = ''
                n = len(nodes)
                while n > 0:
                    end_node_name = random.choice(nodes)
                    if degree_dict[end_node_name] < 4 and end_node_name != start_node_name:
                        total_exp +=1
                        break
                    n -= 1
                # 获取边在染色体中的索引位置
                index = locate_in_chromosome(start_node_name, end_node_name, candidate_edges)
                if index != -1:
                    total_loc += 1
                    # 两个节点还未有边直接相连则添加
                    if graph.has_edge(start_node_name, end_node_name) is False:
                        # 随机选择边的管径并更新染色体对应基因位
                        chromosome[index] = random.choice(var_range)
                        # 更新节点度
                        degree_dict.update({start_node_name: degree_dict[start_node_name] + 1})
                        degree_dict.update({end_node_name: degree_dict[end_node_name] + 1})
                        # 更新网络拓扑
                        name = start_node_name + "," + end_node_name
                        wn.add_pipe(name=name, start_node_name=start_node_name, end_node_name=end_node_name)
                        total_rel += 1
        print(np.count_nonzero(chromosome))
        population.append(chromosome)
    return np.vstack(population)


# 根据两个端点获取边在染色体中的索引
def locate_in_chromosome(start_node_name, end_node_name, edges):
    index = -1
    for i in range(len(edges)):
        if start_node_name in edges[i] and end_node_name in edges[i]:
            index = i
            break
    return index
