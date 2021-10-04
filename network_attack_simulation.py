import wntr
import pipe_evaluation as pe
import network_evaluation as ne
import matplotlib.pyplot as plt
import random


def attack(wn, seq):
    network = ne.Evaluation(wn)
    origin_network_size = len(wn.node_name_list)
    origin_network_source_avg_efficiency = network.source_avg_efficiency()
    print('origin_network_size', origin_network_size)
    print('origin_network_source_avg_efficiency',
          origin_network_source_avg_efficiency)

    network_connected_component_list = list()
    network_source_avg_efficiency = list()

    step = 0.01
    lower = 0
    upper = 10
    begin = 0
    for i in range(lower, upper):
        count = int(step * len(seq))
        end = begin + count
        for j in range(begin, end):
            print(begin, 'remove', j)
            wn.remove_link(seq[j])
        begin = end
        network = ne.Evaluation(wn)
        source_connected_component = network.source_connected_component(
        ) / origin_network_size
        network_connected_component_list.append(source_connected_component)
        source_avg_efficiency = network.source_avg_efficiency(
        ) / origin_network_source_avg_efficiency
        network_source_avg_efficiency.append(source_avg_efficiency)
    print('最大连通子图规模比', network_connected_component_list)
    print('平均效率', network_source_avg_efficiency)

    # 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # x = [step * i for i in range(lower, upper)]
    x = [i for i in range(lower, upper)]

    # 含水源的最大连通子图变化图
    y6 = network_connected_component_list
    plt.title('含水源的最大连通子图变化图')
    plt.xlabel('移除管段数')
    plt.ylabel('含水源最大连通子图规模比')
    plt.plot(x, y6, 'o')
    plt.show()

    # 平均效率
    y9 = network_source_avg_efficiency
    plt.title('平均效率变化图')
    plt.xlabel('移除管段数')
    plt.ylabel('平均最短效率比')
    plt.plot(x, y9, 'o')
    plt.show()
    print('network_source_avg_efficiency', y9)


# 随机攻击
def random_attack(wn):
    pipe_list = wn.pipe_name_list
    random.shuffle(pipe_list)
    print(pipe_list)
    attack(wn, pipe_list)


# 加权介数攻击
def wb_attack(wn, k):
    w_betweenness_dict = pe.weighted_betweenness(wn, k)
    wb_list = sort(w_betweenness_dict)
    print(wb_list)
    attack(wn, wb_list)


# 普通介数攻击
def betweenness_attack(wn):
    betweenness_dict = pe.edge_betweenness(wn)
    b_list = sort(betweenness_dict)
    print(b_list)
    attack(wn, b_list)


# 边度攻击
def edge_degree_attack(wn):
    edge_degree_dict = pe.edge_degree(wn)
    d_list = sort(edge_degree_dict)
    print(d_list)
    attack(wn, d_list)


# 边接近中心性攻击
def edge_closeness_attack(wn):
    edge_closeness_dict = pe.edge_closeness(wn)
    c_list = sort(edge_closeness_dict)
    print(c_list)
    attack(wn, c_list)


# 根据字典的val排序，返回排序后的key的list
def sort(data_dict):
    sorted_list = sorted(data_dict.items(),
                         key=lambda item: item[1],
                         reverse=True)
    res = list()
    for it in sorted_list:
        res.append(it[0])
    return res


if __name__ == '__main__':
    # inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net3.inp'
    inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\管段优化\数据\ZJ.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)
    # random_attack(wn)
    # wb_attack(wn, 10)
    # betweenness_attack(wn)
    # edge_degree_attack(wn)
    edge_closeness_attack(wn)
