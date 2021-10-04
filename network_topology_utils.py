import networkx as nx
import math
import wntr
import matplotlib.pyplot as plt


# 获取网络当前的所有非连接边, 并排除不满足约束条件的边
def get_candidate_edges(wn):
    G = wn.get_graph().to_undirected()
    sources = set.union(set(wn.tank_name_list), set(wn.reservoir_name_list))
    edges = list()
    for ne in nx.non_edges(G):
        # 排除水源节点互连的边
        if ne[0] in sources and ne[1] in sources:
            continue
        # 排除两端的节点度已经大于等于4的边
        degree_dict = dict(G.degree())
        if degree_dict[ne[0]] >= 4 or degree_dict[ne[1]] >= 4:
            continue
        # 排除与现有布局中的管段相交叉的边
        flag = False
        a, b = get_nodes_coordinates(wn, ne)
        for e in nx.edges(G):
            c, d = get_nodes_coordinates(wn, e)
            if is_cross(a, b, c, d):
                flag = True
                break
        if flag is False:
            edges.append(ne)
    return edges


# 获取边的两个端点的坐标
def get_nodes_coordinates(wn, edge):
    start_node_name = edge[0]
    end_node_name = edge[1]
    start_node = wn.get_node(start_node_name)
    end_node = wn.get_node(end_node_name)
    start_xy = start_node.coordinates
    end_xy = end_node.coordinates
    return start_xy, end_xy


# 计算每一管段其交叉管段集
def cal_cross_pipes(wn, edges):
    cross_pipes = list()
    for i in range(len(edges)):
        a, b = get_nodes_coordinates(wn, edges[i])
        pipes = list()
        for j in range(len(edges)):
            if j != i:
                c, d = get_nodes_coordinates(wn, edges[j])
                if is_cross(a, b, c, d):
                    pipes.append(j)
        cross_pipes.append(pipes)
    return cross_pipes


# 判断线段ab是否与线段cd相交
def is_cross(a, b, c, d):
    # 快速排斥实验
    if (min(a[0], b[0]) >= max(c[0], d[0]) or min(a[1], b[1]) >= max(c[1], d[1])
            or max(a[0], b[0]) <= min(c[0], d[0])
            or max(a[1], b[1]) <= min(c[1], d[1])):
        return False

    # 排除平行管段
    if (min(a[0], b[0]) == min(c[0], d[0]) and max(a[0], b[0]) == max(c[0], d[0])
            and min(a[1], b[1]) == min(c[1], d[1]) and max(a[1], b[1]) == max(c[1], d[1])):
        return True

    # 跨立实验
    # 如果两条线段相交，那么必须跨立，就是以一条线段为标准，另一条线段的两端点一定在这条线段的两侧
    # 也就是说a b两点在线段cd的两侧，c d两点在线段ab的两侧
    # (ca x cd)·(cb x cd)<0 则说明ca cb先对于cd的方向不同，则ab在线段cd的两侧
    # (ac x ab)·(ad x ab)<0 则说明ac ad先对于ab的方向不同，则cd在线段ab的两侧
    u = (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])
    v = (d[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (d[1] - a[1])
    w = (a[0] - c[0]) * (d[1] - c[1]) - (d[0] - c[0]) * (a[1] - c[1])
    z = (b[0] - c[0]) * (d[1] - c[1]) - (d[0] - c[0]) * (b[1] - c[1])
    return u * v < 0 and w * z < 0


# 新增管段集,更新网络的拓扑
def update_network(wn, scale, edges, dia_val, dia_idx):
    for i in range(len(dia_idx)):
        start_node_name = edges[i][0]
        end_node_name = edges[i][1]
        name = start_node_name + "," + end_node_name
        start_xy, end_xy = get_nodes_coordinates(wn, edges[i])
        length = cal_length(start_xy, end_xy, scale)
        diameter = dia_val[dia_idx[i]]
        if diameter > 0:
            wn.add_pipe(name=name,
                        start_node_name=start_node_name,
                        end_node_name=end_node_name,
                        length=length,
                        diameter=diameter)
    # wntr.graphics.plot_network(wn)
    # plt.show()


def reset_all_pipe_length(wn, scale):
    for name in wn.pipe_name_list:
        pipe = wn.get_link(name)
        start_node = wn.get_node(pipe.start_node_name)
        end_node = wn.get_node(pipe.end_node_name)
        start_xy = start_node.coordinates
        end_xy = end_node.coordinates
        length = cal_length(start_xy, end_xy, scale)
        pipe.length = length


def cal_length(start_xy, end_xy, scale):
    length = math.sqrt(
        pow(start_xy[1] / 3.2808399 - end_xy[1] / 3.2808399, 2) + pow(start_xy[0] / 3.2808399 - end_xy[0] / 3.2808399,
                                                                      2)) * scale
    return length


if __name__ == '__main__':
    # inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net3.inp'
    inp_file = r'C:\Users\29040\Desktop\WDN\可靠性评价与优化\管段优化\数据\ZJ.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)

    # 测试候选管段集
    graph = nx.Graph(wn.get_graph().to_undirected())
    edges = [e for e in nx.non_edges(graph)]
    candidate_edges = get_candidate_edges(wn)
    print(len(edges))
    print("----------------")
    print(len(candidate_edges))

    wn = wntr.network.WaterNetworkModel(inp_file)
    for i in range(len(edges)):
        start_node_name = edges[i][0]
        end_node_name = edges[i][1]
        name = start_node_name + "," + end_node_name
        wn.add_pipe(name=name,
                    start_node_name=start_node_name,
                    end_node_name=end_node_name)
    wntr.graphics.plot_network(wn)
    plt.show()

    wn = wntr.network.WaterNetworkModel(inp_file)
    for i in range(len(candidate_edges)):
        start_node_name = candidate_edges[i][0]
        end_node_name = candidate_edges[i][1]
        name = start_node_name + "," + end_node_name
        wn.add_pipe(name=name,
                    start_node_name=start_node_name,
                    end_node_name=end_node_name)
    wntr.graphics.plot_network(wn)
    plt.show()

    # # 测试线段相交
    # reset_all_pipe_length(wn=wn, scale=1000)
    # G = wn.get_graph().to_undirected()
    # for e in nx.edges(G):
    #     a, b = get_nodes_coordinates(wn, e)
    #     for e2 in nx.edges(G):
    #         c, d = get_nodes_coordinates(wn, e2)
    #         if is_cross(a, b, c, d):
    #             print(e, e2)
    # a = wn.get_node('16').coordinates
    # b = wn.get_node('19').coordinates
    # c = wn.get_node('19').coordinates
    # d = wn.get_node('32').coordinates
    # res = is_cross(a, b, c, d)
    # print(res)

    # # 测试管段长度与节点坐标是否一致
    # a = wn.get_node('5').coordinates
    # b = wn.get_node('6').coordinates
    # links1 = wn.get_links_for_node('5')
    # links2 = wn.get_links_for_node('6')
    # link_name = set(links1).intersection(set(links2)).pop()
    # pipe = wn.get_link(link_name)
    # length = math.sqrt(pow(a[1]/3.2808399 - b[1]/3.2808399, 2) + pow(a[0]/3.2808399 - b[0]/3.2808399, 2))
    # print(length)
    # print('real_length', pipe.length)

    # 测试每根管段的交叉管段集
    # cross_pipes = cal_cross_pipes(wn, edges)
    # print(edges)
    # print(cross_pipes)
    # for i in range(len(cross_pipes)):
    #     if len(cross_pipes[i]) > 0:
    #         print(edges[i])
    #         print([edges[j] for j in cross_pipes[i]])
    #         print("-------------")
