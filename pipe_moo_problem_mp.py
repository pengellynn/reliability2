import geatpy as ea
import network_topology_utils as ntu
import network_serialization as ns
import pipe_evaluation as pe
import optimization_objective as obj
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
import time
import wntr


class PipeMooProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, wn, scale, M=3):
        self.edges = ntu.get_candidate_edges(wn)
        self.cross_pipes = ntu.cal_cross_pipes(wn, self.edges)
        self.degree_list = wn.get_graph().to_undirected().degree()
        name = 'pipe_moo_problem'  # 初始化name（函数名称，可以随意设置）
        Dim = len(self.edges)  # 初始化Dim（决策变量维数）
        maxormins = [1, -1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [2] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub,
                            lbin, ubin)

        self.diameters = [0, 0.2032, 0.3048]  # , 0.4064, 0.508
        self.prices = [0, 433.93, 720.14]  # ,962.99, 1197.29
        # self.diameters = [0, 200, 300, 400, 500, 600, 700]
        # self.prices = [0, 433.93, 720.14, 962.99, 1197.29, 1562.66, 1910.62]
        self.network_name = "network_temp"
        ns.pickle_network(wn, self.network_name)
        self.num_cores = cpu_count()
        self.pool = Pool(self.num_cores)
        self.scale = scale
        self.gen = 0

    def aimFunc(self, pop):  # 目标函数
        self.gen = self.gen + 1
        print('代数：', self.gen)
        start_time = time.time()
        var_matrix = pop.Phen  # 得到决策变量矩阵
        var_matrix = var_matrix.astype(int)

        # 多进程并行计算目标函数
        results = list()
        for i in range(self.num_cores):
            res = self.pool.apply_async(sub_aimFunc,
                                        args=(self.edges, self.diameters,
                                              self.prices, var_matrix,
                                              self.num_cores, i, self.scale))
            results.append(res)
        pop.ObjV = np.vstack([v.get() for v in results])

        # 利用可行性法则处理约束条件
        cv = list()
        for row in var_matrix:
            degree_dict = dict(self.degree_list)
            degree_exceed_count = 0
            cross_count = 0
            r = list()
            for i in range(len(row)):
                # 判断管段是否存在交叉
                for j in self.cross_pipes[i]:
                    if row[i] * row[j] > 0:
                        cross_count += 1
                # 判断节点度是否超过4
                if row[i] > 0:
                    n1 = self.edges[i][0]
                    n2 = self.edges[i][1]
                    if degree_dict[n1] == 4:
                        degree_exceed_count += 1
                    degree_dict.update({n1: degree_dict[n1] + 1})
                    if degree_dict[n2] == 4:
                        degree_exceed_count += 1
                    degree_dict.update({n2: degree_dict[n2] + 1})
            r.append(cross_count)
            r.append(degree_exceed_count)
            cv.append(r)
        print(cv)
        pop.CV = np.vstack(cv)
        end_time = time.time()
        print(end_time - start_time)


def sub_aimFunc(edges, diameters, prices, data, num, index, scale):
    size = math.ceil(len(data) / num)
    start = size * index
    end = start + size
    f1 = list()
    f2 = list()
    f3 = list()
    network_name = "network_temp"
    for row in data[start:end]:
        wn = ns.reload_network(network_name)
        ntu.update_network(wn, scale, edges, diameters, row)
        cost = obj.cost(wn, edges, diameters, prices, row)
        f1.append(cost)

        # # 针对Net2的数据预处理
        # sources = set(wn.tank_name_list)
        # sources.add('1')
        # targets = set(wn.junction_name_list)
        # targets.remove('1')
        # k = 5
        # betweenness_dict = pe.weighted_betweenness(wn, k, sources, targets)
        betweenness_dict = pe.weighted_betweenness(wn)
        entropy = obj.weighted_betweenness_entropy(betweenness_dict)
        f2.append(entropy)

        wn.options.time.duration = 12 * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600
        wn.options.hydraulic.required_pressure = 10  # 30 psi = 21.097 m
        wn.options.hydraulic.minimum_pressure = 3  # 5 psi = 3.516 m
        wn.options.hydraulic.demand_model = 'PDA'
        sim = wntr.sim.WNTRSimulator(wn=wn)
        results = sim.run_sim(convergence_error=False)
        flowrate = results.link['flowrate'].loc[12 * 3600, :]
        G = wn.get_graph(link_weight=flowrate)

        # # Net2
        # entropy, system_entropy = obj.entropy(G, sources=sources, sinks=targets)
        entropy, system_entropy = obj.entropy(G)
        f2.append(system_entropy)
        wn.reset_initial_values()

        # # Net2
        # todini = obj.todini(wn, sources, targets)
        todini  = obj.todini(wn)
        f3.append(todini)
    f1 = np.array(f1).reshape((-1, 1))
    f2 = np.array(f2).reshape((-1, 1))
    f3 = np.array(f3).reshape((-1, 1))
    return np.hstack([f1, f2, f3])
