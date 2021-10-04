import wntr
import geatpy as ea
import os
import csv
import matplotlib.pyplot as plt
from pipe_moo_problem_mp import PipeMooProblem
import network_serialization as ns
import network_topology_utils as ntu
import prophet_population2 as pp


if __name__ == '__main__':
    inp_file = r'C:\Users\29040\Desktop\WNTR-0.2.2\examples\networks\Net2.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)

    # 对重新设置管段长度，使之与节点坐标距离相一致
    # Net3 scale 800; Net2 scale 150; ZJ scale 700
    scale = 150
    ntu.reset_all_pipe_length(wn=wn, scale=scale)

    """===============================实例化问题对象============================"""
    problem = PipeMooProblem(wn=wn, scale=scale)  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'  # 编码方式
    NIND = 200  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                      problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field,
                               NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """=================================算法参数设置============================"""
    nsga2_algorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    nsga2_algorithm.mutOper.Pm = 0.05  # 修改变异算子的变异概率
    nsga2_algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    nsga2_algorithm.MAXGEN = 500  # 最大进化代数
    nsga2_algorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
    nsga2_algorithm.verbose = False  # 设置是否打印输出日志信息
    nsga2_algorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================根据先验知识创建先知种群========================"""
    size = 100  # 经验种群规模
    prophetChrom = pp.get_prophet_population(problem, size)  # 获取经验种群
    prophetPop = ea.Population(Encoding, Field, size, prophetChrom)  # 实例化种群对象
    nsga2_algorithm.call_aimFunc(prophetPop)  # 计算先知种群的目标函数值及约束（假如有约束）
    """==========================调用算法模板进行种群进化=========================
        调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
        NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
        详见Population.py中关于种群类的定义。
        """
    [NDSet, population] = nsga2_algorithm.run(prophetPop)  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % nsga2_algorithm.passTime)
    print('非支配个体数：%d 个' %
          NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')

    diameters = problem.diameters
    edges = problem.edges

    sol_file = open(r'D:\Projects\reliability\Result\solution.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(sol_file)
    csv_writer.writerow(edges)
    phen = csv.reader(open(r'D:\Projects\reliability\Result\Phen.csv', 'r'))
    for row in phen:
        diameter_index = list()
        solution = list()
        wn = ns.reload_network(problem.network_name)
        for i in range(len(row)):
            index = int(float(row[i]))
            diameter_index.append(index)
            solution.append(diameters[index])

        # # 方案图形化展示
        # ntu.update_network(wn, edges, diameters, diameter_index)
        # wntr.graphics.plot_network(wn)
        # plt.show()

        # 写入方案
        csv_writer.writerow(solution)

    sol_file.close()
    # 清理缓存
    os.remove(problem.network_name)
