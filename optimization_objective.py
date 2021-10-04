import pandas as pd
import numpy as np
import wntr
import math
import networkx as nx


# 计算添加的管段的总成本
def cost(wn, edges, diameters, prices, diameter_index):
    total_cost = 0
    for i in range(len(diameter_index)):
        index = diameter_index[i]
        if diameters[index] > 0:
            start_node_name = edges[i][0]
            end_node_name = edges[i][1]
            name = start_node_name + "," + end_node_name
            pipe = wn.get_link(name)
            total_cost += prices[index] * pipe.length
    return total_cost


# 计算网络的加权介数熵
def weighted_betweenness_entropy(betweenness_dict):
    total = 0
    for val in betweenness_dict.values():
        total += val
    entropy = 0
    for val in betweenness_dict.values():
        if val != 0:
            entropy += float(val) / total * math.log(float(val) / total)
    # 归一化
    # entropy = entropy / math.log(1.0/len(betweenness_dict.keys()))
    return -entropy


def weighted_betweenness_entropy2(wn, betweenness_dict):
    entropy = 0
    total = 0
    for val in betweenness_dict.values():
        total += val
    for junc in wn.junction_name_list:
        node_total = 0
        for link in wn.get_links_for_node(junc):
            node_total += betweenness_dict[link]
        if node_total != 0:
            entropy += float(node_total) / total * math.log(float(node_total) / total)
    return -entropy


# 计算网络的todini
def todini(wn, sources=None, targets=None):
    wn.options.time.duration = 12 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    # ZJ required_pressure 1， minimum_pressure 0.06， threshold 0.06
    wn.options.hydraulic.required_pressure = 10  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3  # 5 psi = 3.516 m
    wn.options.hydraulic.demand_model = 'PDA'
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    head = results.node['head']
    pressure = results.node['pressure']
    demand = results.node['demand']
    pump_flowrate = results.link['flowrate'].loc[:, wn.pump_name_list]
    threshold = 3

    if sources is None:
        sources = wn.reservoir_name_list
    if targets is None:
        targets = wn.junction_name_list
    todini = todini_index(head, pressure, demand, pump_flowrate, wn,
                          threshold, sources, targets)
    return np.mean(todini)


# todini指数的计算
def todini_index(head, pressure, demand, flowrate, wn, Pstar, sources, targets):
    POut = {}
    PExp = {}
    PInRes = {}
    PInPump = {}
    PInTank = {}

    time = head.index

    for name in wn.tank_name_list:
        tank_in_q = list()
        tank_out_q = list()
        h = np.array(head.loc[:, name])  # m
        p = np.array(pressure.loc[:, name])
        e = h - p  # m
        q = np.array(demand.loc[:, name])
        for v in q:
            if v > 0:
                tank_in_q.append(v)
                tank_out_q.append(0)
            else:
                tank_in_q.append(0)
                tank_out_q.append(-v)
        tank_in_q = np.array(tank_in_q)
        tank_out_q = np.array(tank_out_q)
        POut[name] = tank_in_q * h
        PExp[name] = tank_in_q * (Pstar + e)

        PInTank[name] = tank_out_q * h

    for name in targets:
        h = np.array(head.loc[:, name])  # m
        p = np.array(pressure.loc[:, name])
        e = h - p  # m
        q = np.array(demand.loc[:, name])  # m3/s
        POut[name] = q * h
        PExp[name] = q * (Pstar + e)

    for name in sources:
        H = np.array(head.loc[:, name])  # m
        Q = np.array(demand.loc[:, name])  # m3/s
        PInRes[name] = -Q * H  # switch sign on Q.

    for name, link in wn.links(wntr.network.Pump):
        start_node = link.start_node_name
        end_node = link.end_node_name
        h_start = np.array(head.loc[:, start_node])  # (m)
        h_end = np.array(head.loc[:, end_node])  # (m)
        h = h_start - h_end  # (m)
        q = np.array(flowrate.loc[:, name])  # (m^3/s)
        PInPump[name] = q * (
            abs(h))  # assumes that pumps always add energy to the system

    print('POut', sum(POut.values()))
    print('PExp', sum(PExp.values()))
    print('PInRes', sum(PInRes.values()))
    todini = (sum(POut.values()) - sum(PExp.values())) / \
             (sum(PInRes.values()) + sum(PInPump.values()) + sum(PInTank.values()) - sum(PExp.values()))

    todini = pd.Series(data=todini.tolist(), index=time)

    return todini


def entropy(G, sources=None, sinks=None):
    """
    Compute entropy, equations from [AwGB90]_.

    Entropy is a measure of uncertainty in a random variable.
    In a water distribution network model, the random variable is
    flow in the pipes and entropy can be used to measure alternate flow paths
    when a network component fails.  A network that carries maximum entropy
    flow is considered reliable with multiple alternate paths.

    Parameters
    ----------
    G : NetworkX or WNTR graph
        Entropy is computed using a directed graph based on pipe flow direction.
        The 'weight' of each link is equal to the flow rate.

    sources : list of strings, optional (default = all reservoirs)
        List of node names to use as sources.

    sinks : list of strings, optional (default = all nodes)
        List of node names to use as sinks.

    Returns
    -------
    A tuple which includes:
        - A pandas Series that contains entropy for each node
        - System entropy (float)
    """

    if G.is_directed() is False:
        return

    if sources is None:
        sources = [key for key, value in nx.get_node_attributes(G, 'type').items() if value == 'Reservoir']

    if sinks is None:
        sinks = G.nodes()

    S = {}
    Q = {}
    for nodej in sinks:
        if nodej in sources:
            S[nodej] = 0  # nodej is the source
            continue

        # sp = []  # simple path
        # if G.nodes[nodej]['type'] == 'Junction':
        #     for source in sources:
        #         if nx.has_path(G, source, nodej):
        #             simple_paths = nx.all_simple_paths(G, source, target=nodej)
        #             sp = sp + ([p for p in simple_paths])
        #             # all_simple_paths was modified to check 'has_path' in the
        #             # loop, but this is still slow for large networks
        #             # what if the network was skeletonized based on series pipes
        #             # that have the same flow direction?
        #             # what about duplicating paths that have pipes in series?
        #         # print j, nodeid, len(sp)
        #
        # if len(sp) == 0:
        #     S[nodej] = np.nan  # nodej is not connected to any sources
        #     continue
        #
        # # "dtype=object" is needed to create an array from a list of lists with differnet lengths
        # sp = np.array(sp, dtype=object)

        # Uj = set of nodes on the upstream ends of links incident on node j
        Uj = G.predecessors(nodej)
        # qij = flow in link from node i to node j
        qij = []
        # aij = number of equivalnet independent paths through the link from node i to node j
        aij = []
        for nodei in Uj:
            # mask = np.array([nodei in path for path in sp])
            # # NDij = number of paths through the link from node i to node j
            # NDij = sum(mask)
            # if NDij == 0:
            #     continue
            # temp = sp[mask]
            # # MDij = links in the NDij path
            # MDij = [(t[idx], t[idx + 1]) for t in temp for idx in range(len(t) - 1)]

            flow = 0
            for link in G[nodei][nodej].keys():
                flow = flow + G[nodei][nodej][link]['weight']
            qij.append(flow)

            # # dk = degree of link k in MDij
            # dk = Counter()
            # for elem in MDij:
            #     # divide by the numnber of links between two nodes
            #     dk[elem] += 1 / len(G[elem[0]][elem[1]].keys())
            # V = np.array(list(dk.values()))
            # aij.append(NDij * (1 - float(sum(V - 1)) / sum(V)))

        Q[nodej] = sum(qij)  # Total flow into node j

        # Equation 7
        S[nodej] = 0
        for idx in range(len(qij)):
            if Q[nodej] != 0 and qij[idx] / Q[nodej] > 0:
                # S[nodej] = S[nodej] - \
                #            qij[idx] / Q[nodej] * math.log(qij[idx] / Q[nodej]) + \
                #            qij[idx] / Q[nodej] * math.log(aij[idx])
                S[nodej] = S[nodej] - \
                           qij[idx] / Q[nodej] * math.log(qij[idx] / Q[nodej])
    Q0 = sum(nx.get_edge_attributes(G, 'weight').values())

    # Equation 3
    S_ave = 0
    for nodej in sinks:
        if not np.isnan(S[nodej]):
            if nodej not in sources:
                if Q[nodej] / Q0 > 0:
                    S_ave = S_ave + \
                            (Q[nodej] * S[nodej]) / Q0 - \
                            Q[nodej] / Q0 * math.log(Q[nodej] / Q0)

    S = pd.Series(S)  # convert S to a series

    return [S, S_ave]
