import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from numbers import Number
import bezier
import numpy as np
from curved_edges import curved_edges

try:
    import plotly
except:
    plotly = None
try:
    import folium
except:
    folium = None

from wntr.graphics.color import custom_colormap


def _format_node_attribute(node_attribute, wn):
    if isinstance(node_attribute, str):
        node_attribute = wn.query_node_attribute(node_attribute)
    if isinstance(node_attribute, list):
        node_attribute = dict(zip(node_attribute, [1] * len(node_attribute)))
    if isinstance(node_attribute, pd.Series):
        node_attribute = dict(node_attribute)

    return node_attribute


def _format_link_attribute(link_attribute, wn):
    if isinstance(link_attribute, str):
        link_attribute = wn.query_link_attribute(link_attribute)
    if isinstance(link_attribute, list):
        link_attribute = dict(zip(link_attribute, [1] * len(link_attribute)))
    if isinstance(link_attribute, pd.Series):
        link_attribute = dict(link_attribute)

    return link_attribute


def plot_network(wn, node_attribute=None, link_attribute=None, title=None,
                 node_size=20, node_range=[None, None], node_alpha=1, node_cmap=None, node_labels=False,
                 link_width=1, link_range=[None, None], link_alpha=1, link_cmap=None, link_labels=False,
                 valve_layer=None, add_colorbar=True, node_colorbar_label='Node', link_colorbar_label='Link',
                 directed=False, ax=None, filename=None, source_list=None, curved_links=None,
                 curved_links_attribute=None):
    """
    Plot network graphic

    Parameters
    ----------
    wn : wntr WaterNetworkModel
        A WaterNetworkModel object

    node_attribute : None, str, list, pd.Series, or dict, optional

        - If node_attribute is a string, then a node attribute dictionary is
          created using node_attribute = wn.query_node_attribute(str)
        - If node_attribute is a list, then each node in the list is given a
          value of 1.
        - If node_attribute is a pd.Series, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float.
        - If node_attribute is a dict, then it should be in the format
          {nodeid: x} where nodeid is a string and x is a float

    link_attribute : None, str, list, pd.Series, or dict, optional

        - If link_attribute is a string, then a link attribute dictionary is
          created using edge_attribute = wn.query_link_attribute(str)
        - If link_attribute is a list, then each link in the list is given a
          value of 1.
        - If link_attribute is a pd.Series, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.
        - If link_attribute is a dict, then it should be in the format
          {linkid: x} where linkid is a string and x is a float.

    title : str, optional
        Plot title

    node_size : int, optional
        Node size

    node_range : list, optional
        Node range ([None,None] indicates autoscale)

    node_alpha : int, optional
        Node transparency

    node_cmap : matplotlib.pyplot.cm colormap or list of named colors, optional
        Node colormap

    node_labels: bool, optional
        If True, the graph will include each node labelled with its name.

    link_width : int, optional
        Link width

    link_range : list, optional
        Link range ([None,None] indicates autoscale)

    link_alpha : int, optional
        Link transparency

    link_cmap : matplotlib.pyplot.cm colormap or list of named colors, optional
        Link colormap

    link_labels: bool, optional
        If True, the graph will include each link labelled with its name.

    add_colorbar : bool, optional
        Add colorbar

    node_colorbar_label: str, optional
        Node colorbar label

    link_colorbar_label: str, optional
        Link colorbar label

    directed : bool, optional
        If True, plot the directed graph

    ax : matplotlib axes object, optional
        Axes for plotting (None indicates that a new figure with a single
        axes will be used)

    Returns
    -------
    nodes, edges : matplotlib objects for network nodes and edges

    Notes
    -----
    For more network draw options, see nx.draw_networkx
    """

    if ax is None:  # create a new figure
        plt.figure(facecolor='w', edgecolor='k')
        ax = plt.gca()

    # Graph
    G = wn.get_graph()
    if not directed:
        G = G.to_undirected()

    # Position
    pos = nx.get_node_attributes(G, 'pos')
    if len(pos) == 0:
        pos = None

    # Define node properties
    add_node_colorbar = add_colorbar
    if node_attribute is not None:

        if isinstance(node_attribute, list):
            if node_cmap is None:
                node_cmap = ['red', 'red']
            add_node_colorbar = False

        if node_cmap is None:
            node_cmap = plt.cm.Spectral_r
        elif isinstance(node_cmap, list):
            if len(node_cmap) == 1:
                node_cmap = node_cmap * 2
            node_cmap = custom_colormap(len(node_cmap), node_cmap)

        node_attribute = _format_node_attribute(node_attribute, wn)
        nodelist, nodecolor = zip(*node_attribute.items())

    else:
        nodelist = None
        nodecolor = 'k'

    add_link_colorbar = add_colorbar
    if link_attribute is not None:

        if isinstance(link_attribute, list):
            if link_cmap is None:
                link_cmap = ['red', 'red']
            add_link_colorbar = False

        if link_cmap is None:
            link_cmap = plt.cm.Spectral_r
        elif isinstance(link_cmap, list):
            if len(link_cmap) == 1:
                link_cmap = link_cmap * 2
            link_cmap = custom_colormap(len(link_cmap), link_cmap)

        link_attribute = _format_link_attribute(link_attribute, wn)

        # Replace link_attribute dictionary defined as
        # {link_name: attr} with {(start_node, end_node, link_name): attr}
        attr = {}
        for link_name, value in link_attribute.items():
            link = wn.get_link(link_name)
            attr[(link.start_node_name, link.end_node_name, link_name)] = value
        link_attribute = attr

        linklist, linkcolor = zip(*link_attribute.items())
    else:
        linklist = None
        linkcolor = 'k'

    if title is not None:
        ax.set_title(title)

    edge_background = nx.draw_networkx_edges(G, pos, edge_color='grey',
                                             width=0.5, ax=ax)

    nodes = nx.draw_networkx_nodes(G, pos,
                                   nodelist=nodelist, node_color=nodecolor, node_size=node_size,
                                   alpha=node_alpha, cmap=node_cmap, vmin=node_range[0], vmax=node_range[1],
                                   linewidths=0, ax=ax)

    # 水源节点可视化
    if source_list is None:
        source_list = wn.reservoir_name_list
    nodes2 = nx.draw_networkx_nodes(G, pos,
                                    nodelist=source_list, node_shape='s', node_color=nodecolor, node_size=30,
                                    alpha=node_alpha, cmap=node_cmap, vmin=node_range[0], vmax=node_range[1],
                                    linewidths=0, ax=ax)

    edges = nx.draw_networkx_edges(G, pos, edgelist=linklist,
                                   edge_color=linkcolor, width=link_width, alpha=link_alpha, edge_cmap=link_cmap,
                                   edge_vmin=link_range[0], edge_vmax=link_range[1], ax=ax)
    # 曲线边可视化
    if curved_links is not None:
        temp = list()
        for edge in G.edges:
            temp.append(edge)
        for e in temp:
            G.remove_edge(e[0], e[1])
        for link in curved_links:
            G.add_edge(link[0], link[1])
        edges2 = draw_networkx_edges(nx.Graph(G), pos, edgelist=curved_links,
                                     edge_color=curved_links_attribute,
                                     width=link_width, alpha=link_alpha, edge_cmap=link_cmap,
                                     edge_vmin=link_range[0], edge_vmax=link_range[1], ax=ax)
    # 只显示水源节点的标注
    if node_labels:
        # Net3
        # labels = dict(zip(source_list, source_list))
        # nx.draw_networkx_labels(G, pos, labels, verticalalignment='bottom',  font_size=10,
        #                         ax=ax)

        # ZJ
        labels = dict(zip(source_list, ['reservoir']))
        nx.draw_networkx_labels(G, pos, labels, verticalalignment='bottom', horizontalalignment='right', font_size=10,
                                ax=ax)

    if link_labels:
        labels = {}
        for link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            labels[(link.start_node_name, link.end_node_name)] = link_name
        nx.draw_networkx_edge_labels(G, pos, labels, font_size=7, ax=ax)
    if add_node_colorbar and node_attribute:
        clb = plt.colorbar(nodes, shrink=0.5, pad=0, ax=ax)
        clb.ax.set_title(node_colorbar_label, fontsize=10)
    if add_link_colorbar and link_attribute:
        clb = plt.colorbar(edges, shrink=0.5, pad=0.05, ax=ax)
        clb.ax.set_title(link_colorbar_label, fontsize=10)
    ax.axis('off')

    if valve_layer is not None:
        for valve_name, (pipe_name, node_name) in valve_layer.iterrows():
            pipe = wn.get_link(pipe_name)
            if node_name == pipe.start_node_name:
                start_node = pipe.start_node
                end_node = pipe.end_node
            elif node_name == pipe.end_node_name:
                start_node = pipe.end_node
                end_node = pipe.start_node
            else:
                print("Not valid")
                continue
            x0 = start_node.coordinates[0]
            dx = end_node.coordinates[0] - x0
            y0 = start_node.coordinates[1]
            dy = end_node.coordinates[1] - y0
            valve_coordinates = (x0 + dx * 0.1,
                                 y0 + dy * 0.1)
            ax.scatter(valve_coordinates[0], valve_coordinates[1], 15, 'r', 'v')

    if filename:
        plt.savefig(filename)

    return nodes, edges


def draw_networkx_edges(
        G,
        pos,
        edgelist=None,
        width=1.0,
        edge_color="k",
        style="solid",
        alpha=None,
        arrowstyle="-|>",
        arrowsize=10,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        arrows=True,
        label=None,
        node_size=300,
        nodelist=None,
        node_shape="o",
        connectionstyle=None,
        min_source_margin=0,
        min_target_margin=0,
):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color or array of colors (default='k')
       Edge color. Can be a single color or a sequence of colors with the same
       length as edgelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=None)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    connectionstyle : str, optional (default=None)
       Pass the connectionstyle parameter to create curved arc of rounding
       radius rad. For example, connectionstyle='arc3,rad=0.2'.
       See :py:class: `matplotlib.patches.ConnectionStyle` and
       :py:class: `matplotlib.patches.FancyArrowPatch` for more info.

    label : [None| string]
       Label for legend

    min_source_margin : int, optional (default=0)
       The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int, optional (default=0)
       The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size` as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if len(edgelist) == 0:  # no edges!
        if not G.is_directed() or not arrows:
            return LineCollection(None)
        else:
            return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
            np.iterable(edge_color)
            and (len(edge_color) == len(edge_pos))
            and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not G.is_directed() or not arrows:
        # edge_collection = LineCollection(
        #     edge_pos,
        #     colors=edge_color,
        #     linewidths=width,
        #     antialiaseds=(1,),
        #     linestyle=style,
        #     transOffset=ax.transData,
        #     alpha=alpha,
        # )

        curves = get_curved_edge(pos, edgelist)
        edge_collection = LineCollection(curves,
                                         colors=edge_color,
                                         linewidths=width,
                                         antialiaseds=(1,),
                                         linestyle=style,
                                         transOffset=ax.transData,
                                         alpha=alpha, )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection


def get_curved_edge(pos, edgelist):
    temp = list()
    for e in edgelist:
        start_xy = pos[e[0]]
        end_xy = pos[e[1]]

        # Net3
        x = (start_xy[0] + end_xy[0]) / 2 + 10
        y = (start_xy[1] + end_xy[1]) / 2 + 10

        # ZJ
        # x = max(start_xy[0], end_xy[0]) + 0.8
        # y = (start_xy[1] + end_xy[1]) / 2 + 0.5

        mid_xy = (x, y)
        temp.append([start_xy, mid_xy, end_xy])
    return np.asarray(temp)
