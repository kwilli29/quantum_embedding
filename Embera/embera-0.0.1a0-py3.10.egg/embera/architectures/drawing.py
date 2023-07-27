"""
Tools to visualize graphs of architectures.
"""

import warnings
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt


__all__ = ['draw_architecture', 'draw_architecture_embedding', 'draw_architecture_yield']

unused_color=(0.9,0.9,0.9,1.0)
faulty_color=(1.0,0.0,0.0,1.0)

def draw_architecture(target_graph, **kwargs):
    """ Draws graph G according to G's family layout.
    """
    family = target_graph.graph['family']

    if family == 'chimera':
        dnx.draw_chimera(target_graph, **kwargs)

    elif family == 'pegasus':
        dnx.draw_pegasus(target_graph, **kwargs)

    elif family == 'zephyr':
        dnx.draw_zephyr(target_graph, **kwargs)

    else:
        nx.draw_spring(target_graph)
        warnings.warn("Graph family not available. Using NetworkX spring layout")


def draw_architecture_yield(target_graph, **kwargs):
    """ Draws graph G according to G's family layout and highlights
        faulty qubits.
    """
    family = target_graph.graph['family']

    try:
        m = target_graph.graph['columns']
        n = target_graph.graph['rows']
        t = target_graph.graph['tile']
        coordinates = target_graph.graph['labels'] == 'coordinate'
    except:
        raise ValueError("Target graph needs to have columns, rows, and tile \
        attributes to identify faulty qubits.")

    if family == 'chimera':
        dnx.draw_chimera_yield(target_graph, **kwargs)

    elif family == 'pegasus':
        dnx.draw_pegasus_yield(target_graph, **kwargs)

    elif family == 'zephyr':
        dnx.draw_zephyr_yield(target_graph, **kwargs)

    else:
        nx.draw_spring(target_graph)
        warnings.warn("Graph family not available. Using NetworkX spring layout")


def draw_architecture_embedding(target_graph, *args, **kwargs):
    """ Draws an embedding onto the target graph G,
        according to G's family layout.
    """
    family = target_graph.graph.get('family')

    if family == 'chimera':
        dnx.draw_chimera_embedding(target_graph, *args, **kwargs)#show_labels=False,node_shape='.',width=1.0,node_size=10)

    elif family == 'pegasus':
        dnx.draw_pegasus_embedding(target_graph, *args, **kwargs)#show_labels=False, node_shape='.',width=0.75,node_size=10)
    
    elif family == 'zephyr':
        dnx.draw_zephyr_embedding(target_graph, *args, **kwargs)#show_labels=True, node_shape='.',width=0.75,node_size=10)
    else:
        layout = nx.spring_layout(target_graph)
        dnx.drawing.qubit_layout.draw_embedding(target_graph, layout, *args, **kwargs)
        warnings.warn("Graph family not available. Using NetworkX spring layout")

def draw_tiled_graph(m, n, tiles, topology, **kwargs):
    """ Draws a grid representing the architecture tiles
        with an overlay of source nodes.
    """
    #TODO: Use graph colouring from dwave.system.composites.tiling.draw_tiling

    concentrations = {name : "s=%s,d=%s"%(tile.supply, tile.concentration)
                            for name, tile in tiles.items() if name!=None}
    G = nx.empty_graph()
    G.add_nodes_from(topology.keys())

    dnx.drawing.qubit_layout.draw_qubit_graph(G, topology,**kwargs)
    cf = plt.gcf()
    ax = plt.gca()
    plt.grid('on')
    plt.axis('on')
    plt.axis([0,n,0,m])
    x_ticks = range(0, n) # steps are width/width = 1 without scaling
    y_ticks = range(0, m)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # Label tiles
    for (i,j), label in concentrations.items():
        plt.text(j, i, label)
        plt.text(j+0.5, i+0.5, '%s,%s'%(i,j))
    ax.invert_yaxis()
