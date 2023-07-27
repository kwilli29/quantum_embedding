""" Generators for architecture graphs.

    All architecture graphs use the same parameters. Additional parameters
    for the underlying generators are allowed but discouraged.

    Parameters
    ----------
    data : bool (optional, default True)
        If True, each node has a `<family>_index attribute`
    coordinates : bool (optional, default False)
        If True, node labels are tuples, equivalent to the <family>_index
        attribute as above.  In this case, the `data` parameter controls the
        existence of a `linear_index attribute`, which is an int

    Returns
    -------
    G : NetworkX Graph of the chosen architecture.
        Nodes are labeled by integers.

"""
import os
import tarfile
import requests
import dwave.system
import networkx as nx
import dwave_networkx as dnx

__all__ = ['graph_from_solver','dwave_online',
           'rainier_graph', 'vesuvius_graph', 'dw2x_graph', 'dw2000q_graph', 
           'p6_graph', 'p16_graph', 'z15_graph',
           'h20k_graph',
           ]

""" ========================== D-Wave Solver Solutions ===================== """

def graph_from_solver(solver, **kwargs):
    """ D-Wave architecture graph from Dimod Structured Solver
    """
    chip_id = solver.properties['chip_id']
    sampler = dwave.system.DWaveSampler(solver=chip_id)
    target_graph = sampler.to_networkx_graph()
    target_graph.graph['chip_id'] = chip_id

    return target_graph

def dwave_online(squeeze=True, **kwargs):
    """ Architecture graphs from D-Wave devices `online`"""
    import dwave.cloud
    with dwave.cloud.Client.from_config(**kwargs) as client:
        solvers = client.get_solvers()
    graphs = [graph_from_solver(s) for s in solvers if s.properties.get('topology')]
    if squeeze:
        return graphs[0] if len(graphs)==1 else graphs
    else:
        return graphs

def dwave_collection(name=None):
    """ Architecture graphs from current and legacy D-Wave devices

        |name                 | nodes    | edges  |
        | ------------------- |:--------:| ------:|
        |Advantage_system1.1  | 5436     | 37440  |
        |DW_2000Q_6           | 2041     | 5974   |
        |DW_2000Q_5           | 2030     | 5909   |
        |DW_2000Q_2_1         | 2038     | 5955   |
        |DW_2000Q_QuAIL       | 2031     | 5919   |
        |DW_2X_LANL           | 1141     | 3298   |

        Returns list of NetworkX graphs with parameters:
            >>> G.graph = {'columns': <int>,
                           'data': bool,
                           'family': <string>,
                           'labels': <string>,
                           'name': <string>,
                           'rows': <int>,
                           'tile': <int>}
    """
    graph_list = []
    path = "./collection.tar.gz"
    url = "http://www.ece.ubc.ca/~jpinilla/resources/embera/architectures/dwave/collection.tar.gz"

    # Download
    if not os.path.isfile(path):
        print(f"-> Downloading D-Wave architecture collection to {path}")
        with open(path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
    # Unzip, untar, unpickle
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            f = contents.extractfile(member)
            G = nx.read_gpickle(f)
            graph_list.append(G)

    if name is None:
        return graph_list
    else:
        try:
            return next(g for g in graph_list if g.name==name)
        except:
            raise KeyError("Architecture graph name not found in collection")

""" =========================== D-Wave Architectures ======================= """

def rainier_graph(**kwargs):
    """ D-Wave One 'Rainier' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(4, 4, 4, **kwargs)
    target_graph.graph['chip_id'] = 'Rainier'
    return target_graph

def vesuvius_graph(**kwargs):
    """ D-Wave Two 'Vesuvius' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(8, 8, 4, **kwargs)
    target_graph.graph['chip_id'] = 'Vesuvius'
    return target_graph

def dw2x_graph(**kwargs):
    """ D-Wave 2X Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(12, 12, 4, **kwargs)
    target_graph.graph['chip_id'] = 'DW_2X'
    return target_graph

def dw2000q_graph(**kwargs):
    """ D-Wave 2000Q Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(16, 16, 4, **kwargs)
    target_graph.graph['chip_id'] = 'DW_2000Q'
    return target_graph


def p6_graph(**kwargs):
    """ Pegasus 6 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(6, **kwargs)
    target_graph.graph['chip_id'] = 'P6'
    return target_graph

def p16_graph(**kwargs):
    """ Pegasus 16 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(16, **kwargs)
    target_graph.graph['chip_id'] = 'P16'
    return target_graph

## KW ##
def z15_graph(**kwargs):
    ''' Zephyr 15 Graph
        https://www.dwavesys.com/media/fawfas04/14-1056a-a_zephyr_topology_of_d-wave_quantum_processors.pdf
    '''
    target_graph = dnx.generators.zephyr_graph(15, **kwargs)
    target_graph.graph['chip_id'] = 'Z15'
    return target_graph

########
""" ============================== Miscellaneous =========================== """

def h20k_graph(data=True, coordinates=False):
    """ HITACHI 20k-Spin CMOS digital annealer graph.
        https://ieeexplore.ieee.org/document/7350099/
    """
    n, m, t = 128, 80, 2

    target_graph = nx.grid_graph(dim=[t, m, n])

    target_graph.name = 'hitachi_graph(128,80,2)'
    target_graph.graph['chip_id'] = 'HITACHI 20k'
    construction = (("family", "hitachi"),
                    ("rows", 5), ("columns", 4),
                    ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    target_graph.graph.update(construction)

    if coordinates:
        if data:
            for t_node in target_graph:
                (z_coord, y_coord, x_coord) = t_node
                linear = x_coord + n*(y_coord + m*z_coord)
                target_graph.nodes[t_node]['linear_index'] = linear
    else:
        coordinate_labels = {(x, y, z):x+n*(y+m*z) for (x, y, z) in target_graph}
        if data:
            for t_node in target_graph:
                target_graph.nodes[t_node]['grid_index'] = t_node
        target_graph = nx.relabel_nodes(target_graph, coordinate_labels)

    return target_graph
