""" Example of tiling a Zephyr architecture graph. - KW """
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from embera.architectures import drawing, generators
from embera.preprocess.tiling_parser import DWaveNetworkXTiling

import random

colours = {'u':{},'w':{},'k':{},'z':{},'t':{},'ij':{},'k2':{},'tij':{}}
Tg = dnx.zephyr_graph(3,coordinates=True)

colours = {}
for index, tile in DWaveNetworkXTiling(Tg).items():
    if tile.qubits:
        colours[index] = list(tile.qubits)

drawing.draw_architecture_embedding(Tg, colours, show_labels=True, node_size=10)
plt.show()

