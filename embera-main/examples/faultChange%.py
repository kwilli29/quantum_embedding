""" Example comparing the embeddings and data obtained from a Layout-Agnostic
and a Layout-Aware embedding flow using minorminer on faulty Chimera or Zephyr.

Every ratio the code loops 10 [num_runs] times 
and each run the given ratio of nodes is removed 
from the specified topology,

The three algorithms are run once on each of the num_runs faulty topologies,

Each data point in the arrays per ratio represents 
an emmbedding on a topology different from the the previous run

The topology is the same b/w all 3 algorithms in the same run

Each run the topology resets
"""

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from minorminer import find_embedding
from embera.architectures import generators
from embera.architectures import drawing
from embera.preprocess.complete_diffusion_placer import find_candidates

from contextlib import redirect_stdout
import time
import random

n = 63
m = 3
num_runs = 10
d = 60 # degree of constant graph


# The corresponding graph of the D-Wave 2000Q annealer
#Tg = generators.dw2000q_graph()
Tg = generators.z15_graph(coordinates=True) #zephyr must have coords=True


# set up for broken topology 
size_graph = len(list(Tg.nodes())) 

# 2^-2...2^-8
log_ratio_broken_verts = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007182, 0.00390625]
ratio_broken_verts = [0.0,0.1,0.2,0.25,0.3,0.4,0.5]

# arrays to eventually be ints that are the # of verts. to be removed per ratio
num_broken_verts = [0]*len(ratio_broken_verts)

for j,k in enumerate(ratio_broken_verts): # ints of size_graph*each ratio
    num_broken_verts[j]=int(round(size_graph*k))


for rat in num_broken_verts: #run each ratio num_runs times, record data for each ratio

    times_comp_laAg = [0.0]*num_runs
    times_comp_laMg = [0.0]*num_runs
    times_comp_laNoMg = [0.0]*num_runs

    sum_comp_laAg = [0.0]*num_runs
    sum_comp_laMg = [0.0]*num_runs
    sum_comp_laNoMg = [0.0]*num_runs

    max_comp_laAg = [0.0]*num_runs
    max_comp_laMg = [0.0]*num_runs
    max_comp_laNoMg = [0.0]*num_runs

    fail_agn = [0.0]*num_runs
    fail_mig = [0.0]*num_runs
    fail_nmg = [0.0]*num_runs

    for p in range(num_runs):

        broken_graph = generators.z15_graph(coordinates=True) # reset every run
        
        for _ in range(rat): 

            if Tg.graph['family'] == 'zephyr':
                # choose a random coordinate instead of node index
                ru = random.randint(0, 1)
                rw = random.randint(0, (2*int(Tg.graph['rows'])))
                rk = random.randint(0, 3)
                rj = random.randint(0, 1)
                rz = random.randint(0, ((int(Tg.graph['rows']))) )

                rem = (ru, rw, rk, rj, rz)

            else:
                rem = random.randint(1, size_graph)
                
            if broken_graph.has_node(rem):
                broken_graph.remove_node(rem)
            else:
                # keep choosing a random # until it exists in the graph
                node_in = False
                while not node_in:
                    if Tg.graph['family'] == 'zephyr':
                        ru = random.randint(0, 1)
                        rw = random.randint(0, (2*int(Tg.graph['rows'])))
                        rk = random.randint(0, 3)
                        rj = random.randint(0, 1)
                        rz = random.randint(0, ((int(Tg.graph['rows']))) )

                        rem = (ru, rw, rk, rj, rz)

                    else:
                        rem = random.randint(1, size_graph)

                    if broken_graph.has_node(rem):
                        broken_graph.remove_node(rem)
                        node_in = True

            
        Sg = nx.complete_graph(d)
        S_edgelist = list(Sg.edges())

        #convert complete nodes to coordinates
        S_coordlist = [((0,0),(0,0))]*len(S_edgelist)
        for i,edge in enumerate(S_edgelist):
            x,y = edge
            S_coordlist[i] = ((x,x),(y,y)) 

        # Layout of the problem graph
        layout = {v:v for e in S_coordlist for v in e}

        T_edgelist = list(broken_graph.edges())   
        
        # Find a minor-embedding
        start = time.process_time()
        embedding = find_embedding(S_edgelist, T_edgelist) # Minorminer
        times_comp_laAg[p] = time.process_time() - start   # embedding times
        if embedding.values():
            sum_comp_laAg[p] = (sum(len(v) for v in embedding.values())) #qubits
            max_comp_laAg[p] = (max(len(v) for v in embedding.values())) #maxchain size
        else:
            fail_agn[p]=1 # number of failed embeddings

        # Find a global placement for problem graph
        start = time.process_time()
        candidates = find_candidates(S_coordlist, broken_graph, layout=layout)
        # Find a minor-embedding using the initial chains from global placement
        migrated_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
        times_comp_laMg[p] = time.process_time() - start
        if migrated_embedding.values():
            sum_comp_laMg[p] = (sum(len(v) for v in migrated_embedding.values()))
            max_comp_laMg[p] = (max(len(v) for v in migrated_embedding.values()))
        else:
            fail_mig[p]=1

        # Find a global placement for problem graph
        start = time.process_time()
        candidates = find_candidates(S_coordlist, broken_graph, layout=layout, enable_migration=False)
        # Find a minor-embedding using the initial chains from global placement
        guided_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
        times_comp_laNoMg[p] = time.process_time() - start
        if guided_embedding.values():
            sum_comp_laNoMg[p] = (sum(len(v) for v in guided_embedding.values()))
            max_comp_laNoMg[p] = (max(len(v) for v in guided_embedding.values()))
        else:
            fail_nmg[p]=1

        ###################
        #      Plots      #
        ###################
        
        if p>7:
            plt.figure(1)
            plt.title('Layout-Agnostic')
            drawing.draw_architecture_embedding(broken_graph, embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_aware_agn_bz15')
            plt.clf()

            plt.figure(2)
            plt.title('Layout-Aware (enable_migration=True)')
            drawing.draw_architecture_embedding(broken_graph, migrated_embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_aware_mig_bz15')
            plt.clf()

            plt.figure(3)
            plt.title('Layout-Aware (enable_migration=False)')
            drawing.draw_architecture_embedding(broken_graph, guided_embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_aware_nomig_bz15')
            plt.clf()

            ###################
            #    Histogram    #
            ###################
            
            plt.figure(4)
            plt.title(' Agnostic Chain Size Histogram')
            _ = plt.hist([len(v) for v in embedding.values()])#, bins=max_comp_laAg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_agn_hist_bz15')
            plt.clf()

            plt.figure(5)
            plt.title(' Aware-Mig Chain Size Histogram')
            _ = plt.hist([len(v) for v in migrated_embedding.values()])#, bins=max_comp_laMg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_awaremig_hist_bz15')
            plt.clf()

            plt.figure(6)
            plt.title(' Aware-NoMig Chain Size Histogram')
            _ = plt.hist([len(v) for v in guided_embedding.values()])#, bins=max_comp_laNoMg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{p}brknComp_layout_awarenomig_hist_bz15')
            plt.clf()
        
        print(f':{rat}:, {d} done :D')

    ###################
    #      Data       #
    ###################

        with open(f'../Desktop/BigCompleteLayoutAware/Zephyr/{rat}_{d}brknComp_dataFaults_bz15.txt', 'w') as f:
            with redirect_stdout(f):
                
                print(f'BrknComp Agnostic Total Times: {times_comp_laAg}')
                print(f'BrknComp LA Mig Total Times: {times_comp_laMg}')
                print(f'BrknComp LA NO Mig Total Times: {times_comp_laNoMg}')
                print()
                print(f'BrknComp Agnostic Total Sum: {sum_comp_laAg}')
                print(f'BrknComp LA Mig Total Sum: {sum_comp_laMg}')
                print(f'BrknComp LA NO Mig Total Sum: {sum_comp_laNoMg}')
                print()
                print(f'BrknComp Agnostic Total Max: {max_comp_laAg}')
                print(f'BrknComp LA Mig Total Max: {max_comp_laMg}')
                print(f'BrknComp LA NO Mig Total Max: {max_comp_laNoMg}')
                print()
                print(f'BrknComp Agnostic Total Fails: {fail_agn}')
                print(f'BrknComp LA Mig Total Fails: {fail_mig}')
                print(f'BrknComp LA NO Mig Total Fails: {fail_nmg}')

