""" Comparing the embeddings and data obtained from a Layout-Agnostic
and a Layout-Aware embedding flow using minorminer on Pegasus.
"""

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from minorminer import find_embedding
from embera.architectures import generators
from embera.architectures import drawing
from embera.preprocess.complete_diffusion_placer_pegasus import find_candidates

from contextlib import redirect_stdout
import time
import random

n = 61
m = 3 # runs per vertex

# The Pegasus graph
Tg = generators.p16_graph()


# set up for broken topology 
size_graph = len(list(Tg.nodes())) 

# 2^-7 ... 2^-1
ratio_broken_verts = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007182, 0.00390625]

# arrays to eventually be ints that are the # of verts. to be removed per ratio
num_broken_verts = [0]*len(ratio_broken_verts)

for j,k in enumerate(ratio_broken_verts): # ints of size_graph*ratio
    num_broken_verts[j]=int(round(size_graph*k))


for rat in num_broken_verts:

    broken_graph = Tg # reset every run

    times_comp_laAg = [0.0]*n*m
    times_comp_laMg = [0.0]*n*m
    times_comp_laNoMg = [0.0]*n*m

    sum_comp_laAg = [0.0]*n*m
    sum_comp_laMg = [0.0]*n*m
    sum_comp_laNoMg = [0.0]*n*m

    max_comp_laAg = [0.0]*n*m
    max_comp_laMg = [0.0]*n*m
    max_comp_laNoMg = [0.0]*n*m

    fail_agn = [0.0]*n*m
    fail_mig = [0.0]*n*m
    fail_nmg = [0.0]*n*m
    
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
            # keep choosing a random # until it exists in broken target graph
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

    for w in range(3,n,1):

        Sg = nx.complete_graph(w)
        S_edgelist = list(Sg.edges())

        #convert complete nodes to coordinates, pegasus=3 coords
        S_coordlist = [((0,0,0),(0,0,0))]*len(S_edgelist)
        for i,edge in enumerate(S_edgelist):
            x,y = edge
            S_coordlist[i] = ((0,x,x),(0,y,y)) 

        # Layout of the problem graph
        layout = {v:v for e in S_coordlist for v in e}

        T_edgelist = list(broken_graph.edges())

        for j in range(m):
            # all pnts = [w*m+j]=   , avg= [w]+=

            # Agnostic [Minorminer]
            # Find a minor-embedding
            start = time.process_time()
            embedding = find_embedding(S_edgelist, T_edgelist)
            times_comp_laAg[w*m+j] = time.process_time() - start
            if embedding.values():
                sum_comp_laAg[w*m+j] = (sum(len(v) for v in embedding.values()))
                max_comp_laAg[w*m+j] = (max(len(v) for v in embedding.values()))
            else:
                fail_agn[w*m+j]=1

            # Layout Aware [Migration True]
            # Find a global placement for problem graph
            start = time.process_time()
            candidates = find_candidates(S_coordlist, broken_graph, layout=layout)
            # Find a minor-embedding using the initial chains from global placement
            migrated_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
            times_comp_laMg[w*m+j] = time.process_time() - start
            if migrated_embedding.values():
                #print('sum: %s' % sum(len(v) for v in migrated_embedding.values()))
                sum_comp_laMg[w*m+j] = (sum(len(v) for v in migrated_embedding.values()))
                #print('max: %s' % max(len(v) for v in migrated_embedding.values()))
                max_comp_laMg[w*m+j] = (max(len(v) for v in migrated_embedding.values()))
            else:
                fail_mig[w*m+j]=1

            # Layout Aware [Migration False]
            # Find a global placement for problem graph
            start = time.process_time()
            candidates = find_candidates(S_coordlist, broken_graph, layout=layout, enable_migration=False)
            # Find a minor-embedding using the initial chains from global placement
            guided_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
            times_comp_laNoMg[w*m+j] = time.process_time() - start
            if guided_embedding.values():
                #print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
                sum_comp_laNoMg[w*m+j] = (sum(len(v) for v in guided_embedding.values()))
                #print('max: %s' % max(len(v) for v in guided_embedding.values()))
                max_comp_laNoMg[w*m+j] = (max(len(v) for v in guided_embedding.values()))
            else:
                fail_nmg[w*m+j]=1

            # include if finding averages
            #for tensor in [times_comp_laAg, times_comp_laMg, times_comp_laNoMg, sum_comp_laAg, sum_comp_laMg, sum_comp_laNoMg, max_comp_laAg, max_comp_laMg,max_comp_laNoMg]:#qubits_zephyr, qubits_zephyr_clique, maxchain_zephyr, maxchain_zephyr_clique]:
            #    tensor[w] /= m  

            #fail_agn/=m
            #fail_mig/=m 
            #fail_nmg/=m


        ###################
        #      Plots      #
        ###################
        
        if w >= 58:
            plt.figure(1)
            plt.title('Layout-Agnostic')
            drawing.draw_architecture_embedding(broken_graph, embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_aware_agn_bp16')
            plt.clf()

            plt.figure(2)
            plt.title('Layout-Aware (enable_migration=True)')
            drawing.draw_architecture_embedding(broken_graph, migrated_embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_aware_mig_bp16')
            plt.clf()

            plt.figure(3)
            plt.title('Layout-Aware (enable_migration=False)')
            drawing.draw_architecture_embedding(broken_graph, guided_embedding, node_shape='.',width=0.5,node_size=2)
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_aware_nomig_bp16')
            plt.clf()

            ###################
            #    Histogram    #
            ###################
            
            plt.figure(4)
            plt.title(' Agnostic Chain Size Histogram')
            _ = plt.hist([len(v) for v in embedding.values()])#, bins=max_comp_laAg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_agn_hist_bp16')
            plt.clf()

            plt.figure(5)
            plt.title(' Aware-Mig Chain Size Histogram')
            _ = plt.hist([len(v) for v in migrated_embedding.values()])#, bins=max_comp_laMg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_awaremig_hist_bp16')
            plt.clf()

            plt.figure(6)
            plt.title(' Aware-NoMig Chain Size Histogram')
            _ = plt.hist([len(v) for v in guided_embedding.values()])#, bins=max_comp_laNoMg[w])
            plt.tight_layout()
            plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}brknComp_layout_awarenomig_hist_bp16')
            plt.clf()
        
        print(f':{rat}:, {w} done :D')

    ###################
    #      Data       #
    ###################

        with open(f'../Desktop/BigCompleteLayoutAware/Pegasus/{rat}_{w}MMtotal_brknComp_data_bp16.txt', 'w') as f:
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

