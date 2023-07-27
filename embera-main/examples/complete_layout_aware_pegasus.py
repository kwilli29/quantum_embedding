""" Example comparing the embeddings and data obtained from a Layout-Agnostic
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

times_comp_laAg = [0.0]*n # times complete layout aware Agnostic
times_comp_laMg = [0.0]*n # times complete layout aware Migration
times_comp_laNoMg = [0.0]*n # times complete layout aware No Migration

sum_comp_laAg = [0.0]*n
sum_comp_laMg = [0.0]*n
sum_comp_laNoMg = [0.0]*n

max_comp_laAg = [0.0]*n
max_comp_laMg = [0.0]*n
max_comp_laNoMg = [0.0]*n

fail_agn = 0
fail_mig = 0
fail_nmg = 0

# Original Code: A 6-node Complete problem graph
#Sg = nx.grid_2d_graph(6, 6)

for w in range(3,n,1):

    Sg = nx.complete_graph(w)
    S_edgelist = list(Sg.edges())

    #convert complete graph nodes to coordinates 
    S_coordlist = [((0,0,0),(0,0,0))]*len(S_edgelist)
    for i,edge in enumerate(S_edgelist):
        x,y = edge
        S_coordlist[i] = ((0,x,x),(0,y,y)) # pegasus coords = 3 #s, first num can be 0,1, or 2 # 0 is easiest

    # Layout of the problem graph
    layout = {v:v for e in S_coordlist for v in e}
    # Original Code: layout = {v:v for v in S_edgelist}
    
    # The corresponding graph of the D-Wave 2000Q annealer
    Tg = generators.p16_graph() # 3# tile coords
   
    T_edgelist = list(Tg.edges())

    for j in range(m):

        #print('Layout-Agnostic')
        # Find a minor-embedding
        start = time.process_time()
        embedding = find_embedding(S_edgelist, T_edgelist)
        # all pnts = [w*m+j]=   , avg= [w]+=
        times_comp_laAg[w] += time.process_time() - start
        #print('sum: %s' % sum(len(v) for v in embedding.values()))
        sum_comp_laAg[w] += (sum(len(v) for v in embedding.values()))
        #print('max: %s' % max(len(v) for v in embedding.values()))
        max_comp_laAg[w] += (max(len(v) for v in embedding.values()))
        if not embedding:
            fail_agn+=1

        #print('Layout-Aware (enable_migration=True)')
        # Find a global placement for problem graph
        start = time.process_time()
        candidates = find_candidates(S_coordlist, Tg, layout=layout)
        # Find a minor-embedding using the initial chains from global placement
        migrated_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
        times_comp_laMg[w] += time.process_time() - start
        #print('sum: %s' % sum(len(v) for v in migrated_embedding.values()))
        sum_comp_laMg[w] += (sum(len(v) for v in migrated_embedding.values()))
        #print('max: %s' % max(len(v) for v in migrated_embedding.values()))
        max_comp_laMg[w] += (max(len(v) for v in migrated_embedding.values()))
        if not migrated_embedding:
            fail_mig+=1

        #print('Layout-Aware (enable_migration=False)')
        # Find a global placement for problem graph
        start = time.process_time()
        candidates = find_candidates(S_coordlist, Tg, layout=layout, enable_migration=False)
        # Find a minor-embedding using the initial chains from global placement
        guided_embedding = find_embedding(S_coordlist, T_edgelist, initial_chains=candidates)
        times_comp_laNoMg[w] += time.process_time() - start
        #print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
        sum_comp_laNoMg[w] += (sum(len(v) for v in guided_embedding.values()))
        #print('max: %s' % max(len(v) for v in guided_embedding.values()))
        max_comp_laNoMg[w] += (max(len(v) for v in guided_embedding.values()))
        if not guided_embedding:
            fail_nmg+=1

        # include when finding averages
        for tensor in [times_comp_laAg, times_comp_laMg, times_comp_laNoMg, sum_comp_laAg, sum_comp_laMg, sum_comp_laNoMg, max_comp_laAg, max_comp_laMg,max_comp_laNoMg]:#qubits_zephyr, qubits_zephyr_clique, maxchain_zephyr, maxchain_zephyr_clique]:
            tensor[w] /= m  

        fail_agn/=m
        fail_mig/=m 
        fail_nmg/=m


    ###################
    #      Plots      #
    ###################
    # record images of embeddings at specified vertex counts
    if w >= 55:
        plt.figure(1)
        plt.title('Layout-Agnostic')
        drawing.draw_architecture_embedding(Tg, embedding, node_shape='.',width=0.5,node_size=2)
        plt.tight_layout()
        plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_01agnostic_p16')
        plt.clf()

        plt.figure(2)
        plt.title('Layout-Aware (enable_migration=True)')
        drawing.draw_architecture_embedding(Tg, migrated_embedding, node_shape='.',width=0.5,node_size=2)
        plt.tight_layout()
        plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_aware_01mig_p16')
        plt.clf()

        plt.figure(3)
        plt.title('Layout-Aware (enable_migration=False)')
        drawing.draw_architecture_embedding(Tg, guided_embedding, node_shape='.',width=0.5,node_size=2)
        plt.tight_layout()
        plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_aware_01nomig_p16')
        plt.clf()

    ###################
    #    Histogram    #
    ###################
    # Record Chain Size Histograms over specified vertex counts
    
    plt.figure(4)
    plt.title('Avg Agnostic Chain Size Histogram')
    _ = plt.hist([len(v) for v in embedding.values()])#, bins=max_comp_laAg[w])
    plt.tight_layout()
    plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_agn_02hist_p16')
    plt.clf()

    plt.figure(5)
    plt.title('Avg Aware-Mig Chain Size Histogram')
    _ = plt.hist([len(v) for v in migrated_embedding.values()])#, bins=max_comp_laMg[w])
    plt.tight_layout()
    plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_awaremig_02hist_p16')
    plt.clf()

    plt.figure(6)
    plt.title('Avg Aware-NoMig Chain Size Histogram')
    _ = plt.hist([len(v) for v in guided_embedding.values()])#, bins=max_comp_laNoMg[w])
    plt.tight_layout()
    plt.savefig(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}complete_layout_awarenomig_02hist_p16')
    plt.clf()
    
    print(f'{w} done :D')

#########################
#      Data Output      #
#########################

with open(f'../Desktop/BigCompleteLayoutAware/Pegasus/{w}avg_complete_data02_p16.txt', 'w') as f:
    with redirect_stdout(f):

        print(f'Complete Agnostic Avg Times: {times_comp_laAg}')
        print(f'Complete LA Mig Avg Times: {times_comp_laMg}')
        print(f'Complete LA NO Mig Avg Times: {times_comp_laNoMg}')
        print()
        print(f'Complete Agnostic Avg Sum: {sum_comp_laAg}')
        print(f'Complete LA Mig Avg Sum: {sum_comp_laMg}')
        print(f'Complete LA NO Mig Avg Sum: {sum_comp_laNoMg}')
        print()
        print(f'Complete Agnostic Avg Max: {max_comp_laAg}')
        print(f'Complete LA Mig Avg Max: {max_comp_laMg}')
        print(f'Complete LA NO Mig Avg Max: {max_comp_laNoMg}')
        print()
        print(f'Complete Agnostic Avg Fails: {fail_agn}')
        print(f'Complete LA Mig Avg Fails: {fail_mig}')
        print(f'Complete LA NO Mig Avg Fails: {fail_nmg}')

