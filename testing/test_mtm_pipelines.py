#!/usr/bin/env python3
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx

import mtsim
from mtsim import MTM
from mtsim.sample_networks import load_network_2

print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'igraph version: {ig.__version__}')
print(f'networkx version: {nx.__version__}')
print(f'Quince version: {mtsim.__version__}')

def test_simple_assignment():

    df_nodes, df_link_types, df_links = load_network_2()
    
    mtm = MTM(verbose=True)
    
    # read the road network data
    mtm.read_data(df_nodes, df_link_types, df_links)
    
    # trip generation, create a demand stratum
    # 'pop' is a zone attribute containing population
    mtm.generate("main-stratum", "pop", "pop", 0.5)
    
    # calculate skim (resistance) matrices for distance and time
    mtm.compute_skims()
    
    # trip distribution of demand stratum 'stratum_1' using the gravity model
    # distribute the trips exponentially with parameter 0.02
    # disutility defined as time 'tcur' computed above as a skim matrix
    mtm.distribute("main-stratum", "tcur", "exp", -0.02)
    
    # assign vehicle flows to the network
    mtm.assign("tcur")

    # print results
    # print(mtm.)


if __name__ == '__main__':
    test_simple_assignment()
