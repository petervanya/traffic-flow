#!/usr/bin/env python3
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx

import traffic_flow

print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'Igraph version: {ig.__version__}')
print(f'NetworkX version: {nx.__version__}')
print(f'traffic-flow version: {traffic_flow.__version__}')


def test_network_1_networkx():
    from traffic_flow import MTM
    from traffic_flow.sample_networks import load_network_1

    print('\nTesting NetworkX backend, network 1...')

    df_nodes, df_link_types, df_links = load_network_1()
    
    model = MTM(backend="networkx", verbose=True)
    print("Backend:", model.backend)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_2_networkx():
    from traffic_flow import MTM
    from traffic_flow.sample_networks import load_network_2

    print('\nTesting NetworkX backend, network 2...')

    df_nodes, df_link_types, df_links = load_network_2()
    
    model = MTM(backend="networkx", verbose=True)
    print("Backend:", model.backend)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_1_igraph():
    from traffic_flow import MTM
    from traffic_flow.sample_networks import load_network_1

    print('\nTesting Igraph backend, network 1...')

    df_nodes, df_link_types, df_links = load_network_1()
    
    model = MTM(backend="igraph", verbose=True)
    print("Backend:", model.backend)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_2_igraph():
    from traffic_flow import MTM
    from traffic_flow.sample_networks import load_network_2
    
    print('\nTesting Igraph backend, network 2...')

    df_nodes, df_link_types, df_links = load_network_2()

    model = MTM(backend="igraph", verbose=True)
    print("Backend:", model.backend)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


if __name__ == '__main__':
    test_network_1_networkx()
    test_network_2_networkx()
    test_network_1_igraph()
    test_network_2_igraph()
