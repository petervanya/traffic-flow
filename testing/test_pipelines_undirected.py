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


def test_network_1_nx_undirected():
    from traffic_flow import MTMnxUndirected
    from traffic_flow.sample_networks import load_network_1_undirected

    print('\nTesting NetworkX backend, network 1 undirected...')

    df_nodes, df_link_types, df_links = load_network_1_undirected()
    
    model = MTMnxUndirected(verbose=True)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_2_nx_undirected():
    from traffic_flow import MTMnxUndirected
    from traffic_flow.sample_networks import load_network_2_undirected

    print('\nTesting NetworkX backend, network 2 undirected...')

    df_nodes, df_link_types, df_links = load_network_2_undirected()
    
    model = MTMnxUndirected(verbose=True)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_1_ig_undirected():
    from traffic_flow import MTMUndirected
    from traffic_flow.sample_networks import load_network_1_undirected

    print('\nTesting Igraph backend, network 1 undirected...')

    df_nodes, df_link_types, df_links = load_network_1_undirected()
    
    model = MTMUndirected(verbose=True)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


def test_network_2_ig_undirected():
    from traffic_flow import MTMUndirected
    from traffic_flow.sample_networks import load_network_2_undirected
    
    print('\nTesting Igraph backend, network 2 undirected...')

    df_nodes, df_link_types, df_links = load_network_2_undirected()

    model = MTMUndirected(verbose=True)
    model.read_data(df_nodes, df_link_types, df_links)
    
    model.generate("main-stratum", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)
    
    model.assign("tcur")

    print(model.df_links.head())


if __name__ == '__main__':
    test_network_1_nx_undirected()
    test_network_2_nx_undirected()
    test_network_1_ig_undirected()
    test_network_2_ig_undirected()
