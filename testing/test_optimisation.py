#!/usr/bin/env python
"""
Testing of the optimisation procedure using Igraph backend.

2020-09-24
"""
import time
from mtsim import MTM
from mtsim.sample_networks import load_network_2

def test_optimise():
    # loading data
    df_nodes, df_link_types, df_links = load_network_2()

    # MTM run
    mtm = MTM()
    tic = time.time()
    mtm.read_data(df_nodes, df_link_types, df_links)
    mtm.generate("ALL", "pop", "pop", 0.5)
    mtm.compute_skims()
    mtm.distribute("ALL", "tcur", "exp", 0.02)
    #mtm.assign("tcur")
    toc = time.time()
    print("Cycle done.")
    print("Time: %.2f s" % (toc - tic))

    # optimisation
    tic = time.time()
    mtm.optimise(10)
    toc = time.time()
    print(mtm.optres)


if __name__ == '__main__':
    test_optimise()