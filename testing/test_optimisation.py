#!/usr/bin/env python
"""
Testing of the optimisation procedure using Igraph backend.

Created: 2020-09-24
Update: 2023-08-15
"""
import time

from traffic_flow import MTM
from traffic_flow.sample_networks import load_network_2


def test_optimise(method, x0=None):
    # loading data
    df_nodes, df_link_types, df_links = load_network_2()

    # first few steps
    model = MTM()
    print("Backend:", model.backend)

    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    toc = time.time()
    print("Basic cycle done. Time: %.3f s" % (toc - tic))

    # optimisation
    tic = time.time()
    res = model.optimise(method=method, n_iter=10, x0=x0)
    toc = time.time()

    print(res)

    print(model.opt_params)
    print(model.opt_output)


if __name__ == "__main__":
    # test_optimise('dual-annealing')
    print("\nTesting dual annealing...")
    test_optimise("dual-annealing", x0=[0.07, -1e-3])
    print("\nTesting Nelder-Mead method...")
    test_optimise("nelder-mead", x0=[0.07, -1e-3])
