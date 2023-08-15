#!/usr/bin/env python
"""
Testing of the optimisation procedure using Igraph backend.

Created: 2020-09-24
Update: 2023-08-15
"""
import pandas as pd
import time

from mtsim import MTM
from mtsim.sample_networks import load_network_2


def test_optimise():
    # loading data
    df_nodes, df_link_types, df_links = load_network_2()

    # first few steps
    model = MTM()
    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    toc = time.time()
    print("Basic cycle done. Time: %.3f s" % (toc - tic))

    # optimisation
    tic = time.time()
    model.optimise(n_iter=10)
    toc = time.time()

    print("Optimisation done. Time: %.3f s" % (toc - tic))
    print(model.opt_params.to_markdown())
    print(model.opt_output.to_markdown())


if __name__ == "__main__":
    test_optimise()
