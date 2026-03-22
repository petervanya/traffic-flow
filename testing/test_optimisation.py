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
    res = model.optimise(method=method, n_iter=10, x0=x0, record=True)
    toc = time.time()

    print(res)

    print(model.opt_params)


def test_optimise_train_test_split(method, train_test_split=0.6, x0=None):
    """Test optimisation with train-test split for validation."""
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

    # optimisation with train-test split
    tic = time.time()
    res = model.optimise(
        method=method, 
        n_iter=10, 
        x0=x0, 
        record=True, 
        train_test_split=train_test_split
    )
    toc = time.time()

    print(res)
    print(model.opt_params)
    
    # Print train vs test error if available
    print(f"\nTrain error (GEH): {res.train_error:.3f}")
    print(f"Test error (GEH): {res.test_error:.3f}")


if __name__ == "__main__":
    print("\nTesting Nelder-Mead method...")
    test_optimise("nelder-mead", x0=[0.07, -1e-3])
    print("\nTesting dual annealing...")
    test_optimise("dual-annealing", x0=[0.07, -1e-3])
    print("\nTesting Nelder-Mead with 60% train-test split...")
    test_optimise_train_test_split("nelder-mead", train_test_split=0.6, x0=[0.07, -1e-3])
    print("\nTesting dual annealing with 60% train-test split...")
    test_optimise_train_test_split("dual-annealing", train_test_split=0.6, x0=[0.07, -1e-3])
