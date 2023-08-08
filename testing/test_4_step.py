#!/usr/bin/env python3
"""
A test of the MTM object (macroscopic transport model) with Igraph backend.

2020-08-03
"""
import os
import pandas as pd
from mtsim import MTM


def test_network_2():
    # reading data
    xls = pd.ExcelFile(os.path.abspath("mtsim/examples/network_2.xlsx"))
    df_nodes = xls.parse("nodes")
    df_links = xls.parse("links")
    if "LENGTH" in df_links.columns:
        df_links = df_links.drop("LENGTH", 1)
    df_link_types = xls.parse("link_types")

    # the model object
    mtm = MTM()
    mtm.read_data(df_nodes, df_link_types, df_links)
    mtm.compute_skims()
    mtm.generate("ALL", "pop", "pop", 0.5)
    mtm.distribute("ALL", "t0", "exp", -0.02)
    mtm.assign("tcur")

    print(mtm['df_links'])


if __name__ == '__main__':
    test_network_2()