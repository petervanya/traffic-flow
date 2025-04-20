#!/usr/bin/env python
"""
Test reading inputs.

2023-09-23
"""
import time

import traffic_flow as tfl
from traffic_flow import MTM
from traffic_flow.utils import read_inputs_excel, read_inputs_shapefile


def test_classmethod():
    print("\nTesting classmethod reading...")
    fname = "Internal/Networks/network_raw_I51_201002.xlsx"
    df_nodes, df_link_types, df_links = read_inputs_excel(fname)

    tic = time.time()
    model = tfl.from_dataframes(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")
    toc = time.time()
    print("Mean modelled flow:", model.df_links["q"].mean())
    print("Basic cycle done. Time: %.3f s" % (toc - tic))


def test_reading_csv():
    print("\nTesting raw input reading...")
    fname = "Internal/Networks/network_raw_I51_201002.xlsx"
    df_nodes, df_link_types, df_links = read_inputs_excel(fname)

    # first few steps
    model = MTM()

    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")
    toc = time.time()
    print("Mean modelled flow:", model.df_links["q"].mean())
    print("Basic cycle done. Time: %.3f s" % (toc - tic))


def test_reading_shapefile_ptv():
    print("\nTesting PTV shapefile reading...")
    basepath = (
        "/Users/peter/Tatra/UHP/Res/Transport_Models/Data/Shapefiles_I51/nulovy_stav/"
    )
    basename = "I51_I76_siet_nulovy_stav"
    df_nodes, df_link_types, df_links = read_inputs_shapefile(
        basepath, basename, verbose=True
    )

    # verify data quality
    set_links_from = set(df_links["node_from"])
    set_links_to = set(df_links["node_to"])
    set_nodes = set(df_nodes["id"])
    print(
        len(set_links_from.difference(set_nodes)),
        len(set_nodes.difference(set_links_from)),
    )
    print(
        len(set_links_to.difference(set_nodes)), len(set_nodes.difference(set_links_to))
    )
    print(
        len(set_links_from.difference(set_links_to)),
        len(set_links_to.difference(set_links_from)),
    )

    # first few steps
    model = MTM()

    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")
    toc = time.time()
    print("Mean modelled flow:", model.df_links["q"].mean())
    print("Basic cycle done. Time: %.3f s" % (toc - tic))


if __name__ == "__main__":
    test_classmethod()
    test_reading_csv()
    test_reading_shapefile_ptv()
