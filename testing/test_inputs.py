#!/usr/bin/env python
"""
Test reading inputs.

2023-09-23
"""
import time

from traffic_flow import MTM
from traffic_flow.utils import read_inputs_excel, read_inputs_shapefile

def test_reading_csv():
    fname = 'Internal/Networks/network_raw_I51_201002.xlsx'
    df_nodes, df_link_types, df_links = read_inputs_excel(fname)

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


if __name__ == '__main__':
    test_reading_csv()
