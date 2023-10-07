#!/usr/bin/env python
"""
Test reading inputs.

2023-09-23
"""
import time

from traffic_flow import MTM
from traffic_flow.utils import read_inputs_excel, read_inputs_shapefile

def test_reading_csv():
    print('Testing raw input reading...')
    fname = 'Internal/Networks/network_raw_I51_201002.xlsx'
    df_nodes, df_link_types, df_links = read_inputs_excel(fname)

    # first few steps
    model = MTM()

    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    toc = time.time()
    print("Basic cycle done. Time: %.3f s" % (toc - tic))

def test_reading_shapefile_ptv():
    print('Testing PTV shapefile reading...')
    basepath = '/Users/peter/Tatra/UHP/Res/Transport_Models/Data/Shapefiles_I51/nulovy_stav/'
    basename = 'I51_I76_siet_nulovy_stav'
    df_nodes, df_link_types, df_links = read_inputs_shapefile(basepath, basename)

    # first few steps
    model = MTM()

    tic = time.time()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    toc = time.time()
    print("Basic cycle done. Time: %.3f s" % (toc - tic))



if __name__ == '__main__':
    test_reading_csv()
    test_reading_shapefile_ptv()