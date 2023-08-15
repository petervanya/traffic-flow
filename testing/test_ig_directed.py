#!/usr/bin/env python
"""
Test Igraph backend with directed sample network 2.

2020-10-28
"""
from mtsim import MTM
from mtsim.sample_networks import load_network_2

def test_ig_directed():
    df_n, df_lt, df_l = load_network_2()
    df_n["pop2"] = df_n["pop"] * 2

    mtm = MTM()

    # 1 =====
    # test what if no data is read, should throw a meaningful message
    #print(hasattr(mtm, "df_nodes"))
    #mtm.generate("all", "pop", "pop", 0.2)

    # 2 =====
    print("Production balancing with mobility 2")
    mtm = MTM()
    mtm.read_data(df_n, df_lt, df_l)
    mtm.generate("all", "pop", "pop2", 2)
    mtm.compute_skims()
    mtm.distribute("all", "t0", "exp", -0.1, balancing="production")
    mtm.assign("t0")
    print(mtm.df_links["geh"].mean())

    print("Attraction balancing with any mobility (should be same)")
    mtm.read_data(df_n, df_lt, df_l)
    mtm.generate("all", "pop", "pop2", 0.22)
    mtm.compute_skims()  # diagonal="area")
    mtm.distribute("all", "t0", "exp", -0.1, balancing="attraction")
    mtm.assign("t0")
    print(mtm.df_links["geh"].mean())


if __name__ == '__main__':
    test_ig_directed()
