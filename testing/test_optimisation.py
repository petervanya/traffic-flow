#!/usr/bin/env python
"""
Optimisation testing

24/09/20
"""
import os
import time
import pandas as pd
from mtsim import DiMTMig as MTM
from mtsim.sample_networks import load_network_2_directed

# loading data
df_nodes, df_link_types, df_links = load_network_2_directed()

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
