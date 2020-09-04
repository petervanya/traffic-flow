# macroscopic-traffic-simulator

A minimal Python module for macroscopic transport modelling.
Work in progress.

An exploration of multiple graph libraries and ways to represent the network 
(unoriented or oriented graph).


## Contributions
Katarína Šimková, University of Glasgow


## Dependencies
* numpy
* pandas
* [networkx](https://networkx.github.io/documentation/latest/)


## Inputs
Any road network is represented by three pandas dataframes:
* nodes (zones, crossroads etc)
* links (road sections, graph edges)
* link types (types of roads)


## Example
Several examples were created:
1. Network 1: 4 zones, 8 links
2. Network 2: 12 zones, 42 links

Loading example networks:
```python
from mtsim.sample_networks import load_network_1

df_nodes, df_link_types, df_links = load_network_1()
```


Running the simulation:
```python
from mtsim import MTMnx

mtm = MTMnx()

# read the road network data
mtm.read_data(df_nodes, df_link_types, df_links)

# trip generation, create a demand stratum
# 'pop' is a zone attribute containing population
mtm.generate("all", "pop", "pop", 0.5)

# calculate skim (resistance) matrices for distance and time
mtm.compute_skims()

# trip distribution of demand stratum 'all'
# distribute exponentially with parameter 0.02 
# via distance 'l' computed as a skim matrix
mtm.distribute("all", "l", "exp", 0.02)

# assignment of vehicle flows
mtm.assign("t0")
```


