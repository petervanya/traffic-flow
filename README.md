# macroscopic-traffic-simulator

A minimal Python module for macroscopic transport modelling.
Implements the [three-step model](https://www.transitwiki.org/TransitWiki/index.php/Four-step_travel_model) 
(no mode choice) with only one transport system.

Work in progress.

An exploration of multiple graph libraries and ways to represent the network 
(unoriented or oriented graph).


## Contributions
Katarína Šimková, University of Glasgow


## Dependencies
* numpy
* pandas
* [networkx](https://networkx.github.io/documentation/latest/)
* [igraph](https://igraph.org/python/)

## Inputs
Any road network is represented by three pandas dataframes:
* nodes (zones, crossroads etc)
* links (road sections, graph edges)
* link types (types of roads)


## Example
Several examples were created:
* Network 1: minimal setting, 4 zones, 8 links
* Network 2: larger area, 12 zones, 42 links
* Directed network 2

Loading an example network:
```python
from mtsim.sample_networks import load_network_2

df_nodes, df_link_types, df_links = load_network_2()
```


Running the simulation:
```python
from mtsim import MTM

mtm = MTM()

# read the road network data
mtm.read_data(df_nodes, df_link_types, df_links)

# trip generation, create a demand stratum
# 'pop' is a zone attribute containing population
mtm.generate("stratum_1", "pop", "pop", 0.5)

# calculate skim (resistance) matrices for distance and time
mtm.compute_skims()

# trip distribution of demand stratum 'stratum_1' using the gravity model
# distribute the trips exponentially with parameter 0.02 
# disutility defined as time 'tcur' computed above as a skim matrix
mtm.distribute("stratum_1", "tcur", "exp", -0.02)

# assign vehicle flows to the network
mtm.assign("tcur")
```

## Optimisation of model parameters
The module contains a method to tune the generation and distribution parameters 
to minimise the objective function, which is defined as an error of modelled and
measured traffic flows via the [GEH function](https://en.wikipedia.org/wiki/GEH_statistic).

Having done the above described cycle, run the following:
```python
from mtsim import MTM

mtm = MTM()
mtm.read_data(df_nodes, df_link_types, df_links)
mtm.generate("stratum_1", "pop", "pop", 0.5)
mtm.compute_skims()
mtm.distribute("stratum_1", "tcur", "exp", -0.02)

# optimise using 10 iterations
mtm.optimise(10)
```
Currently only [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
available, other optimisation functions to be explored.



