# macroscopic-traffic-simulator

A minimal Python module for macroscopic transport modelling.
Implements the [three-step model](https://www.transitwiki.org/TransitWiki/index.php/Four-step_travel_model) 
(no mode choice) with only one transport system (for now, work in progress).

An exploration of multiple graph libraries and ways to represent the network 
(unoriented or oriented graph).

## Inputs
Any road network can be represented by three Pandas dataframes:
* nodes (zones, crossroads etc)
* links (road sections, graph edges)
* link types (types of roads)

To understand desired structure, download one of sample inputs.

## Example
Several examples are available:
* Network 1: minimal setting, 4 zones, 8 links
* Network 2: larger area, 12 zones, 42 links
* Directed network 2

Loading an example network:
```python
from mtsim.sample_networks import load_network_2

df_nodes, df_link_types, df_links = load_network_2()
```


## Simulation
```python
from mtsim import MTM

# initialise the object
mtm = MTM()

# read the road network data
mtm.read_data(df_nodes, df_link_types, df_links)

# trip generation, create a demand stratum
# 'pop' is a zone attribute containing population
mtm.generate("stratum-1", "pop", "pop", 0.5)

# calculate skim (resistance) matrices for distance and time
mtm.compute_skims()

# trip distribution of demand stratum 'stratum-1' using the gravity model
# distribute the trips exponentially with parameter 0.02 
# disutility defined as time 'tcur' computed above as a skim matrix
mtm.distribute("stratum-1", "tcur", "exp", -0.02)

# assign vehicle flows to network edges (ie road sections):
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
mtm.generate("stratum-1", "pop", "pop", 0.5)
mtm.compute_skims()
mtm.distribute("stratum-1", "tcur", "exp", -0.02)

# optimise using 10 iterations
mtm.optimise(10)
```
Currently only [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
available, other optimisation functions to be explored.


## Contributions
- Katarína Šimková, University of Glasgow

Started as a side project during an internship at the Ministry of Finance of Slovakia.
