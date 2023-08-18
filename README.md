# traffic-flow

A Python module for macroscopic transport modelling; forecasting future traffic flows after major infrastructure projects.

Installation:
```bash
pip install git+https://github.com/petervanya/traffic-flow.git
```

## Overview

### Key features
- A minimal, open-source and Pythonic alternative to closed-source products such as PTV Visum, Aimsum or Cube
- An implementation of the [three-step transport model](https://www.transitwiki.org/TransitWiki/index.php/Four-step_travel_model)
- Simple input structure in excel: nodes (zones and crossroads), links (road sections and connectors to zones) and link types (road types)
- Only one transport system available at the moment (eg passenger vehicles), more to be added in the future (work in progress)

On top of the features contained in closed-source products, this package enables a data-driven parameter optimisation (a simplified machine-learning) based on measured traffic flows.

### A quick overview of macroscopic transport modelling
Prior to modelling, one must create a network of road sections (links) and zones (nodes) with attributes representing the composition of the population. (NB: This step cannot be currently done with this package, we advise to use one of GIS packages or standard modelling products such as PTV Visum.)

Standard four-step modelling:
1. Trip generation: decide the demand stratum representing eg people daily travelling to work between zones
2. Trip distribution: compute the travel distance between the zones and distribute the generated trips
3. Mode choice: decide between private and public transport (not available here yet)
4. Assignment: assign the trips between the zones to the road network

On the output, there are traffic flows (eg 20000 vehicles/24h on a specific highway section), which can be compared with measured flows.

Then, a new link representing a new highway section can be added to the network and we can get a prediction for the traffic flow on the link with travel times. This can then serve as input for the cost-benefit analysis to estimate if the new section is worth building.

### Technical details
Multiple graph libraries (backends) exploited to represent the road network:
- [Igraph](https://igraph.org/python/), high-performance with C++ backend (default)
- [NetworkX](https://networkx.org/), purely Pythonic, lower performance

Both directed (default) and undirected networks are available.


## Inputs
The network is represented by three tables (dataframes):
* nodes (zones, crossroads and other point features)
* links (road sections or connectors to zones)
* link types (types of roads specified by their capacities and speeds)

In the simulation, these tables are read into a graph structure, which is then used to compute the shortest paths between zones and perform network assignment.

Each table has a set of columns that must be present:
- nodes: `[id, is_zone, name, pop]` (`pop` represents the population as a basic demand stratum)
- links: `[id, node_from, node_to, type, name, length, count]` (`count` for traffic flows can be empty)
- link types: `[type, type_name, v0, qmax, a, b]`

To better appreciate the logic of the input tables, please download one of the sample inputs (see below).


## Example

### Model inputs
Several sample networks are available:
* Network 1: minimal setting, 4 zones, 8 links
* Network 2: larger area, 12 zones, 42 links

Loading an example network:
```python
from traffic_flow.sample_networks import load_network_2

df_nodes, df_link_types, df_links = load_network_2()
```

### Simulation
```python
from traffic_flow import MTM  # macroscopic transport model

# initialise the object
model = MTM()

# read the input network data
model.read_data(df_nodes, df_link_types, df_links)

# trip generation, create a demand stratum
# 'pop' is a zone attribute containing population
model.generate("stratum-1", "pop", "pop", 0.5)

# calculate skim (resistance) matrices for distance and time
model.compute_skims()

# trip distribution of demand stratum 'stratum-1' using the gravity model
# distribute the trips exponentially with decay parameter -0.02 
# disutility defined as time 'tcur' computed above as a skim matrix
model.distribute("stratum-1", "tcur", "exp", -0.02)

# assign vehicle flows to network edges (ie road sections) based on real travel time:
model.assign("tcur")
```

As a result of assignment, the `model.df_links` attribute obtains the `q` column with predicted traffic flows.


## Data-driven optimisation of model parameters
This package enables tuning the generation and distribution parameters so as to achieve a minimal error between predicted and measured traffic flows. The objective function is the [GEH function](https://en.wikipedia.org/wiki/GEH_statistic).

```python
from traffic_flow import MTM

model = MTM()
model.read_data(df_nodes, df_link_types, df_links)
model.generate("stratum-1", "pop", "pop", 0.5)
model.compute_skims()
model.distribute("stratum-1", "tcur", "exp", -0.02)

# optimise using 10 iterations
model.optimise(n_iter=10)
```

Get the results:
```python
>>> print(model.opt_params)
```
|           |   attr_param |   dist_param |
|:----------|-------------:|-------------:|
| stratum-1 |     0.699951 |   -0.0581738 |

After optimisation, trip generation changed to 0.7 and the distribution exponent to -0.06 respectively.

Currently [dual annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html)
is the only available global optimisation method, others can be added upon request.


## Contributions
- Katarína Šimková, University of Glasgow/Vrije Universiteit Brussel

Started as a side project during an internship at the Ministry of Finance of Slovakia.

To contribute, please fork the repo, make *reasonable* changes (ideally contact me at `peter.vanya~gmail` before) and create a pull request.
