# macroscopic-traffic-simulator

A minimal Python module for macroscopic transport modelling.
Work in progress.


## Contributions
Katarina Šimková, University of Glasgow


## Dependencies
numpy, pandas, [networkx](https://networkx.github.io/documentation/latest/)


## Inputs
Read road network data as Pandas dataframes. Three kinds required:
* nodes (zones, crossroads etc)
* links (road sections)
* link types (types of roads)

Example networks available in the `data` directory.
Automatic loading functions to be added soon.


## Example
Running the simulation:
```python
from mtsim import MTM

mtm = MTM()

# read the road network data in the correct format
mtm.read_data(df_nodes, df_lt, df_links)

# trip generation, 'pop' is a zone attribute containing population
mtm.generate("all", "pop", "pop", 0.5)

# skim (resistance) matrix calculation
mtm.compute_skims()

# trip distribution
mtm.distribute("all", "l", "exp", 0.02)

# assignment
mtm.assign("t0")
```

