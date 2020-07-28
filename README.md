# macroscopic-traffic-simulator

A Python module for macroscopic transport modelling.

## Dependencies
numpy, pandas, networkx

## Inputs
Read data as Pandas dataframes of nodes, links and link types.

## Example
```
mtm = MTM()
mtm.read_data(df_nodes, df_lt, df_links)

# trip generation
mtm.generate("all", "pop", "pop", 0.5)

# skim (resistance) matrix calculation
mtm.compute_skims()

# trip distribution
mtm.distribute("all", "l", "exp", 0.02)

# assignment
mtm.assign("t0")
```

