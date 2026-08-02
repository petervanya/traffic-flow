# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A Python library for forecasting road traffic flows on roads.
It implements the classic three-step travel model (trip generation → distribution → assignment).
There is an extra feature: data-driven calibration of model parameters against measured traffic counts.

It is used as a library for importing into scripts and notebooks. Conceptually similar to scikit-learn.

## Package layout

- `traffic_flow/` — the shipped package
  - `mtm.py` — **canonical, actively-developed** `MTM` class. Directed graphs, dual-backend
    (`igraph` default, `networkx`). Contains the full pipeline (generate/skims/distribute/assign)
    plus optimisation. **This is the file to modify for most changes.**
  - `mtm_ig_optimisation.py`, `mtm_nx.py` — older, parallel/legacy variants of the model, not
    exported from `__init__.py`. They duplicate logic rather than sharing a base class with
    `mtm.py` — don't assume a change to `MTM` needs to (or does) propagate here.
  - `mtm_ig_undirected.py` (`MTMUndirected`), `mtm_nx_undirected.py` (`MTMnxUndirected`) —
    undirected-network variants, exported.
  - `parameters.py` — single source of truth for valid option strings and required table columns:
    `BACKENDS`, `ASSIGNMENT_KINDS`, `BASIC_SKIM_KINDS`, `DIST_FUNCS`, `OPT_FUNS`, `COLS_NODES`,
    `COLS_LINKS`, `COLS_LINK_TYPES`. Check here before hardcoding option lists elsewhere.
  - `sample_networks.py` — bundled example loaders: `load_network_1()`, `load_network_2()`, and
    `_undirected` variants.
  - `utils.py` — to read inputs from external software like PTV Visum: `read_inputs_excel()`, `read_inputs_shapefile()`.
  - `examples/*.xlsx` — bundled sample data (network_1, network_2, and undirected variants).
- `testing/` — pytest suite plus exploratory Jupyter notebooks. Note the directory is `testing/`,
  not `tests/`.
- `Internal/`, `Pip_Testing/`, `venv/`, `dist/`, `*.egg-info/` — research material, throwaway
  venvs, and build artifacts. Not part of the shipped package; generally out of scope.

## Domain model and pipeline

Three input tables (pandas DataFrames) define a network:
- **nodes** — `[id, is_zone, name, pop]`. A node is a **zone** (`is_zone=True`, trip
  generator/attractor) or a crossroad/junction.
- **links** — `[id, node_from, node_to, type, length]` (optionally `count` = measured flow, used
  for calibration). A road section or a connector (connecting zones to the network).
- **link_types** — `[type, type_name, v0, qmax, a, b]`: free-flow speed `v0`, capacity `qmax`,
  and BPR volume-delay coefficients `a`, `b`.

Pipeline on an `MTM` instance:
1. `read_data(nodes, link_types, links)` — validates columns, builds the graph.
2. `generate(name, prod, attr, param)` — trip generation; registers a demand **stratum** with
   production/attraction zone attributes and a mobility parameter (trips/person).
3. `compute_skims()` — zone-to-zone skim/impedance matrices via shortest paths: `t0` (free-flow
   time), `tcur` (congested time), `length`.
4. `distribute(stratum, skim, func, param)` — gravity model with doubly-constrained Furness/IPF
   balancing, producing an OD matrix. Deterrence functions (`DIST_FUNCS`): `exp`, `poly`, `power`.
5. `assign(imp)` — incremental, capacity-restrained assignment over shortest paths; updates link
   flow `q` and congested time via the BPR function `tcur = t0*(1 + a*(q/qmax)^b)`.
6. `optimise(...)` (optional) — tunes generation/distribution parameters to minimize mean GEH
   error against the measured `count` column. Methods (`OPT_FUNS`): `dual-annealing` (default),
   `nelder-mead`, `grid-search`, `gradient-descent` (stubbed). Supports train/test splitting of
   measured links for validation.

Backends: `igraph` (default, fast C core) vs `networkx` (pure Python). Networks can be directed
(default, `MTM`) or undirected (`MTMUndirected`, `MTMnxUndirected`).

There are no config files (YAML/JSON) — everything is configured via constructor/method
arguments and the `dstrat`/`dpar` tables built up at runtime on the model instance.

## Typical usage

```python
from traffic_flow import MTM

model = MTM(backend="igraph")
model.read_data(df_nodes, df_link_types, df_links)
model.generate("stratum-1", "pop", "pop", 0.5)
model.compute_skims()
model.distribute("stratum-1", "tcur", "exp", -0.02)
model.assign("tcur")   # results land in model.df_links["q"]
```

## Dev workflow

- **Install**: `pip install -e .`. `setup.py` is the authoritative packaging source (deps:
  numpy, pandas, scipy, geopandas, openpyxl, networkx, python-igraph). `tmp.pyproject.toml` is
  stale/inconsistent with `setup.py` — ignore it.
- **Test**: run individual files, not the CI glob, since `test_inputs.py` hardcodes
  machine-specific absolute paths under `Internal/` that don't exist on a fresh checkout:
  ```bash
  pytest testing/test_pipelines.py testing/test_pipelines_undirected.py \
         testing/test_ig_directed.py testing/test_optimisation.py
  ```
- **Lint**: flake8 only, run in CI (`.github/workflows/python-package.yml`); no local config file,
  no mypy/ruff/pre-commit/Makefile.
- **Git workflow**: feature branches → PR → merge to `master`. CI runs only on `master`.
