# simulation options to select
BACKENDS = ["networkx", "igraph"]
ASSIGNMENT_KINDS = ["incremental"]
BASIC_SKIM_KINDS = ["t0", "tcur", "length"]
DIST_FUNCS = ["exp", "poly", "power"]

# table columns
COLS_NODES = ["id", "is_zone", "name"]
COLS_LINKS = ["id", "node_from", "node_to", "type", "length"]
COLS_LINK_TYPES = ["type", "type_name", "v0", "qmax", "a", "b"]

# optmisation
OPT_FUNS = ["dual-annealing", "nelder-mead", "gradient-descent"]
