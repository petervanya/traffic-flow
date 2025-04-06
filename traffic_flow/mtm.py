"""
Authors: Katarina Simkova and Peter Vanya
"""
import numpy as np
import pandas as pd
import time
import igraph as ig
import networkx as nx

from scipy.optimize import dual_annealing

from .parameters import ASSIGNMENT_KINDS, BASIC_SKIM_KINDS, DIST_FUNCS, BACKENDS
from .parameters import COLS_NODES, COLS_LINKS, COLS_LINK_TYPES
from .parameters import OPT_FUNS


class MTM:
    """
    Macroscopic transport modelling class.
    Using iGraph library and based on directed graphs,
    compatible with standard transport modelling software packages.
    """

    def __init__(self, backend="igraph", v_intra=40.0, verbose=False):
        """
        Inputs
        ------
        - backend : str,
            igraph or networkx
        - v_intra : float
            intrazonal speed in kmh
        - verbose : bool
            select True for more detailed information
        """
        backend = backend.lower()
        if backend not in BACKENDS:
            raise ValueError(f"choose backend from {BACKENDS}")
        self.backend = backend

        # initialise the graph object
        if backend == "igraph":
            self.G = ig.Graph(directed=True)
        elif backend == "networkx":
            self.G = nx.DiGraph()

        # dictionaries storing matrices
        self.skims = {}  # skim matrices
        self.dmats = {}  # demand matrices
        # demand strata
        self.dstrat = pd.DataFrame(columns=["prod", "attr", "param"])
        self.dpar = pd.DataFrame(
            columns=["skim", "func", "param", "symmetric"]
        )  # distribution params
        # optimisation results
        self.opt_params = pd.DataFrame(columns=["attr_param", "dist_param"])
        self.opt_output = pd.DataFrame(columns=["error", "nit", "nfev", "success"])
        # ad-hoc data
        self.v_intra = v_intra
        self.verbose = verbose

    def read_data(self, nodes, link_types, links):
        """
        Inputs
        ------
        - Nodes : pd.dataframe
            table containing columns as specified in `parameters.py`
        - Link types : pd.dataframe
            table containing columns as specified in `parameters.py`
        - Links : pd.dataframe
            table containing columns as specified in `parameters.py`
        """
        if self.verbose:
            print("Preparing nodes...")
        self.df_nodes = nodes.copy()
        self._verify_nodes()
        self.df_nodes.set_index("id", inplace=True)

        if self.verbose:
            print("Prepaging zones...")
        self.df_zones = self.df_nodes[self.df_nodes["is_zone"] == True]
        self.Nz = self.df_zones.shape[0]

        if self.verbose:
            print("Preparing link types...")
        self.df_lt = link_types.copy()
        self._verify_link_types()

        if self.verbose:
            print("Preparing link types...")
        self.df_links = links.copy()
        self._verify_links()
        self._assign_link_data()

        if self.verbose:
            print("Building the network graph...")
        self._fill_graph()

    def _verify_nodes(self):
        """Check that key columns are present"""
        assert np.isin(
            COLS_NODES, self.df_nodes.columns
        ).all(), f"node list does not contain expected columns: {COLS_NODES}"

    def _verify_link_types(self):
        """Verify columns"""
        assert np.isin(
            COLS_LINK_TYPES, self.df_lt.columns
        ).all(), f"link type list does not contain expected columns: {COLS_LINK_TYPES}"

    def _verify_links(self):
        """Verify if link type numbers subset of link types
        and if they connect the nodes"""
        assert np.isin(
            COLS_LINKS,
            self.df_links.columns,
        ).all(), f"link list does not contain expected columns: {COLS_LINKS}"

    def _assign_link_data(self):
        """
        Assign the attributes of the link types to the links:
        [v0, qmax, alpha, beta]
        Create new empty attributes for links:
        [t0, q, tcur, vcur]
        """
        # merge with link types
        self.df_links = self.df_links.merge(self.df_lt, how="left", on="type")
        self.df_links = self.df_links.set_index(["node_from", "node_to"])

        # assign empty attributes
        self.df_links["t0"] = (
            self.df_links["length"] / self.df_links["v0"] * 60.0
        )  # minutes
        self.df_links["q"] = 0.0
        self.df_links["tcur"] = self.df_links["t0"]
        self.df_links["vcur"] = self.df_links["v0"]

        # check if any values are missing
        assert self.df_links["type"].isna().any() == False, "Missing link types."

    def compute_tcur_links(self):
        """Compute current travel time and speed wrt the flows
        and assign to the graph G"""
        self.df_links["tcur"] = self.df_links["t0"] * (
            1.0
            + self.df_links["a"]
            * (self.df_links["q"] / self.df_links["qmax"]) ** self.df_links["b"]
        )

        self.df_links["vcur"] = self.df_links["length"] / self.df_links["tcur"] * 60.0

        # assign to the graph
        if self.backend == "igraph":
            self.G.es["tcur"] = self.df_links["tcur"].values
            self.G.es["vcur"] = self.df_links["vcur"].values

        elif self.backend == "networkx":
            nx.set_edge_attributes(self.G, self.df_links["tcur"], "tcur")
            nx.set_edge_attributes(self.G, self.df_links["vcur"], "vcur")

    def _fill_graph(self):
        """Fill the graph with read-in nodes and links"""
        """
        KS: making sure the graph is empty (needed when self.read_data is run
        multiple times in the code, but MTM does not initialise at every run)
        """
        if self.backend == "igraph":
            if len(self.G.vs) > 0:
                self.G = ig.Graph(directed=True)

            # adding vertices
            att_nodes = {}
            att_nodes["id"] = self.df_nodes.index
            for k, v in self.df_nodes.items():
                att_nodes[k] = v.values

            self.G.add_vertices(self.df_nodes.shape[0], attributes=att_nodes)

            # adding edges
            # create mapping between node index and id attribute
            nodes_to_vertices = pd.DataFrame(
                {"id": self.G.vs["id"], "index": self.G.vs.indices}
            )
            nodes_to_vertices = nodes_to_vertices.set_index("id")

            self.df_links = self.df_links.reset_index()
            self.df_links["vertex_from"] = nodes_to_vertices.loc[
                self.df_links["node_from"]
            ].values
            self.df_links["vertex_to"] = nodes_to_vertices.loc[
                self.df_links["node_to"]
            ].values

            self.df_links = self.df_links.set_index(["node_from", "node_to"])

            # get list of all graph edges
            list_edges = zip(self.df_links["vertex_from"], self.df_links["vertex_to"])

            # drop temporary columns
            self.df_links = self.df_links.drop(columns=["vertex_from", "vertex_to"])

            att_links = {}
            for k, v in self.df_links.items():
                att_links[k] = v.values

            self.G.add_edges(list_edges, attributes=att_links)

            del att_links
            del att_nodes

        elif self.backend == "networkx":
            for k, v in self.df_nodes.iterrows():
                self.G.add_node(k, **v)

            for k, v in self.df_links.iterrows():
                self.G.add_edge(k[0], k[1], **v)

    # =====
    # Trip generation
    # =====
    def generate(self, name, prod, attr, param):
        """
        Generate the key-value pairs of node columns.
        Apply the method separately for each demand stratum.

        Inputs
        ------
        - prod : production zone attribute
        - attr : attraction zone attribute
        - param : mobility parameter, mean number of trips per day weighted
            by the fraction of the population making the trips
        """
        assert hasattr(
            self, "df_nodes"
        ), "no input dataframe of nodes found, did you read it?"
        assert (
            prod in self.df_nodes.columns
        ), "production attribute not found in node columns"
        assert (
            attr in self.df_nodes.columns
        ), "attraction attribute not found in node columns"

        if (self.df_zones[prod] < 1.0).any():
            print(f"Warning: Zone production {prod} contains zeros")
        if (self.df_zones[attr] < 1.0).any():
            print(f"Warning: Zone attraction {attr} contains zeros.")

        # convert to floats
        self.df_nodes[prod] = self.df_nodes[prod].astype(float)
        self.df_nodes[attr] = self.df_nodes[attr].astype(float)

        self.dstrat.loc[name] = [prod, attr, param]

    # =====
    # Skim/impedance matrices
    # =====
    def compute_skims(
        self, diagonal="density", density=1000.0, diagonal_param=0.5, fillna=True
    ):
        """
        Compute skim matrices, choose from the following:
        - "length" : distance between zones
        - "t0" : free flow travel time between zones
        - "tcur" : traffic travel time between zones
        """
        kw = {
            "diagonal": diagonal,
            "density": density,
            "diagonal_param": diagonal_param,
            "fillna": fillna,
        }

        self._compute_skim_basic("length", **kw)
        self._compute_skim_basic("t0", **kw)
        self._compute_skim_basic("tcur", **kw)

    def _compute_skim_basic(
        self, kind, diagonal="density", density=1000.0, diagonal_param=0.5, fillna=True
    ):
        """
        General method to compute skim matrices from basic quantities
        (free flow time, current time or distance).
        For the matrix diagonal, "density" chosen, compute zone area from
        the chosen population density and then the distance.
        If "area" chosen, use the "area" zone attribute and compute distance.
        In both cases, scale the distance by the `"diagonal_param"`.
        For time-based skim matrices, convert distance to time via intrazonal
        speed `"v_intra"`.

        Parameters
        ----------
        - kind : string
            Choose from `"t0"`, `"tcur"`, `"length"`
        - diagonal : string, optional
            A flag to determine the way to compute ompute the matrix diagonal.
            If `"density"` chosen, zone area is computed from the population
            density supplied in the `density` keyword and from it the distance
            as a square root. Choice `"area"` computes distance as a square
            root of the zone attribute "area".
        - diagonal_param : float, optional
            Parameter to scale the distance computed from the area
        - density : float, optional
            Average population density per zone
        - fillna : bool, optional
            True if existing nan values are to be filled by a large number
        """
        if kind not in BASIC_SKIM_KINDS:
            raise ValueError(f"Choose kind among {BASIC_SKIM_KINDS}")

        # get shortest paths
        if self.backend == "igraph":
            paths = self.G.shortest_paths(
                source=self.G.vs.select(is_zone_eq=True),
                target=self.G.vs.select(is_zone_eq=True),
                weights=kind,
            )

            self.skims[kind] = pd.DataFrame(paths)
            self.skims[kind].index = self.G.vs.select(is_zone_eq=True)["id"]
            self.skims[kind].columns = self.G.vs.select(is_zone_eq=True)["id"]

        elif self.backend == "networkx":
            paths = nx.all_pairs_dijkstra_path_length(self.G, weight=kind)

            self.skims[kind] = pd.DataFrame(dict(paths)).loc[
                self.df_zones.index, self.df_zones.index
            ]

        # compute diagonal based on distance
        if diagonal == "density":
            np.fill_diagonal(
                self.skims[kind].values,
                np.sqrt(self.df_zones["pop"].values / density) * diagonal_param,
            )
        else:
            assert (
                "area" in self.df_zones.columns
            ), "'area' not found among zone attributes."
            np.fill_diagonal(
                self.skims[kind].values,
                np.sqrt(self.df_zones["area"].values) * diagonal_param,
            )

        # adjust time-related skims
        if kind in ["t0", "tcur"]:
            np.fill_diagonal(
                self.skims[kind].values,
                self.skims[kind].values.diagonal() / self.v_intra * 60.0,
            )

        # check for nan's or inf's
        if self.skims[kind].isin([np.nan, np.inf, -np.inf]).values.any():
            vals = np.prod(self.skims[kind].shape)
            vals_bad = np.logical_or(
                np.isinf(self.skims[kind].values), np.isnan(self.skims[kind].values)
            ).sum()
            pc = vals_bad / vals * 100
            if fillna:
                self.skims[kind].replace([np.inf, -np.inf, np.nan], 1e6, inplace=True)
                print(
                    f"Warning: nan's/inf's in skim matrix '{kind}', {vals_bad} values ({pc:.2f}%) filling with 1e6"
                )
            else:
                print(
                    f"Warning: nan's/inf's in skim matrix '{kind}', {vals_bad} values ({pc:.2f}%)"
                )

    def compute_skim_utility(self, name, params):
        """Compute the utility skim matrix composed of several
        basic link attributes (distance or times) and their unit
        values and specific link attributes"""
        raise NotImplementedError

    # =====
    # Trip distribution
    # =====
    def dist_func(self, func, C, beta):
        if func == "power":
            try:
                iter(beta)
            except TypeError:
                print("power law parameters should be in a list")
            assert len(beta) == 2, "power law function must have two parameters"

        if func == "exp":
            return np.exp(beta * C)
        elif func == "poly":
            return C**beta
        elif func == "power":
            return (C + beta[1]) ** beta[0]

    def distribute(
        self, ds, C, func, param, n_iter=10, balancing="production", symm=True
    ):
        """
        Compute OD matrices for a given demand stratum
        via a doubly constrained iterative algorithm

        Inputs
        ------
        - ds : str
            demand stratum
        - C : str
            cost function as skim matrix, t0, tcur, length or utility
        - func : str
            distribution function
        - param : float
            parameter of the distribution function
        - n_iter : int, optional
            number of iterations
        - balancing : str, optional
            normalisation of trips wrt production or attraction
        - symm : bool, optional
            symmetrise the demand matrix
        """
        assert ds in self.dstrat.index, f"{ds} not found in demand strata"
        assert C in self.skims.keys(), f"cost {C} not found among skim matrices"
        assert func in DIST_FUNCS, f"choose distribution function from {DIST_FUNCS}"
        assert n_iter > 0, "number of iterations must be positive number"
        assert balancing in [
            "production",
            "attraction",
        ], "incorrect choice of balancing"
        assert symm in [True, False], "choose True/False for matrix symmetrisation"
        if func == "power" and param[0] >= 0.0:
            print("warning: parameter of decay should be < 0")
        elif func != "power" and param >= 0.0:
            print("warning: parameter of decay should be < 0")

        # define set of distribution parameters
        self.dpar.loc[ds] = [C, func, param, symm]

        O = (
            self.df_zones[self.dstrat.loc[ds, "prod"]].values.copy()
            * self.dstrat.loc[ds, "param"]
        )
        D = self.df_zones[self.dstrat.loc[ds, "attr"]].values.copy()

        if balancing == "production":
            D *= O.sum() / D.sum()  #  normalisation wrt production
        elif balancing == "attraction":
            O *= D.sum() / O.sum()  # normalisation wrt attraction

        a, b = np.ones_like(O), np.ones_like(D)
        T = np.zeros((self.Nz, self.Nz))
        T = np.outer(O, D) * self.dist_func(func, self.skims[C].values, param)

        for i in range(n_iter):
            a = O / T.sum(1)
            T = T * np.outer(a, b)
            b = D / T.sum(0)
            if np.isnan(T).any():
                print(f"warning: nan's in OD matrix in iteration {i}")

        # compute final mean average errors
        self.dist_errs = {
            "O": (abs(T.sum(1) - O)).sum() / len(T) / self.Nz,
            "D": (abs(T.sum(0) - D)).sum() / len(T) / self.Nz,
        }
        if symm:
            T = (T + T.T) / 2.0
        self.dmats[ds] = pd.DataFrame(
            T, columns=self.df_zones.index, index=self.df_zones.index
        )

    # =====
    # Assignment
    # =====
    def assign(self, imp, kind="incremental", weights=[50, 50]):
        """
        Assign demand matrix to the network.
        Use only one transport system here.

        1. Sum all demand matrices.
        2. For each weight, calculate all the shortest paths
           by given impedance (distance or current time)
           between two zones and add to the links on the path.
           Update current time.

        KS: Impedance in assignment is defined on graph's edges, not from
        skim matrices. Changed "assert imp in self.skims.keys()" and its
        error prompt to "assert imp in self.basic_skim_kinds".
        Alternatively, it could be "assert imp in mtm.G.es.attributes()"
        but then it's risky that other attribute gets called by accident.

        Inputs
        ------
        - imp : str
            impedance (link attribute) for path search, one of skim matrices
        - kind : str
            type of assignment, now only incremental
        - weights : iterable
            assignment weights
        """
        if kind not in ASSIGNMENT_KINDS:
            raise ValueError(f"choose assignment from {ASSIGNMENT_KINDS}")
        if imp not in BASIC_SKIM_KINDS:
            raise ValueError(f"choose impedance among {BASIC_SKIM_KINDS}")

        weights = np.array(weights)
        weights = weights / weights.sum()  # normalise weights

        # sum all demand matrices into one
        self.DM = sum(self.dmats.values())

        # remove flows and reset tcur/vcur before assignment
        """
        KS: added resetting the tcur and vcur on graph to t0 and v0, since
        this caused the inconsistency of 1 run of the module with the module
        running in the loop if read_data was not in the loop
        """
        if self.backend == "igraph":
            self.G.es["q"] = 0.0
            self.G.es["tcur"] = self.df_links["t0"].values
            self.G.es["vcur"] = self.df_links["v0"].values
            self.df_links["q"] = 0.0
        elif self.backend == "networkx":
            nx.set_edge_attributes(self.G, 0.0, "q")

        # perform assignment
        if kind == "incremental":
            for wi, w in enumerate(weights):
                if self.verbose:
                    print(f"Assigning batch {wi}, weight {w:.2f} ...")

                if self.backend == "igraph":
                    vs_zones = [v.index for v in self.G.vs.select(is_zone_eq=True)]

                    for i in vs_zones:
                        p = self.G.get_shortest_paths(
                            v=i, to=vs_zones, weights=imp, output="epath"
                        )
                        dq = self.DM.loc[self.G.vs[i]["id"], :].values * w
                        for j, _ in enumerate(dq):
                            self.G.es[p[j]]["q"] += dq[j]

                    self.df_links["q"] = self.G.es["q"]

                elif self.backend == "networkx":
                    paths = dict(nx.all_pairs_dijkstra_path(self.G, weight=imp))

                    for i in self.df_zones.index:
                        for j in self.df_zones.index:
                            p = paths[i][j]
                            dq = self.DM.loc[i, j] * w
                            for k, _ in enumerate(p[:-1]):
                                self.G.edges[p[k], p[k + 1]]["q"] += dq

                    self.df_q = nx.to_pandas_edgelist(self.G).set_index(
                        ["source", "target"]
                    )

                    self.df_links["q"] = self.df_q["q"]

                self.compute_tcur_links()  # update current time and speed

    def compute_error(self, measured_col="count"):
        """Compuate precision metric wrt measured flows"""
        if measured_col not in self.df_links.columns:
            raise ValueError(f"{measured_col} not found among link attributes")
        self._geh(measured_col)
        self._var_geh(measured_col)
        print(f"Average error: {self.df_links['geh'].mean()}")

    # =====
    # Error-measuring tools
    # =====
    def _geh(self, measured_col):
        """Compute the GEH error of each link with a measurement"""
        self.df_links["geh"] = np.sqrt(
            2.0
            * (self.df_links["q"] - self.df_links[measured_col]) ** 2
            / (self.df_links["q"] + self.df_links[measured_col])
            / 10.0
        )

    def _geh_vehkm(self, measured_col):
        """Compute GEH adjusted for section lengths"""
        l = self.df_links["length"]
        self.df_links["geh"] = np.sqrt(
            2.0
            * (self.df_links["q"] * l - self.df_links[measured_col] * l) ** 2
            / (self.df_links["q"] * l + self.df_links[measured_col] * l)
            / 10.0
        )

    def _var_geh(self, measured_col):
        """Compute GEH as a variance without the square root"""
        self.df_links["var_geh"] = (
            2.0
            * (self.df_links["q"] - self.df_links[measured_col]) ** 2
            / (self.df_links["q"] + self.df_links[measured_col])
            / 10.0
        )

    def _var_geh_vehkm(self, measured_col):
        """Compute GEH as a variance without the square root and
        adjusted for section lengths"""
        l = self.df_links["length"]
        self.df_links["var_geh"] = (
            2.0
            * (self.df_links["q"] * l - self.df_links[measured_col] * l) ** 2
            / (self.df_links["q"] * l + self.df_links[measured_col] * l)
            / 10.0
        )

    # =====
    # Optimisation
    # =====
    def optimise(
        self,
        n_iter,
        optfun="dual-annealing",
        x0=None,
        bounds=None,
        skim="tcur",
        seed=1101,
        weights=[100],  # [50, 50],
    ):
        """
        Global optimisation of model parameters.

        Inputs
        ------
        - n_iter : number of iterations
        - optfun : optimisation function from Scipy
        - x0 : initial estimates of the parameters
        - imp : assignment impedance (t0, tcur, l)
        - seed : random seed
        - ws : weights in incremental assignment
        """
        # basic checks
        assert (
            len(self.dstrat) > 0
        ), "no demand strata defined, need to run trip generation first"
        if optfun not in OPT_FUNS:
            raise ValueError(f"choose optimisation functions from {OPT_FUNS}")

        # compute the number of optimisation parameters
        n_param = 0
        for ds in self.dstrat.index:
            par = 2 if self.dpar.loc[ds, "func"] == "power" else 1
            n_param += par + 1

        # compose list of bounds if required
        if optfun in ["dual-annealing"]:
            if bounds == None:
                bounds = []
                for m in self.dstrat.index:
                    bounds += [(1e-6, 3.0)]
                    par = 2 if self.dpar.loc[ds, "func"] == "power" else 1
                    if par == 1:  # exp, poly
                        bounds += [(-3.0, -1e-6)]
                    else:  # power law
                        bounds += [(0.0, 50.0), (-5.0, -1e-6)]
            else:
                assert len(bounds) == n_param, "incorrect number of bounds"

        optargs = (skim, weights)
        if "geh" not in self.df_links.columns:
            self.compute_error()
        print(f"Initial error: {self.df_links['geh'].mean()}")

        # optimisation core
        tic = time.time()
        if optfun == "dual-annealing":
            res = dual_annealing(
                self._obj_function,
                args=optargs,
                bounds=bounds,
                seed=seed,
                maxiter=n_iter,
            )

        # elif optfun == "gradient-descent":
        #     """Optimize using gradient descent with given dh"""
        #     thermo = 10
        #     lmbda = 1e-7
        #     decay = 0.99
        #     print(thermo, lmbda, decay)

        #     if x0 is None:
        #         raise ValueError(f"gradient descent requires x0")

        #     X = np.zeros((n_iter + 1, len(x0)))
        #     X[0] = x0
        #     f = np.zeros(n_iter + 1)
        #     f[0] = self._obj_function(x0, *optargs)
        #     print(f"Starting gradient descent: {f[0]}, {X[0]}")

        #     for i in range(n_iter):
        #         G = grad(self._obj_function, X[i - 1], *optargs)
        #         print("STEP", i, X[i], G)
        #         X[i] = X[i - 1] - lmbda * G
        #         f[i] = self._obj_function(X[i], *optargs)

        #         if i % thermo == 0:
        #             print(f"step {i} {f[i]}, {X[i]}, {G}")

        #         if i % 20 == 0:
        #             lmbda *= decay

        toc = time.time()

        if optfun == "dual-annealing":
            print(f"Optimisation terminated. Success: {res.success}")
            print(f"Resulting parameters: {res.x}")
            print(f"Resulting error: {res.fun}")
        elif optfun == "gradient-descent":
            raise NotImplementedError

        print("Time: %.2f s" % (toc - tic))

        for m, n in enumerate(self.dstrat.index):
            self.opt_params.loc[n] = [res.x[2 * m], res.x[2 * m + 1]]

        self.opt_output.loc[1] = [res.fun, res.nit, res.nfev, res.success]

    def _obj_function(self, z, imp, weights=[50, 50], measured_col="count"):
        """
        The sum of all GEH differences between traffic counts and modelled
        flows on links that contain the counts.
        Serves as the objective function for the optimisation.
        Goes through all transport modelling steps.

        Inputs
        ------
        - z : iterable
            array of parameters to optimise
        - imp : str
            impedance kind: t0, tcur, length
        - ws : iterable
            assignment weights
        """
        # basic checks
        assert len(self.dstrat) > 0, "no demand strata defined"

        # trip generation
        for m, n in enumerate(self.dstrat.index):
            self.generate(n, self.dstrat["prod"][m], self.dstrat["attr"][m], z[2 * m])

        # trip distribution
        for m, n in enumerate(self.dpar.index):
            self.distribute(
                n,
                self.dpar["skim"][m],
                self.dpar["func"][m],
                z[2 * m + 1],
                symm=self.dpar["symmetric"][m],
            )

        # assignment
        self.assign(imp=imp, weights=weights)

        relevant = self.df_links[
            np.logical_and(
                (np.logical_not(np.isnan(self.df_links[measured_col]))),
                (self.df_links[measured_col] != 0),
            )
        ]

        # compute error
        gehs = np.sqrt(
            2.0
            * (relevant["q"].values - relevant[measured_col].values) ** 2
            / (relevant["q"].values + relevant[measured_col].values)
        ) / np.sqrt(10.0)

        #     reg = alpha*(\
        #             self.dmats["all"].multiply(self.skims["t0"]).sum().sum()\
        #                         /self.dmats["all"].sum().sum() - xexp)**2
        #     vals = np.append(vals, suma)

        return np.mean(gehs)  # + reg

    # =====
    # Processing functions
    # =====
    def compute_mean_trip_length(self, ds):
        """Compute the mean trip length for a given demand stratum"""
        assert ds in self.dstrat.index, "Demand stratum not available."
        return (self.skims["length"] * self.dmats[ds]).sum().sum() / self.dmats[
            ds
        ].sum().sum()


"""
Helper functions
"""


def grad(func, X, *args, h=1e-8):
    """Compute gradient of a function at a given point X"""
    # dX = np.ones_like(X)
    dX = 1e-5
    G = np.zeros_like(X)
    for i, _ in enumerate(X):
        dX = np.zeros_like(X)
        dX[i] = h
        G[i] = (func(X + dX, *args) - func(X - dX, *args)) / (2 * h)
        print("grad", G)
    return G
