import numpy as np
import pandas as pd
import networkx as nx

from .parameters import ASSIGNMENT_KINDS, BASIC_SKIM_KINDS


class MTMnx:
    """
    Macroscopic transport modelling class.
    Using NetworkX library and based on directed graphs
    for compatiblity with standard modelling software.
    """

    def __init__(self, v_intra=40.0, verbose=False):
        """
        Inputs
        ======
        - v_intra : intrazonal speed in kmh
        - verbose : level of printing information
        """
        self.G = nx.DiGraph()  # the graph structure
        self.skims = {}  # skim matrices
        self.dmats = {}  # demand matrices
        self.dstrat = pd.DataFrame(columns=["prod", "attr", "param"])  # demand strata
        self.dpar = pd.DataFrame(
            columns=["skim", "func", "param"]
        )  # distribution parameters

        # ad-hoc data
        self.v_intra = v_intra
        self.verbose = verbose

    def read_data(self, nodes, link_types, links):
        """
        Inputs
        ======
        - Nodes : dataframe [is_zone, name, coords, pop], index: id
        - Link types : dataframe [name, v0, qmax, a, b], index: id
        - Links : dataframe [id, type, name, length, count], index: id_node_pair
        """
        self.df_nodes = nodes.copy()
        self._verify_nodes()
        self.df_nodes.set_index("id", inplace=True)

        self.df_zones = self.df_nodes[self.df_nodes["is_zone"] == True]
        self.Nz = self.df_zones.shape[0]

        self.df_lt = link_types.copy()
        self._verify_lt()

        self.df_links = links.copy()
        self._verify_links()
        self._assign_link_data()

        self._fill_graph()

    def _verify_nodes(self):
        """Verify columns 
        ADD INDEX"""
        assert np.isin(
            ["id", "is_zone", "name", "pop"], self.df_nodes.columns
        ).all(), "Node list does not have the expected structure."

    def _verify_lt(self):
        """Verify columns"""
        assert np.isin(
            ["type", "type_name", "v0", "qmax", "a", "b"], self.df_lt.columns
        ).all(), "Link type list does not have the expected structure."

    def _verify_links(self):
        """Verify if link type numbers subset of link types
        and if they connect the nodes"""
        assert np.isin(
            ["id", "type", "name", "length", "count", "node_from", "node_to"],
            self.df_links.columns,
        ).all(), "Link list does not have the expected structure."

    def _assign_link_data(self):
        """
        Assign the attributes of the link types to the links:
        [v0, qmax, alpha, beta]
        Create new empty attributes:
        [t0, q, tcur, vcur]
        """
        # merge with link types
        self.df_links = self.df_links.merge(self.df_lt, how="left", on="type")
        self.df_links = self.df_links.set_index(["node_from", "node_to"])

        # assign empty attributes
        self.df_links["t0"] = self.df_links["length"] / self.df_links["v0"] * 60.0
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
        nx.set_edge_attributes(self.G, self.df_links["tcur"], "tcur")
        nx.set_edge_attributes(self.G, self.df_links["vcur"], "vcur")

    def _fill_graph(self):
        """Fill the graph with read nodes and links"""
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
        ======
        - prod : productivity column
        - attr : attractivity column
        - param : parameter with which to multiply the productivity
            (mean number of trips per day, fraction of the population)
        """
        assert (
            prod in self.df_nodes.columns
        ), "Production attribute not found in node columns."
        assert (
            attr in self.df_nodes.columns
        ), "Attraction attribute not found in node columns."

        self.dstrat.loc[name] = [prod, attr, param]

    # =====
    # Skim matrices
    # =====
    def compute_skims(self, diagonal="density", density=1000.0):
        """
        Compute skim matrices:
        - D : distance between zones
        - T0 : free flow travel time between zones
        - TC : traffic travel time between zones
        """
        kw = {"diagonal": diagonal, "density": density}
        self._compute_skim_basic("length", **kw)
        self._compute_skim_basic("t0", **kw)
        self._compute_skim_basic("tcur", **kw)

    def _compute_skim_basic(self, kind, diagonal="density", density=1000.0):
        """
        General method to compute skim matrices from basic quantities
        (free flow time, current time or distance).
        
        Inputs
        ======
        - kind : eg t0, tcur, l
        - diagonal : way to compute the matrix diagonal
        - density : average density per zone
        """
        assert kind in BASIC_SKIM_KINDS, (
            "Choose kind among %s." % BASIC_SKIM_KINDS
        )

        paths = nx.all_pairs_dijkstra_path_length(self.G, weight=kind)

        self.skims[kind] = pd.DataFrame(dict(paths)).loc[
            self.df_zones.index, self.df_zones.index
        ]

        # compute diagonal based on distance
        if diagonal == "density":
            np.fill_diagonal(
                self.skims[kind].values,
                np.sqrt(self.df_zones["pop"].values / density) * 0.5,
            )
        else:
            raise NotImplementedError("Only 'density'-based diagonal available.")

        # adjust time-related skims
        if kind in ["t0", "tcur"]:
            np.fill_diagonal(
                self.skims[kind].values,
                self.skims[kind].values.diagonal() / self.v_intra * 60.0,
            )

    def compute_skim_utility(self, name, params):
        """Compute the utility skim matrix composed of several
        basic link attributes (distance or times) and their unit
        values and specific link attributes"""
        # FILL
        pass

    # =====
    # Trip distribution
    # =====
    def dist_func(self, func, C, beta):
        if func == "exp":
            return np.exp(beta * C)
        elif func == "poly":
            return C ** (beta)
        else:
            raise ValueError("Function should be exp or poly.")

    def distribute(self, ds, C, func, param, n_iter=10):
        """
        Compute OD matrices for a given demand stratum
        via a doubly constrained iterative algorithm
        
        Inputs
        ======
        - ds : demand stratum
        - C : cost function as skim matrix, t0, tcur, l or utility
        - func : distribution function, exp or poly
        - param : parameter of the distribution function
        - Nit : number of iterations
        """
        assert ds in self.dstrat.index, "%s not found in demand strata" % ds
        assert C in self.skims.keys(), "cost %s not found among skim matrices" % C
        assert func in ["exp", "poly"], "choose exp or poly function"
        assert param <= 0.0, "parameter should be <= 0"
        assert n_iter > 0, "number of iterations should be positive"

        # define set of distribution parameters
        self.dpar.loc[ds] = [C, func, param]

        O = self.df_zones[self.dstrat.loc[ds, "prod"]].values.copy().astype(float)
        D = (
            self.df_zones[self.dstrat.loc[ds, "attr"]].values.copy()
            * self.dstrat.loc[ds, "param"]
        )
        # O *= D.sum() / O.sum() # normalisation wrt attraction
        O = np.multiply(O, D.sum() / O.sum())

        a, b = np.ones_like(O), np.ones_like(D)
        T = np.zeros((self.Nz, self.Nz))
        T = np.outer(O, D) * self.dist_func(func, self.skims[C].values, param)

        for i in range(n_iter):
            a = O / T.sum(1)
            T = T * np.outer(a, b)
            b = D / T.sum(0)

        # compute final mean average errors
        self.dist_errs = {
            "O": (abs(T.sum(1) - O)).sum() / len(T) / self.Nz,
            "D": (abs(T.sum(0) - D)).sum() / len(T) / self.Nz,
        }

        self.dmats[ds] = pd.DataFrame(
            T, columns=self.df_zones.index, index=self.df_zones.index
        )

    # =====
    # Assignment
    # =====
    def assign(self, imp, kind="incremental", weights=[50, 50]):
        """
        Perform assignment of traffic to the network.
        Use only one transport system here.
        
        1. Sum all demand matrices.
        2. For each weight, calculate all the shortest paths 
           by given impedance (distance or current time) 
           between two zones and add to the links on the path. 
           Update current time.
        
        Inputs
        ======
        - imp : impedance (link attribute) for path search 
            that is computed as a skim matrix
        - kind : type of assignment
        - ws : assignment weights
        """
        assert kind in ASSIGNMENT_KINDS, (
            "choose assignment kind from %s" % ASSIGNMENT_KINDS
        )
        assert imp in self.skims.keys(), f"impedance '{imp}' not defined."

        weights = np.array(weights)
        weights = weights / weights.sum()  # normalise weights

        # sum all demand matrices into one
        self.DM = sum(self.dmats.values())

        # remove flows and reset tcur/vcur before assignment
        nx.set_edge_attributes(self.G, 0.0, "q")
        self.df_links["q"] = 0.0

        if kind == "incremental":
            for wi, w in enumerate(weights):
                if self.verbose:
                    print("Assigning batch %i, weight %.2f ..." % (wi + 1, w))

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

        self._geh()
        self._var_geh()

    def _geh(self):
        """Compute the GEH error of each link with a measurement"""
        self.df_links["geh"] = np.sqrt(
            2.0
            * (self.df_links["q"] - self.df_links["count"]) ** 2
            / (self.df_links["q"] + self.df_links["count"])
            / 10.0
        )

    def _var_geh(self):
        """Compute GEH statistic variance, without the square root"""
        self.df_links["var_geh"] = (
            2.0
            * (self.df_links["q"] - self.df_links["count"]) ** 2
            / (self.df_links["q"] + self.df_links["count"])
            / 10.0
        )
