"""
Author: Katarina Simkova
"""
import numpy as np
from numpy import sqrt, exp
import pandas as pd
import time
import igraph as ig


class DiMTMig:
    """Macroscopic transport modelling class"""
    # global varialbles
    assignment_kinds = ["incremental"]
    basic_skim_kinds = ["t0", "tcur", "l"]
    
    def __init__(self, v_intra=40.0, verbose=False):
        """
        Inputs
        ======
        - v_intra : intrazonal speed in kmh
        - verbose : level of printing information
        """
        self.G = ig.Graph(directed=True)
        self.skims = {} # skim matrices
        self.dmats = {} # demand matrices
        # demand strata
        self.dstrat = pd.DataFrame(columns=["prod", "attr", "param"])
        self.dpar = pd.DataFrame(\
            columns=["skim", "func", "param", "symmetric"]) # distribution params
        # optimisation results
        self.optres = pd.DataFrame(columns=["attr_param", "dist_param"])
        self.optout = pd.DataFrame(columns=["sum_geh", "nit", "nfev", "success"])
        # ad-hoc data
        self.v_intra = v_intra
        self.verbose = verbose

        
    def read_data(self, nodes, link_types, links):
        """
        Inputs
        ======
        - Nodes : dataframe [is_zone, name, coords, pop], index: id
        - Link types : dataframe [name, v0, qmax, a, b], index: id
        - Links : dataframe [id, type, name, l, count], index: id_node_pair
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
        """Check that key columns are present"""
        assert np.isin(["id", "is_zone", "name", "coords", "pop"],\
                       self.df_nodes.columns).all(),\
            "Node list does not have the expected structure."
        
    
    def _verify_lt(self):
        """Verify columns"""
        assert np.isin(["id", "v0", "qmax", "a", "b"],\
                       self.df_lt.columns).all(),\
            "Link type list does not have the expected structure."
            
    
    def _verify_links(self):
        """Verify if link type numbers subset of link types
        and if they connect the nodes"""
        assert np.isin(["id", "type", "name", "l", "count", "node_from", "node_to"],\
                       self.df_links.columns).all(),\
            "Link list does not have the expected structure."
    
    
    def _assign_link_data(self):
        """
        Assign the attributes of the link types to the links:
        [v0, qmax, alpha, beta]
        Create new empty attributes:
        [t0, q, tcur, vcur]
        """
        # merge with link types
        # RENAME ID_LT to ID first
        self.df_links = self.df_links.merge(\
                self.df_lt.drop(["name"], 1), how="left", \
                left_on="type", right_on="id", suffixes=("", "_dum"))\
            .drop("id_dum", 1)
        
        self.df_links = self.df_links.set_index(["node_from", "node_to"])
        # sort node order
        
        # assign empty attributes
        self.df_links["t0"] = self.df_links["l"] / self.df_links["v0"] * 60.0
        self.df_links["q"] = 0.0
        self.df_links["tcur"] = self.df_links["t0"]
        self.df_links["vcur"] = self.df_links["v0"]
        
        # check if any values are missing
        assert self.df_links["type"].isna().any() == False, \
            "Missing link types."
        
    
    def compute_tcur_links(self):
        """Compute current travel time and speed wrt the flows
        and assign to the graph G"""
        self.df_links["tcur"] = self.df_links["t0"] \
                                *(1.0 + self.df_links["a"]\
                                * (self.df_links["q"]/self.df_links["qmax"])\
                                **self.df_links["b"])
        
        self.df_links["vcur"] = self.df_links["l"] / self.df_links["tcur"] * 60.0
       
        # assign to the graph
        self.G.es["tcur"] = self.df_links["tcur"].values
        self.G.es["vcur"] = self.df_links["vcur"].values

    def _fill_graph(self):
        """Fill the graph with read nodes and links"""
    #############################################################
    # K: making sure the graph is empty (needed when self.read_data is run
    # multiple times in the code, but MTM does not initialise at every run)
        if len(self.G.vs)>0:
            self.G = ig.Graph(directed=True)
        
    #adding vertices
        self.G.add_vertices(self.df_nodes.shape[0])
        self.G.vs["id"] = self.df_nodes.index.values
        for k, v in self.df_nodes.iteritems():
            self.G.vs[k] = v.values

    #adding edges
        for k, _ in self.df_links.iterrows():
            self.G.add_edges([(self.G.vs.find(id=k[0]).index,\
                           self.G.vs.find(id=k[1]).index)])
        for k, v in self.df_links.iteritems():
            self.G.es[k] = v.values
    #############################################################        
            
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
        assert prod in self.df_nodes.columns, \
            "Production attribute not found in node columns."
        assert attr in self.df_nodes.columns, \
            "Attraction attribute not found in node columns."
        
        self.dstrat.loc[name] = [prod, attr, param]
    
    # =====
    # Skim matrices
    # =====
    def compute_skims(self, diagonal="density", density=1000.0):
        """
        Compute skim matrices, choose from:
        - "l" : distance between zones
        - "t0" : free flow travel time between zones
        - "tcur" : traffic travel time between zones
        """
        kw = {"diagonal": diagonal, "density": density}
        
        self._compute_skim_basic("l", **kw)
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
        assert kind in self.basic_skim_kinds, \
            "Choose kind among %s." % self.basic_skim_kinds
        
        ######################################################################
        paths = self.G.shortest_paths(source=self.G.vs.select(is_zone_eq=True)\
                     ,target=self.G.vs.select(is_zone_eq=True), weights=kind)
        
        self.skims[kind] = pd.DataFrame(paths)
        self.skims[kind].index = self.G.vs.select(is_zone_eq=True)["id"]
        self.skims[kind].columns = self.G.vs.select(is_zone_eq=True)["id"]
        ######################################################################
        
        # compute diagonal based on distance
        if diagonal == "density":
            np.fill_diagonal(self.skims[kind].values, \
            np.sqrt(self.df_zones["pop"].values / density) * 0.5)
        else:
            raise NotImplementedError("Only 'density'-based diagonal available.")
            
        # adjust time-related skims
        if kind in ["t0", "tcur"]:
            np.fill_diagonal(self.skims[kind].values, \
                self.skims[kind].values.diagonal() / self.v_intra * 60.0)
              
        
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
            return np.exp(-beta*C)
        elif func == "poly":
            return C**(-beta)
        else:
            raise ValueError("Function should be exp or poly.")
        
    # K: added symmetry option for e.g. work demand stratum
    def distribute(self, ds, C, func, param, Nit=10, sym=False):
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
        - sym : if True, makes the demand matrix symmetric; default is False
        """
        assert ds in self.dstrat.index,\
            "%s not found in demand strata." % ds
        assert C in self.skims.keys(),\
            "Cost %s not found among skim matrices" % C
        assert func in ["exp", "poly"], \
            "Choose exp or poly function."
        assert param >= 0.0, "Parameter should be >= 0."
        assert Nit > 0, "Number of iterations should be positive."
        assert np.logical_or((sym == False), (sym == True)),\
             "Choose boolean value for the symmetry of the matrix."
        
        # define set of distribution parameters
        self.dpar.loc[ds] = [C, func, param, sym]
        
        O = self.df_zones[self.dstrat.loc[ds, "prod"]].values.copy().astype(float)
        D = self.df_zones[self.dstrat.loc[ds, "attr"]].values.copy() * \
            self.dstrat.loc[ds, "param"]
        O *= D.sum() / O.sum() # normalisation wrt attraction
        
        a, b = np.ones_like(O), np.ones_like(D)
        T = np.zeros((self.Nz, self.Nz))
        T = np.outer(O, D) * \
            self.dist_func(func, self.skims[C].values, param)
        
        for i in range(Nit):
            a = O / T.sum(1)
            T = T * np.outer(a, b)
            b = D / T.sum(0)

        # compute final mean average errors
        self.dist_errs = {"O": (abs(T.sum(1) - O)).sum() / len(T) / self.Nz,\
                         "D": (abs(T.sum(0) - D)).sum() / len(T) / self.Nz}
        if sym:
            T = (T + T.T) / 2.0
        self.dmats[ds] = pd.DataFrame(T, columns=self.df_zones.index, \
                              index=self.df_zones.index)
    
    
    # =====
    # Assignment
    # =====
    def assign(self, imp, kind="incremental", ws=[50, 50]):
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
        assert kind in self.assignment_kinds, \
            "Assignment kind not available, choose from %s" % \
            self.assignment_kinds
        
#         K: Impedance in assignment is defined on graph's edges, not from
#         skim matrices. Changed "assert imp in self.skims.keys()" and its
#         error prompt to "assert imp in self.basic_skim_kinds".
#         Alternatively, it could be "assert imp in mtm.G.es.attributes()" 
#         but then it's risky that other attribute gets called by accident.
        
        assert imp in self.basic_skim_kinds, \
            "Choose impedance among %s." % self.basic_skim_kinds
            
        ws = np.array(ws)
        ws = ws / ws.sum() # normalise weights
        
        # sum all demand matrices into one
        self.DM = sum(self.dmats.values())

        # remove flows and reset tcur/vcur before assignment
        ####################################################
        # K: added resetting the tcur and vcur on graph to t0 and v0, since
        # this caused the inconsistency of 1 run of the module with the module
        # running in the loop if read_data was not in the loop
        self.G.es["q"] = 0.0
        self.G.es["tcur"] = self.df_links["t0"].values
        self.G.es["vcur"] = self.df_links["v0"].values
        self.df_links["q"] = 0.0
        
        if kind == "incremental":
            for wi, w in enumerate(ws):
                if self.verbose:
                    print("Assigning batch %i, weight %.2f ..." % (wi+1, w))
                vs_zones =[v.index for v in self.G.vs.select(is_zone_eq=True)]

                for i in vs_zones:
                    p = self.G.get_shortest_paths(v=i, to=vs_zones,\
                                            weights=imp, output="epath")
                    dq = self.DM.loc[self.G.vs[i]["id"], :].values * w
                        
                    for j, _ in enumerate(dq):
                        self.G.es[p[j]]["q"] += dq[j]
                
                self.df_links["q"] = self.G.es["q"]
                self.compute_tcur_links() # update current time and speed


    # =====
    # Optimisation
    # =====
    def optimise(self, Nit, K_low=1e-6, K_up=3.1, c_low=1e-6, c_up=0.11, \
        asimp="tcur", seed=1101, ws=[50, 50]):
        """
        Optimisation of model parameters.

        Input
        =====
        - Nit : number of iterations
        - K_low : lower bound on trip generation parameter
        - K_up : upper bound on trip generation parameter
        - c_low : lower bound on trip distribution parameter
        - c_up : upper bound on trip distribution parameter
        - asimp : assignment impedance (t0, tcur, l)
        - seed : random seed
        - ws : weights in incremental assignment
        """
        from scipy.optimize import dual_annealing
        # DEFINE INITIAL VALUES

        optargs = (asimp, ws,)
        bounds = [] # vector of bounds
        for m in self.dstrat.index:
            bounds = bounds + [(K_low, K_up), (c_low, c_up)]
        
        tic = time.time()
        res = dual_annealing(self._obj_function, args=optargs, bounds=bounds, \
            seed=seed, maxiter=Nit)
        toc = time.time()
        print("Optimisation terminated. Success: %s" % res.success)
        print("Resulting parameters: %s" % res.x)
        print("Time: %.2f s" % (toc - tic))
    
        for m, n in enumerate(self.dstrat.index):
            self.optres.loc[n] = [res.x[2*m], res.x[2*m+1]]
            
        self.optout.loc[1] = [res.fun, res.nit, res.nfev, res.success]
    

    def geh(self):
        """Compute the GEH of each link with a measurement"""
        pass

    
    def _obj_function(self, z, imp, ws=[50, 50]):
        """
        The sum of all GEH differences between traffic counts and modelled
        flows on links that contain the counts.
        Serves as the objective function for the optimisation.
        Goes through all transport modelling steps.

        Input
        =====
        - z : array of parameters to optimise
        - imp : impedance kind (t0, tcur, l)
        - ws : assignment weights
        """
        # CATCH ERROR IF DEMAND STRATA NOT DEFINED

        # trip generation
        for m, n in enumerate(self.dstrat.index):
            self.generate(n, self.dstrat["prod"][m], self.dstrat["attr"][m], z[2*m])

        # trip distribution
        for m, n in enumerate(self.dpar.index):
            self.distribute(n, self.dpar["skim"][m], self.dpar["func"][m], \
                z[2*m+1], sym=self.dpar["symmetric"][m])
        
        # assignment
        self.assign(imp=imp, ws=ws)
    
        relevant = self.df_links[np.logical_and(\
                (np.logical_not(np.isnan(self.df_links["count"]))),\
                (self.df_links["count"] != 0))]
    
        geh = sqrt(2*(relevant["q"].values - relevant["count"].values)**2/\
                (relevant["q"].values + relevant["count"].values)) \
                / sqrt(10.0)
    
#     reg = alpha*(\
#             self.dmats["all"].multiply(self.skims["t0"]).sum().sum()\
#                         /self.dmats["all"].sum().sum() - xexp)**2
#     vals = np.append(vals, suma)
    
        return np.sum(geh) #+ reg    





