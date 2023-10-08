"""
A list of utility functions for reading and formatting input files.
Not released yet.

Collected: 2023-08-17
"""
import numpy as np
import pandas as pd
import geopandas as gpd


def read_inputs_excel(fname, offset=1_000_000, verbose=False):
    """
    Read inputs from excel file with raw sheets with PTV Visum tables with renamed columns:
    - zones
    - connectors
    - links
    - link_types
    - nodes

    Format these tables into a suitable input for the traffic-flow object.
    Change node and connector ID's (counters) by an offset to keep them distinct from zones and links.

    Input
    -----
    - fname : str, input excel file
    - offset : int, optional, number for offsetting the counter for nodes and connectors
    - verbose : bool
    
    Returns
    -------
    - nodes : pd.DataFrame
    - link_types : pd.DataFrame
    - links : pd.DataFrame
    """
    xls = pd.ExcelFile(fname)
    df0_zones = xls.parse("zones")
    df0_conn = xls.parse("connectors")
    df0_links = xls.parse("links")
    df0_link_types = xls.parse("link_types")
    df0_nodes = xls.parse("nodes")

    # merging zones and nodes
    if verbose:
        print('Preparing zones and nodes...')
    df0_zones["is_zone"] = True
    # df0_zones["id"] += offset  # FUTURE

    df0_nodes["is_zone"] = False
    df0_nodes["id"] += offset  # CHANGE
    # df0_nodes = df0_nodes.drop(["x_coord", "y_coord", "type"], 1) # CLEAN
    df_nodes = pd.concat([df0_zones, df0_nodes], sort=True)
    df_nodes = df_nodes.set_index("id").reset_index()

    # merging links and connectors
    if verbose:
        print('Preparing links...')
    df0_links["node_from"] += offset
    df0_links["node_to"] += offset

    if verbose:
        print('Preparing connectors...')
    df0_conn["node"] += offset  # CHANGE
    # df0_conn['zone'] += offset  # FUTURE

    df_O = df0_conn[df0_conn["direction"] == "O"].copy()
    df_D = df0_conn[df0_conn["direction"] == "D"].copy()
    
    if len(df_O) == 0:
        print(f'Warning: zero "O" connectors, filling from "D"')
        df_O = df_D.copy()
    if len(df_D) == 0:
        print(f'Warning: zero "D" connectors, filling from "O"')
        df_D = df_O.copy()
        
    df_O["node_from"] = df_O["node"]
    df_O["node_to"] = df_O["zone"]
    df_O["id"] = np.arange(offset, offset + len(df_O))

    df_D["node_from"] = df_D["zone"]
    df_D["node_to"] = df_D["node"]
    df_D["id"] = np.arange(offset, offset + len(df_D))

    df_conn = pd.concat([df_O, df_D], sort=False).drop(["node", "zone", "direction"], 1)
    df_conn.sort_values(by="id", inplace=True)
    df_conn = df_conn.reindex(columns=df0_links.columns)

    df_links = pd.concat([df0_links, df_conn], sort=False)
    df_links.sort_values(by="id", inplace=True)
    df_links = df_links.set_index("id").reset_index()
    df_links["name"] = ""

    return df_nodes, df0_link_types, df_links


def read_inputs_shapefile(basepath, basename, offset=1_000_000, verbose=False):
    """Read shapefiles extracted from PTV Visum and format them into input dataframes"""
    # define inputs
    path_shp_node = f'{basepath}/{basename}_node.SHP'
    path_shp_zone = f'{basepath}/{basename}_zone_centroid.SHP'
    path_shp_link = f'{basepath}/{basename}_link.SHP'
    path_shp_connector = f'{basepath}/{basename}_connector.SHP'

    # read all files
    if verbose:
        print('Reading shapefiles...')
    df_node = gpd.read_file(path_shp_node)
    df_zone = gpd.read_file(path_shp_zone)
    df_link = gpd.read_file(path_shp_link)
    df_connector = gpd.read_file(path_shp_connector)
    
    # wrangle nodes
    if verbose:
        print('Preparing nodes...')
    df_node['is_zone'] = False
    df_node = df_node.to_crs(4326)
    df_node = df_node.rename(columns={'NO': 'id'})
    df_node['id'] += offset
    df_node['is_zone'] = False
    
    # wrangle zones
    if verbose:
        print('Preparing zones...')
    df_zone['is_zone'] = True
    df_zone = df_zone.to_crs(4326)
    df_zone = df_zone.rename(columns={'NO': 'id', 'CODE': 'code', 'NAME': 'name', 'OBYV': 'pop'})
    df_zone['is_zone'] = True
    
    # wrangle links
    if verbose:
        print('Preparing links...')
    df_link['TYPENO'] = df_link['TYPENO'].astype(int)
    df_link['LENGTH'] = df_link['LENGTH'].map(lambda x: float(x.rstrip('km')))
    df_link['V0PRT'] = df_link['V0PRT'].map(lambda x: float(x.rstrip('km/h')))

    df_link = df_link.rename(columns={
        'NO': 'id',
        'FROMNODENO': 'node_from',
        'TONODENO': 'node_to',
        'LENGTH': 'length',
        'TYPENO': 'type',
    })
    df_link['node_from'] += offset
    df_link['node_to'] += offset

    # wrangle connectors
    if verbose:
        print('Preparing connectors...')
    df_connector['NODENO'] += offset

    df_O = df_connector.query('DIRECTION == "O"').copy()
    df_D = df_connector.query('DIRECTION == "D"').copy()

    if len(df_O) == 0:
        print(f'Warning: zero "O" connectors, filling from "D"')
        df_O = df_D.copy()

    if len(df_D) == 0:
        print(f'Warning: zero "D" connectors, filling from "O"')
        df_D = df_O.copy()
    
    else:    
        df_O["node_from"] = df_O["NODENO"]
        df_O["node_to"] = df_O["ZONENO"]
        df_O["id"] = np.arange(offset, offset + len(df_O))
        
        df_D["node_from"] = df_D["ZONENO"]
        df_D["node_to"] = df_D["NODENO"]
        df_D["id"] = np.arange(offset, offset + len(df_D))
        
    df_connector = pd.concat([df_O, df_D], sort=False).drop(["NODENO", "ZONENO", "DIRECTION"], 1)

    df_connector['LENGTH'] = df_connector['LENGTH'].map(lambda x: float(x.rstrip('km')))
    df_connector = df_connector.rename(columns={
        'LENGTH': 'length',
        'TYPENO': 'type',
    })

    # extract link types
    if verbose:
        print('Extracting link types...')
    df_lt = df_link[['type', 'CAPPRT', 'V0PRT', 'NUMLANES']].drop_duplicates()
    df_lt['type'] = df_lt['type'].astype(int)
    df_lt = df_lt.sort_values(by='type')
    df_lt.columns = ['type', 'qmax', 'v0', 'num_lanes']
    df_lt['a'] = 0.15
    df_lt['b'] = 4.0
    df_lt['type_name'] = 'undefined'

    # add connector link type
    df_lt = df_lt.append({
        'type': 0,
        'qmax': 20000,
        'v0': 40.0,
        'num_lanes': 4,
        'a': 0.15,
        'b': 4.0,
        'type_name': 'conn',
}, ignore_index=True)
    
    # merge into trafficflow structure
    if verbose:
        print('Merging dataframes...')
    df_all_links = pd.concat([df_link, df_connector], sort=False, ignore_index=True)
    df_all_nodes = pd.concat([df_zone, df_node], sort=False, ignore_index=True)
    
    return df_all_nodes, df_lt, df_all_links

