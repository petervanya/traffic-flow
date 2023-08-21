"""
A list of utility functions for reading and formatting input files.
Not released yet.

Collected: 2023-08-17
"""
import numpy as np
import pandas as pd
import geopandas as gpd


def prepare_inputs_excel(fname, offset=1_000_000):
    """
    Read excel consisting from the raw sheets copied from PTV Visum tables 
    with already renamed columns:
    - zones
    - connectors
    - links
    - link_types
    - nodes

    Format these tables into a suitable input for the traffic-flow object.
    Offset node and connector id's to distinguish from zones and links.
    """
    xls = pd.ExcelFile(fname)
    df0_zones = xls.parse("zones")
    df0_conn = xls.parse("connectors")
    df0_links = xls.parse("links")
    df0_link_types = xls.parse("link_types")
    df0_nodes = xls.parse("nodes")

    # merging zones and nodes
    df0_zones["is_zone"] = True
    # df0_zones["id"] += offset  # FUTURE

    df0_nodes["is_zone"] = False
    df0_nodes["id"] += offset  # CHANGE
    df0_nodes = df0_nodes.drop(["x_coord", "y_coord", "type"], 1)
    df_nodes = pd.concat([df0_zones, df0_nodes], sort=True)
    df_nodes = df_nodes.set_index("id").reset_index()

    # merging links and connectors
    df0_links["node_from"] += offset
    df0_links["node_to"] += offset

    df0_conn["node"] += offset  # CHANGE
    # df0_conn['zone'] += offset  # FUTURE
    df_O = df0_conn[df0_conn["direction"] == "O"].copy()
    df_O["node_from"] = df_O["node"]
    df_O["node_to"] = df_O["zone"]
    df_O["id"] = np.arange(offset, offset + len(df_O))

    df_D = df0_conn[df0_conn["direction"] == "D"].copy()
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


def prepare_inputs_shapefile(basepath, basename):
    """Read shapefiles extracted from PTV Visum and format them into input dataframes"""
    pass
