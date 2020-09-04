import os
import pandas as pd


def load_network_1():
    """Load a small network of 4 zones and 8 undirected links"""
    dirn = os.path.dirname(__file__) + "/examples/"

    xls = pd.ExcelFile(dirn + "network_1.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")

    return df_nodes, df_types, df_links


def load_network_2():
    """Load a large network of main corridors of Slovakia"""
    dirn = os.path.dirname(__file__) + "/examples/"

    xls = pd.ExcelFile(dirn + "network_2.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")

    return df_nodes, df_types, df_links


def load_network_2_directed():
    """Load a large directed network of main corridors of Slovakia"""
    dirn = os.path.dirname(__file__) + "/examples/"

    xls = pd.ExcelFile(dirn + "network_2_directed.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")

    return df_nodes, df_types, df_links

