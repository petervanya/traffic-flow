import os
import pandas as pd


DIRN = os.path.dirname(__file__) + "/examples/"


def load_network_1_undirected():
    """Load a small undirected network of 4 zones and 8 undirected links"""
    xls = pd.ExcelFile(DIRN + "network_1_undirected.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")
    return df_nodes, df_types, df_links


def load_network_1():
    """Load a small directed network of 4 zones and 8 directed links"""
    xls = pd.ExcelFile(DIRN + "network_1.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")
    return df_nodes, df_types, df_links


def load_network_2_undirected():
    """Load a large undirected network of main corridors of Slovakia"""
    xls = pd.ExcelFile(DIRN + "network_2_undirected.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")
    return df_nodes, df_types, df_links


def load_network_2():
    """Load a large directed network of main corridors of Slovakia"""
    xls = pd.ExcelFile(DIRN + "network_2.xlsx")
    df_nodes = xls.parse("nodes")
    df_types = xls.parse("link_types")
    df_links = xls.parse("links")
    return df_nodes, df_types, df_links
