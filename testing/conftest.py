"""
Shared pytest fixtures for the traffic_flow test suite.
"""
import sys
sys.path.append(".")

import pytest

from traffic_flow.sample_networks import (
    load_network_1,
    load_network_1_undirected,
    load_network_2,
    load_network_2_undirected,
)


@pytest.fixture(scope="session")
def network_1_data():
    """4 zones, 8 directed links, with measured `count` on 3 links."""
    return load_network_1()


@pytest.fixture(scope="session")
def network_2_data():
    """12 zones, 42 directed links (main road corridors of Slovakia)."""
    return load_network_2()


@pytest.fixture(scope="session")
def network_1_undirected_data():
    return load_network_1_undirected()


@pytest.fixture(scope="session")
def network_2_undirected_data():
    return load_network_2_undirected()
