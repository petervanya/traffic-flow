#!/usr/bin/env python3
"""
Unit tests for individual `MTM` building blocks (input validation, the BPR
volume-delay function, the GEH error metric, distribution kernels), as
opposed to the full pipeline integration tests in `test_pipelines*.py`.
"""
import numpy as np
import pandas as pd
import pytest

from traffic_flow import MTM


# ---------------------------------------------------------------------------
# Construction / backend validation
# ---------------------------------------------------------------------------

def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        MTM(backend="not-a-backend")


def test_backend_is_case_insensitive():
    assert MTM(backend="IGRAPH").backend == "igraph"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_read_data_missing_node_column_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    with pytest.raises(AssertionError):
        MTM().read_data(df_nodes.drop(columns=["is_zone"]), df_link_types, df_links)


def test_read_data_missing_link_type_column_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    with pytest.raises(AssertionError):
        MTM().read_data(df_nodes, df_link_types.drop(columns=["v0"]), df_links)


def test_read_data_missing_link_column_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    with pytest.raises(AssertionError):
        MTM().read_data(df_nodes, df_link_types, df_links.drop(columns=["length"]))


def test_read_data_unknown_link_type_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    df_links = df_links.copy()
    df_links.loc[0, "type"] = 999  # not present in df_link_types
    with pytest.raises(ValueError, match="Missing v0 values"):
        MTM().read_data(df_nodes, df_link_types, df_links)


def test_generate_requires_read_data():
    with pytest.raises(AssertionError):
        MTM().generate("s", "pop", "pop", 0.5)


def test_generate_unknown_attribute_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    with pytest.raises(AssertionError):
        model.generate("s", "not_a_column", "pop", 0.5)


# ---------------------------------------------------------------------------
# Trip distribution validation
# ---------------------------------------------------------------------------

@pytest.fixture
def model_with_skims(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("s", "pop", "pop", 0.5)
    model.compute_skims()
    return model


def test_distribute_unknown_stratum_raises(model_with_skims):
    with pytest.raises(AssertionError):
        model_with_skims.distribute("does-not-exist", "tcur", "exp", -0.02)


def test_distribute_unknown_skim_raises(model_with_skims):
    with pytest.raises(AssertionError):
        model_with_skims.distribute("s", "not-a-skim", "exp", -0.02)


def test_distribute_unknown_function_raises(model_with_skims):
    with pytest.raises(AssertionError):
        model_with_skims.distribute("s", "tcur", "not-a-function", -0.02)


def test_distribute_non_positive_n_iter_raises(model_with_skims):
    with pytest.raises(AssertionError):
        model_with_skims.distribute("s", "tcur", "exp", -0.02, n_iter=0)


# ---------------------------------------------------------------------------
# Assignment validation
# ---------------------------------------------------------------------------

def test_assign_unknown_impedance_raises(model_with_skims):
    model_with_skims.distribute("s", "tcur", "exp", -0.02)
    with pytest.raises(ValueError):
        model_with_skims.assign("not-an-impedance")


def test_assign_unknown_kind_raises(model_with_skims):
    model_with_skims.distribute("s", "tcur", "exp", -0.02)
    with pytest.raises(ValueError):
        model_with_skims.assign("tcur", kind="not-a-kind")


# ---------------------------------------------------------------------------
# dist_func: distribution / deterrence kernels
# ---------------------------------------------------------------------------

def test_dist_func_exp():
    m = MTM()
    C = np.array([0.0, 10.0])
    beta = -0.1
    np.testing.assert_allclose(m.dist_func("exp", C, beta), np.exp(beta * C))


def test_dist_func_poly():
    m = MTM()
    C = np.array([1.0, 4.0])
    beta = -0.5
    np.testing.assert_allclose(m.dist_func("poly", C, beta), C**beta)


def test_dist_func_power():
    m = MTM()
    C = np.array([1.0, 4.0])
    beta = [1.0, 0.5]
    np.testing.assert_allclose(m.dist_func("power", C, beta), (C + beta[1]) ** beta[0])


# ---------------------------------------------------------------------------
# BPR volume-delay function
# ---------------------------------------------------------------------------

def test_compute_tcur_links_bpr_formula(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)

    # zero flow: current time equals free-flow time
    model.df_links["q"] = 0.0
    model.compute_tcur_links()
    np.testing.assert_allclose(model.df_links["tcur"], model.df_links["t0"])

    # flow at capacity: tcur = t0 * (1 + a)
    model.df_links["q"] = model.df_links["qmax"]
    model.compute_tcur_links()
    expected = model.df_links["t0"] * (1.0 + model.df_links["a"])
    np.testing.assert_allclose(model.df_links["tcur"], expected)


# ---------------------------------------------------------------------------
# GEH error metric
# ---------------------------------------------------------------------------

def test_geh_zero_when_flow_matches_count(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.df_links["q"] = model.df_links["count"]
    model.compute_error()
    measured = model.df_links["count"].notna()
    np.testing.assert_allclose(model.df_links.loc[measured, "geh"], 0.0, atol=1e-9)


def test_geh_known_value(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.df_links["q"] = model.df_links["count"] + 10.0
    model.compute_error()
    measured = model.df_links["count"].notna()
    q = model.df_links.loc[measured, "count"] + 10.0
    count = model.df_links.loc[measured, "count"]
    expected = np.sqrt(2.0 * (q - count) ** 2 / (q + count) / 10.0)
    np.testing.assert_allclose(model.df_links.loc[measured, "geh"], expected)


def test_compute_error_missing_column_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    with pytest.raises(ValueError):
        model.compute_error(measured_col="not-a-column")


# ---------------------------------------------------------------------------
# compute_percentile
# ---------------------------------------------------------------------------

def test_compute_percentile_matches_manual_count(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    # flow deviates from count by an amount that yields a known GEH
    model.df_links["q"] = model.df_links["count"] + 1.0

    pct = model.compute_percentile(1000.0)  # extremely lax threshold
    assert pct == 1.0

    pct_strict = model.compute_percentile(1e-9)  # essentially zero tolerance
    assert pct_strict == 0.0


def test_compute_percentile_rejects_non_positive_threshold(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    with pytest.raises(ValueError):
        model.compute_percentile(0)


# ---------------------------------------------------------------------------
# compute_mean_trip_length
# ---------------------------------------------------------------------------

def test_compute_mean_trip_length_within_skim_bounds(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("s", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("s", "tcur", "exp", -0.02)

    mean_length = model.compute_mean_trip_length("s")
    lengths = model.skims["length"].values
    assert lengths.min() <= mean_length <= lengths.max()


def test_compute_mean_trip_length_unknown_stratum_raises(network_1_data):
    df_nodes, df_link_types, df_links = network_1_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    with pytest.raises(AssertionError):
        model.compute_mean_trip_length("does-not-exist")
