#!/usr/bin/env python
"""
Testing of the optimisation procedure using the igraph backend.

Created: 2020-09-24
Update: 2023-08-15
"""
import numpy as np
import pytest

from traffic_flow import MTM
from traffic_flow.sample_networks import load_network_2


@pytest.fixture
def base_model(network_2_data):
    """A model with generation, skims and distribution already run,
    ready to be passed into `optimise()`."""
    df_nodes, df_link_types, df_links = network_2_data
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    return model


@pytest.mark.parametrize(
    "method, x0",
    [
        ("nelder-mead", [0.07, -1e-3]),
        ("dual-annealing", [0.07, -1e-3]),
    ],
)
def test_optimise_reduces_error(base_model, method, x0):
    base_model.compute_error()
    initial_error = base_model.df_links["geh"].mean()

    res = base_model.optimise(method=method, n_iter=5, x0=x0, record=True)

    assert np.isfinite(res.train_error)
    assert res.train_error < initial_error
    assert base_model.opt_params.shape == (1, 2)
    assert list(base_model.opt_params.columns) == ["attr_param", "dist_param"]


def test_optimise_grid_search(base_model):
    res = base_model.optimise(
        method="grid-search", grids=[[0.05, 0.075], [-0.01, -0.02]]
    )

    assert len(res) == 4  # full cartesian product of the two 2-value grids
    assert "objective" in res.columns
    assert np.isfinite(res["objective"]).all()
    assert (res["objective"] >= 0).all()
    assert base_model.opt_params.shape == (1, 2)


def test_optimise_train_test_split(base_model):
    res = base_model.optimise(
        method="nelder-mead",
        n_iter=5,
        x0=[0.07, -1e-3],
        train_test_split=0.6,
        seed=42,
    )

    assert np.isfinite(res.train_error)
    assert np.isfinite(res.test_error)
    assert "train_mask" in base_model.df_links.columns

    measured = base_model.df_links["count"].notna()
    n_measured_ids = base_model.df_links.loc[measured, "id"].nunique()
    n_train_ids = base_model.df_links.loc[
        base_model.df_links["train_mask"], "id"
    ].nunique()
    assert n_train_ids == int(np.ceil(0.6 * n_measured_ids))


def test_optimise_requires_generation_step():
    df_nodes, df_link_types, df_links = load_network_2()
    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    with pytest.raises(AssertionError):
        model.optimise(method="nelder-mead", x0=[0.07, -1e-3])


def test_optimise_invalid_method(base_model):
    with pytest.raises(ValueError):
        base_model.optimise(method="not-a-method", x0=[0.07, -1e-3])


def test_optimise_grid_search_requires_grids(base_model):
    with pytest.raises(ValueError):
        base_model.optimise(method="grid-search")
