#!/usr/bin/env python
"""
Test igraph backend with directed sample network 2, including error
computation against measured counts.

Verifies that production-balancing and attraction-balancing yield the same
result when the mobility parameter is scaled accordingly (as documented in
`MTM.distribute`), and that the GEH error is well-defined on measured links.

2020-10-28
"""
import pytest

from traffic_flow import MTM
from traffic_flow.sample_networks import load_network_2


def _run(df_n, df_lt, df_l, mobility, balancing):
    mtm = MTM()
    mtm.read_data(df_n, df_lt, df_l)
    mtm.generate("all", "pop", "pop2", mobility)
    mtm.compute_skims()
    mtm.distribute("all", "t0", "exp", -0.1, balancing=balancing)
    mtm.assign("t0")
    mtm.compute_error()
    return mtm


@pytest.fixture
def network_2_with_pop2(network_2_data):
    df_n, df_lt, df_l = network_2_data
    df_n = df_n.copy()
    df_n["pop2"] = df_n["pop"] * 2
    return df_n, df_lt, df_l


def test_production_vs_attraction_balancing_equivalent(network_2_with_pop2):
    df_n, df_lt, df_l = network_2_with_pop2

    m_prod = _run(df_n, df_lt, df_l, mobility=2, balancing="production")
    m_attr = _run(df_n, df_lt, df_l, mobility=0.22, balancing="attraction")

    measured = m_prod.df_links["count"].notna()
    assert m_prod.df_links.loc[measured, "geh"].mean() == pytest.approx(
        m_attr.df_links.loc[measured, "geh"].mean(), rel=1e-6
    )
    assert m_prod.df_links["q"].mean() == pytest.approx(
        m_attr.df_links["q"].mean(), rel=1e-6
    )


def test_geh_finite_on_measured_links(network_2_with_pop2):
    df_n, df_lt, df_l = network_2_with_pop2
    model = _run(df_n, df_lt, df_l, mobility=2, balancing="production")

    measured = model.df_links["count"].notna()
    assert measured.any()
    geh_measured = model.df_links.loc[measured, "geh"]
    assert geh_measured.notna().all()
    assert (geh_measured >= 0).all()

    assert model.df_links["q"].notna().all()
    assert (model.df_links["q"] >= 0).all()
