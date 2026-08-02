#!/usr/bin/env python3
"""
End-to-end pipeline tests (generate -> compute_skims -> distribute -> assign)
for the directed `MTM` class, across both graph backends and both sample
networks.
"""
import pytest

from traffic_flow import MTM

BACKENDS = ["igraph", "networkx"]
NETWORK_FIXTURES = ["network_1_data", "network_2_data"]


@pytest.mark.parametrize("network_fixture", NETWORK_FIXTURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_full_pipeline(request, backend, network_fixture):
    df_nodes, df_link_types, df_links = request.getfixturevalue(network_fixture)
    mobility = 0.5

    model = MTM(backend=backend)
    assert model.backend == backend
    model.read_data(df_nodes, df_link_types, df_links)

    total_pop = model.df_zones["pop"].sum()
    model.generate("main-stratum", "pop", "pop", mobility)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)

    # doubly-constrained gravity model must conserve total generated demand
    total_demand = model.dmats["main-stratum"].values.sum()
    assert total_demand == pytest.approx(total_pop * mobility, rel=1e-6)

    # skim matrices are square, one row/col per zone
    for kind in ("length", "t0", "tcur"):
        assert model.skims[kind].shape == (model.Nz, model.Nz)

    model.assign("tcur")

    q = model.df_links["q"]
    assert q.notna().all()
    assert (q >= 0).all()
    assert q.sum() > 0

    # BPR volume-delay function never speeds traffic up below free flow
    assert (model.df_links["tcur"] >= model.df_links["t0"] - 1e-9).all()
