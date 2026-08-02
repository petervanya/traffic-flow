#!/usr/bin/env python3
"""
End-to-end pipeline tests (generate -> compute_skims -> distribute -> assign)
for the undirected model variants, across both graph backends and both
sample networks.
"""
import pytest

from traffic_flow import MTMUndirected, MTMnxUndirected

MODEL_CLASSES = [MTMUndirected, MTMnxUndirected]
NETWORK_FIXTURES = ["network_1_undirected_data", "network_2_undirected_data"]


@pytest.mark.parametrize("network_fixture", NETWORK_FIXTURES)
@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_full_pipeline_undirected(request, model_cls, network_fixture):
    df_nodes, df_link_types, df_links = request.getfixturevalue(network_fixture)
    mobility = 0.5

    model = model_cls()
    model.read_data(df_nodes, df_link_types, df_links)

    total_pop = model.df_zones["pop"].sum()
    model.generate("main-stratum", "pop", "pop", mobility)
    model.compute_skims()
    model.distribute("main-stratum", "tcur", "exp", -0.02)

    total_demand = model.dmats["main-stratum"].values.sum()
    assert total_demand == pytest.approx(total_pop * mobility, rel=1e-6)

    for kind in ("length", "t0", "tcur"):
        assert model.skims[kind].shape == (model.Nz, model.Nz)

    model.assign("tcur")

    q = model.df_links["q"]
    assert q.notna().all()
    assert (q >= 0).all()
    assert q.sum() > 0
    assert (model.df_links["tcur"] >= model.df_links["t0"] - 1e-9).all()
