#!/usr/bin/env python
"""
Test reading inputs, both the synthetic PTV-Visum-style fixture (always run)
and the real internal networks (only run when the machine-specific data is
present, e.g. on a maintainer's own checkout, never in CI since `Internal/`
is gitignored).

2023-09-23
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import traffic_flow as tfl
from traffic_flow import MTM
from traffic_flow.utils import read_inputs_excel, read_inputs_shapefile

REPO_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_EXCEL = REPO_ROOT / "Internal/Networks/network_raw_I51_201002.xlsx"
SHAPEFILE_BASEPATH = Path(
    "/Users/peter/Tatra/UHP/Res/Transport_Models/Data/Shapefiles_I51/nulovy_stav/"
)


@pytest.fixture
def synthetic_raw_visum_excel(tmp_path):
    """
    Build a minimal excel workbook in the raw PTV-Visum-export shape expected
    by `read_inputs_excel` (sheets: zones, nodes, links, connectors,
    link_types), so the import/wrangling logic in `utils.py` is exercised in
    CI without depending on any real (gitignored) network data.
    """
    zones = pd.DataFrame({"id": [1, 2], "pop": [1000.0, 2000.0], "name": ["Zone1", "Zone2"]})
    nodes = pd.DataFrame({"id": [1, 2], "name": ["Node1", "Node2"]})
    links = pd.DataFrame(
        {
            "id": [1, 2],
            "node_from": [1, 2],
            "node_to": [2, 1],
            "type": [1, 1],
            "length": [10.0, 10.0],
        }
    )
    connectors = pd.DataFrame(
        {
            "node": [1, 2, 1, 2],
            "zone": [1, 2, 1, 2],
            "direction": ["O", "O", "D", "D"],
            "type": [0, 0, 0, 0],
            "length": [1.0, 1.0, 1.0, 1.0],
        }
    )
    link_types = pd.DataFrame(
        {
            "type": [0, 1],
            "type_name": ["conn", "road"],
            "v0": [40.0, 90.0],
            "qmax": [20000.0, 30000.0],
            "a": [0.15, 0.15],
            "b": [4.0, 4.0],
        }
    )

    fname = tmp_path / "synthetic_raw_visum.xlsx"
    with pd.ExcelWriter(fname) as writer:
        zones.to_excel(writer, sheet_name="zones", index=False)
        nodes.to_excel(writer, sheet_name="nodes", index=False)
        links.to_excel(writer, sheet_name="links", index=False)
        connectors.to_excel(writer, sheet_name="connectors", index=False)
        link_types.to_excel(writer, sheet_name="link_types", index=False)
    return fname


def test_read_inputs_excel_synthetic(synthetic_raw_visum_excel):
    """Round-trips a minimal raw-Visum-style workbook through the full
    generate/skims/distribute/assign pipeline."""
    df_nodes, df_link_types, df_links = read_inputs_excel(
        synthetic_raw_visum_excel, offset=1000
    )

    # two zones, two crossroad nodes
    assert df_nodes["is_zone"].sum() == 2
    assert (~df_nodes["is_zone"]).sum() == 2
    # both directed road links plus 4 connector links (O+D per zone)
    assert len(df_links) == 6

    model = tfl.from_dataframes(df_nodes, df_link_types, df_links)
    model.generate("all", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("all", "tcur", "exp", -0.02)
    model.assign("tcur")

    assert model.df_links["q"].notna().all()
    assert (model.df_links["q"] >= 0).all()
    assert model.df_links["q"].sum() > 0


@pytest.fixture
def synthetic_visum_shapefiles(tmp_path):
    """
    Build a minimal set of shapefiles in the raw PTV-Visum-export shape
    expected by `read_inputs_shapefile` (`*_node`, `*_zone_centroid`,
    `*_link`, `*_connector`), so the geometry/unit-parsing/renaming logic in
    `utils.py` is exercised in CI without any real GIS export.
    """
    basename = "synthetic_net"
    basepath = tmp_path

    nodes = gpd.GeoDataFrame(
        {"NO": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:3857"
    )
    nodes.to_file(basepath / f"{basename}_node.SHP")

    zones = gpd.GeoDataFrame(
        {
            "NO": [1, 2],
            "CODE": ["Z1", "Z2"],
            "NAME": ["Zone1", "Zone2"],
            "OBYV": [1000, 2000],
        },
        geometry=[Point(-1, -1), Point(2, 2)],
        crs="EPSG:3857",
    )
    zones.to_file(basepath / f"{basename}_zone_centroid.SHP")

    links = gpd.GeoDataFrame(
        {
            "NO": [1, 2],
            "FROMNODENO": [1, 2],
            "TONODENO": [2, 1],
            "TYPENO": [1, 1],
            "LENGTH": ["10km", "10km"],
            "V0PRT": ["90km/h", "90km/h"],
            "CAPPRT": [30000.0, 30000.0],
            "NUMLANES": [2, 2],
        },
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (0, 0)])],
        crs="EPSG:3857",
    )
    links.to_file(basepath / f"{basename}_link.SHP")

    connectors = gpd.GeoDataFrame(
        {
            "NODENO": [1, 2, 1, 2],
            "ZONENO": [1, 2, 1, 2],
            "DIRECTION": ["O", "O", "D", "D"],
            "TYPENO": [0, 0, 0, 0],
            "LENGTH": ["1km", "1km", "1km", "1km"],
        },
        geometry=[
            LineString([(0, 0), (-1, -1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(-1, -1), (0, 0)]),
            LineString([(2, 2), (1, 1)]),
        ],
        crs="EPSG:3857",
    )
    connectors.to_file(basepath / f"{basename}_connector.SHP")

    return str(basepath), basename


def test_read_inputs_shapefile_synthetic(synthetic_visum_shapefiles):
    """Round-trips a minimal raw-Visum-style shapefile export through the
    full generate/skims/distribute/assign pipeline."""
    basepath, basename = synthetic_visum_shapefiles
    df_nodes, df_link_types, df_links = read_inputs_shapefile(basepath, basename)

    assert df_nodes["is_zone"].sum() == 2
    assert (~df_nodes["is_zone"]).sum() == 2
    # 2 directed road links + 4 connector links (O+D per zone)
    assert len(df_links) == 6

    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("all", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("all", "tcur", "exp", -0.02)
    model.assign("tcur")

    assert model.df_links["q"].notna().all()
    assert (model.df_links["q"] >= 0).all()
    assert model.df_links["q"].sum() > 0


@pytest.mark.skipif(
    not INTERNAL_EXCEL.exists(),
    reason="requires machine-local Internal/Networks data, not present in CI",
)
def test_classmethod():
    df_nodes, df_link_types, df_links = read_inputs_excel(str(INTERNAL_EXCEL))

    model = tfl.from_dataframes(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")

    assert model.df_links["q"].notna().all()


@pytest.mark.skipif(
    not INTERNAL_EXCEL.exists(),
    reason="requires machine-local Internal/Networks data, not present in CI",
)
def test_reading_csv():
    df_nodes, df_link_types, df_links = read_inputs_excel(str(INTERNAL_EXCEL))

    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")

    assert model.df_links["q"].notna().all()


@pytest.mark.skipif(
    not SHAPEFILE_BASEPATH.is_dir(),
    reason="requires machine-local PTV Visum shapefile export, not present in CI",
)
def test_reading_shapefile_ptv():
    basename = "I51_I76_siet_nulovy_stav"
    df_nodes, df_link_types, df_links = read_inputs_shapefile(
        str(SHAPEFILE_BASEPATH), basename, verbose=True
    )
    assert len(df_nodes) > 0
    assert len(df_links) > 0

    model = MTM()
    model.read_data(df_nodes, df_link_types, df_links)
    model.generate("ALL", "pop", "pop", 0.5)
    model.compute_skims()
    model.distribute("ALL", "tcur", "exp", -0.02)
    model.assign("tcur")

    assert model.df_links["q"].notna().all()
