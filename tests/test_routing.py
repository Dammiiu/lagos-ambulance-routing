"""
tests/test_routing.py — Unit tests for all routing models
Models covered: 1 (Dijkstra), 2 (Closest Facility), 4 (Two-Leg), 5 (Straight-Line)
Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from src.routing import (
    shortest_path, dijkstra_distance, find_closest_station,
    two_leg_route, straight_line_time, INCIDENT_TO_FACILITY
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    """6-node DiGraph: medical station@1, medical@5, fire@6, incident targets: 3,4."""
    G = nx.DiGraph()
    edges = [
        (1, 2, 1.0), (2, 1, 1.0),
        (2, 3, 2.0), (3, 2, 2.0),
        (3, 4, 1.5), (4, 3, 1.5),
        (1, 5, 3.0), (5, 1, 3.0),
        (5, 6, 1.0), (6, 5, 1.0),
        (4, 6, 2.0), (6, 4, 2.0),
    ]
    for u, v, t in edges:
        G.add_edge(u, v, time_min=t, weight=t)
    coords = {1:(0,0), 2:(1,0), 3:(2,0), 4:(3,0), 5:(0,2), 6:(3,2)}
    for n, (x, y) in coords.items():
        G.nodes[n]["x"] = x * 1000
        G.nodes[n]["y"] = y * 1000
    return G


@pytest.fixture
def nodes_gdf():
    coords = {1:(0,0), 2:(1,0), 3:(2,0), 4:(3,0), 5:(0,2), 6:(3,2)}
    records = [{"osmid": n, "geometry": Point(x*1000, y*1000)} for n, (x,y) in coords.items()]
    return gpd.GeoDataFrame(records, crs="EPSG:32631").set_index("osmid")


@pytest.fixture
def stations_gdf():
    return gpd.GeoDataFrame({
        "name":          ["Medical Alpha", "Medical Beta", "Fire Gamma"],
        "facility_type": ["medical",       "medical",      "fire"],
        "node_id":       [1,               5,              6],
        "geometry":      [Point(0, 0),     Point(0, 2000), Point(3000, 2000)],
    }, crs="EPSG:32631")


# ── Model 1: Dijkstra shortest path ─────────────────────────────────────────

class TestDijkstra:
    def test_direct_path(self, simple_graph):
        path, t = shortest_path(simple_graph, 1, 2)
        assert path == [1, 2]
        assert abs(t - 1.0) < 1e-6

    def test_multi_hop_path(self, simple_graph):
        path, t = shortest_path(simple_graph, 1, 4)
        assert path[0] == 1 and path[-1] == 4
        assert t > 0

    def test_optimal_not_direct(self, simple_graph):
        # 1→2→3→4 = 1+2+1.5 = 4.5 min
        path, t = shortest_path(simple_graph, 1, 4)
        assert abs(t - 4.5) < 1e-6

    def test_no_path(self, simple_graph):
        path, t = shortest_path(simple_graph, 1, 99)
        assert path == []
        assert t == float("inf")

    def test_self_path(self, simple_graph):
        _, t = shortest_path(simple_graph, 3, 3)
        assert t == 0.0


# ── Model 2: Type-aware closest facility ─────────────────────────────────────

class TestClosestFacility:
    def test_medical_incident_routes_to_medical(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        assert r is not None and r["facility_type"] == "medical"

    def test_fire_incident_routes_to_fire(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 4, "Fire", stations_gdf)
        assert r is not None and r["facility_type"] == "fire"

    def test_rta_incident_routes_to_medical(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 2, "RTA", stations_gdf)
        assert r is not None
        assert r["facility_type"] == "medical", f"RTA→medical expected, got {r['facility_type']}"

    def test_result_has_required_fields(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 3, "Medical", stations_gdf)
        assert r is not None
        for f in ["station_name","facility_type","station_node","path","network_time_min"]:
            assert f in r, f"Missing field: {f}"

    def test_path_endpoints(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 4, "Fire", stations_gdf)
        if r and r["path"]:
            assert r["path"][-1] == 4
            assert r["path"][0] == r["station_node"]

    def test_fire_does_not_route_to_medical(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 3, "Medical", stations_gdf)
        assert r["facility_type"] != "fire"

    def test_medical_does_not_route_to_fire(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 3, "Fire", stations_gdf)
        assert r["facility_type"] != "medical"


# ── Model 4: Two-leg full response chain ─────────────────────────────────────

class TestTwoLegRoute:
    def test_medical_has_leg2(self, simple_graph, nodes_gdf, stations_gdf):
        """Medical incident → Leg 2 (scene→hospital) must exist."""
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        assert r is not None and r["has_leg2"] is True

    def test_rta_has_leg2(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 3, "RTA", stations_gdf)
        assert r is not None and r["has_leg2"] is True

    def test_fire_no_leg2(self, simple_graph, nodes_gdf, stations_gdf):
        """Fire incident → NO Leg 2."""
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Fire", stations_gdf)
        assert r is not None and r["has_leg2"] is False

    def test_total_time_equals_leg1_plus_leg2(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        assert r is not None
        assert abs(r["total_time_min"] - (r["leg1_time_min"] + r["leg2_time_min"])) < 1e-6

    def test_fire_total_equals_leg1_only(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Fire", stations_gdf)
        assert r is not None
        assert abs(r["total_time_min"] - r["leg1_time_min"]) < 1e-6
        assert r["leg2_time_min"] == 0.0

    def test_leg1_facility_type_matches_incident(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        assert r["leg1_facility_type"] == "medical"
        r2 = two_leg_route(simple_graph, nodes_gdf, 4, "Fire", stations_gdf)
        assert r2["leg1_facility_type"] == "fire"

    def test_required_keys_present(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        for k in ["leg1_station_name","leg1_facility_type","leg1_station_node",
                  "leg1_path","leg1_time_min","leg1_station_geom",
                  "leg2_hospital_name","leg2_path","leg2_time_min",
                  "total_time_min","has_leg2"]:
            assert k in r, f"Missing key: {k}"

    def test_leg2_uses_medical_facility(self, simple_graph, nodes_gdf, stations_gdf):
        r = two_leg_route(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        if r["has_leg2"] and r["leg2_hospital_name"]:
            hosp = stations_gdf[stations_gdf["name"] == r["leg2_hospital_name"]]
            if len(hosp) > 0:
                assert hosp.iloc[0]["facility_type"] == "medical"


# ── Model 5: Straight-line baseline ─────────────────────────────────────────

class TestStraightLineTime:
    def test_basic_calculation(self):
        t = straight_line_time(Point(0, 0), Point(3000, 0), speed_kmh=30.0)
        assert abs(t - 6.0) < 0.01

    def test_zero_distance(self):
        assert straight_line_time(Point(0, 0), Point(0, 0), speed_kmh=30.0) == 0.0

    def test_result_is_nonnegative(self, simple_graph, nodes_gdf, stations_gdf):
        r = find_closest_station(simple_graph, nodes_gdf, 4, "Medical", stations_gdf)
        if r:
            srow = stations_gdf[stations_gdf["node_id"] == r["station_node"]].iloc[0]
            sl = straight_line_time(srow.geometry, Point(3000, 0), speed_kmh=30.0)
            assert sl >= 0.0 and isinstance(sl, float)


# ── Incident type mapping ────────────────────────────────────────────────────

class TestIncidentTypeMapping:
    def test_medical_maps_to_medical(self):
        assert INCIDENT_TO_FACILITY["Medical"] == "medical"

    def test_rta_maps_to_medical(self):
        assert INCIDENT_TO_FACILITY["RTA"] == "medical"

    def test_fire_maps_to_fire(self):
        assert INCIDENT_TO_FACILITY["Fire"] == "fire"
