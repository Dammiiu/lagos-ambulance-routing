"""
src/routing.py

Implements all routing models from the GIS-based ambulance routing project
(Makanjuola, Thomas Oluwadamilare — Abiola Ajimobi Technical University, Ibadan, 2026)

MODELS:
  Model 1 — Route Optimization (§3.5):
      shortest_path(G, origin, dest) → path, time_min

  Model 2 — Closest Facility Analysis (§3.6):
      find_closest_station(G, nodes_gdf, incident_node, incident_type, stations_gdf)
      Type-aware: Medical/RTA → medical stations; Fire → fire stations

  Model 3 — Service Area (§3.7):
      [Implemented in service_area.py]

  Model 4 — Two-Leg Full Response Chain (Practical Extension):
      two_leg_route(G, nodes_gdf, inc_node, inc_type, stations_gdf)
      Leg 1: Dispatch station → Incident scene
      Leg 2: Incident scene → Nearest medical facility (for Medical/RTA)
      Fire incidents: Leg 1 only (no patient transport leg)

  Model 5 — Straight-Line Baseline Comparison:
      straight_line_time(pt1, pt2, speed_kmh=30) → time_min
"""
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

# --- Incident type to facility type mapping ---
INCIDENT_TO_FACILITY = {
    "Medical": "medical",
    "medical": "medical",
    "RTA":     "medical",
    "rta":     "medical",
    "Fire":    "fire",
    "fire":    "fire",
}


# --- Model 1: Route Optimization (Dijkstra Shortest Path) ---

def shortest_path(G: nx.DiGraph, origin_node, dest_node, blocked_edges=None):
    """
    Compute Dijkstra shortest path weighted by 'time_min'.
    Implements the Route Optimization Model (Section 3.5).

    Objective: min Σ w_ij · x_ij  subject to flow conservation constraints.

    Returns
    -------
    path     : list of node IDs ([] if unreachable)
    time_min : float travel time in minutes (inf if unreachable)
    """
    original_weights = {}
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                original_weights[(u, v)] = G[u][v][0].get("time_min", float("inf"))
                G[u][v][0]["time_min"] = float("inf")
    try:
        path = nx.dijkstra_path(G, origin_node, dest_node, weight="time_min")
        time_min = nx.dijkstra_path_length(G, origin_node, dest_node, weight="time_min")
        return path, float(time_min)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float("inf")
    finally:
        for (u, v), w in original_weights.items():
            G[u][v][0]["time_min"] = w


def dijkstra_distance(G: nx.DiGraph, origin_node, dest_node, blocked_edges=None) -> float:
    """Return travel time (minutes) on shortest path, or inf if unreachable."""
    original_weights = {}
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                original_weights[(u, v)] = G[u][v][0].get("time_min", float("inf"))
                G[u][v][0]["time_min"] = float("inf")
    try:
        return float(
            nx.dijkstra_path_length(G, origin_node, dest_node, weight="time_min")
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")
    finally:
        for (u, v), w in original_weights.items():
            G[u][v][0]["time_min"] = w


# --- Model 2: Closest Facility Analysis (Type-Aware, Section 3.6) ---

def find_closest_station(
    G: nx.DiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    incident_node,
    incident_type: str,
    stations_gdf: gpd.GeoDataFrame,
    blocked_edges: list = None,
):
    """
    Type-aware Closest Facility Analysis (Section 3.6).

    Implements: s* = argmin_{s_k ∈ S_type(i)} d(s_k, i)

    S_type(i) = stations whose facility_type matches the incident's requirement:
        Medical / RTA → medical stations
        Fire          → fire stations

    Parameters
    ----------
    G            : nx.DiGraph with 'time_min' edge weights
    nodes_gdf    : GeoDataFrame indexed by node IDs
    incident_node: node ID of the incident (already snapped to graph)
    incident_type: one of "Medical", "RTA", "Fire"
    stations_gdf : GeoDataFrame with columns [name, facility_type, geometry, node_id]

    Returns
    -------
    dict with: station_name, facility_type, station_node, path, network_time_min
    None if no reachable station found.
    """
    required_type = INCIDENT_TO_FACILITY.get(incident_type, "medical")
    candidates = stations_gdf[stations_gdf["facility_type"] == required_type].copy()

    if len(candidates) == 0:
        print(f"[routing] WARNING: No '{required_type}' stations — using all stations.")
        candidates = stations_gdf.copy()

    best = {"time_min": float("inf")}

    for _, srow in candidates.iterrows():
        s_node = srow.get("node_id")
        if s_node is None or s_node not in G.nodes:
            continue
        t = dijkstra_distance(G, s_node, incident_node, blocked_edges=blocked_edges)
        if t < best["time_min"]:
            best = {
                "station_name":  srow["name"],
                "facility_type": srow["facility_type"],
                "station_node":  s_node,
                "time_min":      t,
                "station_geom":  srow.geometry,
            }

    if best["time_min"] == float("inf"):
        return None

    path, _ = shortest_path(G, best["station_node"], incident_node, blocked_edges=blocked_edges)
    return {
        "station_name":    best["station_name"],
        "facility_type":   best["facility_type"],
        "station_node":    best["station_node"],
        "station_geom":    best["station_geom"],
        "path":            path,
        "network_time_min": best["time_min"],
    }


# --- Model 4: Two-Leg Full Response Chain ---

def two_leg_route(
    G: nx.DiGraph,
    nodes_gdf: gpd.GeoDataFrame,
    incident_node,
    incident_type: str,
    stations_gdf: gpd.GeoDataFrame,
    blocked_edges: list = None,
    custom_hospital_node = None,
):
    """
    Two-Leg Full Emergency Response Chain.

    Models the complete emergency response cycle:

      LEG 1: Dispatch → Incident scene
          The nearest appropriate dispatch station responds to the incident.
          (Medical/RTA → nearest medical station; Fire → nearest fire station)

      LEG 2: Incident scene → Nearest hospital (Medical/RTA only)
          After attending to the patient, the ambulance transports them to
          the nearest medical receiving facility.
          Fire incidents do not require a patient-transport leg.

    Parameters
    ----------
    G, nodes_gdf, incident_node, incident_type, stations_gdf : same as find_closest_station

    Returns
    -------
    dict with keys:
        leg1_station_name, leg1_facility_type, leg1_station_node,
        leg1_path, leg1_time_min, leg1_station_geom,
        leg2_hospital_name, leg2_hospital_node, leg2_path, leg2_time_min,
        leg2_hospital_geom,
        total_time_min,
        has_leg2 (bool)
    None if Leg 1 not reachable.
    """
    # --- Leg 1: dispatch station to incident ---
    leg1 = find_closest_station(G, nodes_gdf, incident_node, incident_type, stations_gdf, blocked_edges=blocked_edges)
    if leg1 is None:
        return None

    result = {
        "leg1_station_name":  leg1["station_name"],
        "leg1_facility_type": leg1["facility_type"],
        "leg1_station_node":  leg1["station_node"],
        "leg1_station_geom":  leg1["station_geom"],
        "leg1_path":          leg1["path"],
        "leg1_time_min":      leg1["network_time_min"],
        "has_leg2":           False,
        "leg2_hospital_name": None,
        "leg2_hospital_node": None,
        "leg2_hospital_geom": None,
        "leg2_path":          [],
        "leg2_time_min":      0.0,
        "total_time_min":     leg1["network_time_min"],
    }

    # --- Leg 2: incident scene to nearest medical facility (Medical/RTA only) ---
    if incident_type in ("Medical", "RTA", "medical", "rta"):
        if custom_hospital_node is not None and custom_hospital_node in G.nodes:
            # Find in stations_gdf
            h_match = stations_gdf[stations_gdf["node_id"] == custom_hospital_node]
            if len(h_match) > 0:
                hrow = h_match.iloc[0]
                h_name = hrow["name"]
                h_geom = hrow.geometry
            else:
                h_name = "Custom Destination"
                node_row = nodes_gdf.loc[custom_hospital_node]
                h_geom = node_row.geometry
                
            t = dijkstra_distance(G, incident_node, custom_hospital_node, blocked_edges=blocked_edges)
            if t < float("inf"):
                leg2_path, _ = shortest_path(G, incident_node, custom_hospital_node, blocked_edges=blocked_edges)
                result["has_leg2"]           = True
                result["leg2_hospital_name"] = h_name
                result["leg2_hospital_node"] = custom_hospital_node
                result["leg2_hospital_geom"] = h_geom
                result["leg2_path"]          = leg2_path
                result["leg2_time_min"]      = t
                result["total_time_min"]     = result["leg1_time_min"] + t
        else:
            # All medical stations act as potential receiving facilities
            medical_stations = stations_gdf[stations_gdf["facility_type"] == "medical"].copy()

            if len(medical_stations) == 0:
                return result

            best2 = {"time_min": float("inf")}
            for _, hrow in medical_stations.iterrows():
                h_node = hrow.get("node_id")
                if h_node is None or h_node not in G.nodes:
                    continue
                # From incident to hospital
                t = dijkstra_distance(G, incident_node, h_node, blocked_edges=blocked_edges)
                if t < best2["time_min"]:
                    best2 = {
                        "hospital_name": hrow["name"],
                        "hospital_node": h_node,
                        "hospital_geom": hrow.geometry,
                        "time_min":      t,
                    }

            if best2["time_min"] != float("inf"):
                leg2_path, _ = shortest_path(G, incident_node, best2["hospital_node"], blocked_edges=blocked_edges)
                result["has_leg2"]           = True
                result["leg2_hospital_name"] = best2["hospital_name"]
                result["leg2_hospital_node"] = best2["hospital_node"]
                result["leg2_hospital_geom"] = best2["hospital_geom"]
                result["leg2_path"]          = leg2_path
                result["leg2_time_min"]      = best2["time_min"]
                result["total_time_min"]     = result["leg1_time_min"] + best2["time_min"]

    return result


# --- Model 5: Straight-Line Baseline ---

def straight_line_time(pt1, pt2, speed_kmh: float = 50.0) -> float:
    """
    Straight-line travel time baseline (Section 3.5 comparison model).

    Uses Euclidean distance in projected CRS (metres).
    Speed default of 50-60 km/h matches Lagos average urban road speed.

    Returns time_min : float
    """
    dist_m = pt1.distance(pt2)
    speed_mps = speed_kmh * 1000.0 / 3600.0
    if speed_mps == 0:
        return float("inf")
    return (dist_m / speed_mps) / 60.0


# --- Utility: snap all stations/incidents to nearest graph nodes ---

def snap_stations_to_graph(
    stations_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Add 'node_id' and 'snap_dist_m' columns to stations_gdf."""
    from src.network_builder import snap_point_to_node
    stations = stations_gdf.copy()
    nids, dists = [], []
    for _, row in stations.iterrows():
        nid, dist = snap_point_to_node(nodes_gdf, row.geometry)
        nids.append(nid)
        dists.append(dist)
    stations["node_id"] = nids
    stations["snap_dist_m"] = dists
    return stations


def snap_incidents_to_graph(
    incidents_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Add 'node_id' and 'snap_dist_m' columns to incidents_gdf."""
    from src.network_builder import snap_point_to_node
    incidents = incidents_gdf.copy()
    nids, dists = [], []
    for _, row in incidents.iterrows():
        nid, dist = snap_point_to_node(nodes_gdf, row.geometry)
        nids.append(nid)
        dists.append(dist)
    incidents["node_id"] = nids
    incidents["snap_dist_m"] = dists
    return incidents


# --- Self-test ---

def _run_synthetic_test():
    """Verify all routing models on a small synthetic graph."""
    print("\n" + "=" * 60)
    print("SYNTHETIC GRAPH TEST")
    print("=" * 60)

    G = nx.DiGraph()
    edges = [
        (1,2,1.0),(2,1,1.0),(2,3,2.0),(3,2,2.0),(3,4,1.5),(4,3,1.5),
        (1,5,3.0),(5,1,3.0),(5,6,1.0),(6,5,1.0),(4,6,2.0),(6,4,2.0),
    ]
    for u, v, t in edges:
        G.add_edge(u, v, time_min=t, weight=t)
    coords = {1:(0,0),2:(1,0),3:(2,0),4:(3,0),5:(0,2),6:(3,2)}
    for n,(x,y) in coords.items():
        G.nodes[n]["x"] = x*1000
        G.nodes[n]["y"] = y*1000

    node_records = [{"osmid":n,"geometry":Point(x*1000,y*1000)} for n,(x,y) in coords.items()]
    nodes_gdf = gpd.GeoDataFrame(node_records, crs="EPSG:32631").set_index("osmid")

    stations_data = {
        "name":          ["Medical Alpha","Medical Beta","Fire Gamma"],
        "facility_type": ["medical","medical","fire"],
        "node_id":       [1, 5, 6],
        "geometry":      [Point(0,0), Point(0,2000), Point(3000,2000)]
    }
    stations_gdf = gpd.GeoDataFrame(stations_data, crs="EPSG:32631")

    # Model 1 test
    path, t = shortest_path(G, 1, 4)
    assert path == [1,2,3,4] and abs(t-4.5)<0.01, f"Model 1 failed: {path}, {t}"
    print("Model 1 (Dijkstra): PASS")

    # Model 2 tests
    r = find_closest_station(G, nodes_gdf, 4, "Medical", stations_gdf)
    assert r["facility_type"] == "medical", "Medical incident must go to medical station"
    print("Model 2 (Medical->medical): PASS")

    r = find_closest_station(G, nodes_gdf, 4, "Fire", stations_gdf)
    assert r["facility_type"] == "fire", "Fire incident must go to fire station"
    print("Model 2 (Fire->fire): PASS")

    r = find_closest_station(G, nodes_gdf, 2, "RTA", stations_gdf)
    assert r["facility_type"] == "medical", "RTA incident must go to medical station"
    print("Model 2 (RTA->medical): PASS")

    # Model 4 tests
    r4 = two_leg_route(G, nodes_gdf, 4, "Medical", stations_gdf)
    assert r4 is not None and r4["has_leg2"], "Medical 2-leg should have Leg 2"
    assert r4["leg1_facility_type"] == "medical"
    assert r4["total_time_min"] == r4["leg1_time_min"] + r4["leg2_time_min"]
    print("Model 4 (Medical 2-leg): PASS")

    r4f = two_leg_route(G, nodes_gdf, 4, "Fire", stations_gdf)
    assert r4f is not None and not r4f["has_leg2"], "Fire 2-leg should NOT have Leg 2"
    assert r4f["leg1_facility_type"] == "fire"
    print("Model 4 (Fire 1-leg only): PASS")

    # Model 5 test
    t = straight_line_time(Point(0,0), Point(3000,0), speed_kmh=50.0)
    assert abs(t - 6.0) < 0.01, f"Expected 6.0 min, got {t}"
    print("Model 5 (Straight-line): PASS")

    print("\nAll synthetic tests PASSED.")
