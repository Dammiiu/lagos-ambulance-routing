"""
src/network_builder.py

Builds a NetworkX DiGraph from road_network_final.gpkg.
- Edge weight: time_min (travel time in minutes)
- Respects oneway field
- Stores geometry and street name on each edge (for route rendering + turn-by-turn)
- Snaps arbitrary points to nearest graph nodes
- Validates connectivity, uses largest SCC if needed
- Fast loading: Caches the built graph to a pickle file for sub-second startup times.
- Congestion Factor: Scales travel times by 2.2 to represent real-world Lagos traffic delays.

Public API:
    build_graph(edges_path, nodes_path) -> (G, nodes_gdf)
    snap_point_to_node(nodes_gdf, point_geom) -> (node_id, dist_m)
    get_node_coords(nodes_gdf, node_id) -> (x, y)
    path_to_detailed_coords(G, path) -> list of (x, y) in UTM
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import math
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString

EDGES_PATH = "data/road_network_final.gpkg"
NODES_PATH = "data/road_network_nodes.gpkg"
CACHE_PATH = "data/road_network_cache.pkl"
LAGOS_CONGESTION_FACTOR = 2.2  # Real-world traffic weighting delay factor


def build_graph(edges_path: str = EDGES_PATH, nodes_path: str = NODES_PATH, cache_path: str = CACHE_PATH):
    """
    Load road network and return a NetworkX DiGraph. Uses pickle cache if available.

    Edge attributes stored on graph:
        time_min   : float — travel time (minutes) scaled for Lagos traffic, Dijkstra weight
        length_m   : float — road segment length (metres)
        speed_kmh  : float — average speed
        name       : str   — OSM street name
        geometry   : list  — list of (x,y) UTM coord tuples

    Returns
    -------
    G          : nx.DiGraph
    nodes_gdf  : GeoDataFrame  (index = osmid)
    """
    # ── Check Pickle Cache ────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            print(f"[network_builder] Loading cached graph from: {cache_path}")
            with open(cache_path, "rb") as f:
                G, nodes_gdf = pickle.load(f)
            print(f"[network_builder] Loaded cached graph successfully: {len(G.nodes)} nodes, {len(G.edges)} edges.")
            return G, nodes_gdf
        except Exception as e:
            print(f"[network_builder] Error loading cache: {e}. Rebuilding graph...")

    # ── Rebuild Graph ─────────────────────────────────────────────────────────
    print("[network_builder] Loading edges from:", edges_path)
    edges_gdf = gpd.read_file(edges_path)
    print(f"[network_builder] Loaded {len(edges_gdf)} edges.")

    # Load nodes
    try:
        nodes_gdf = gpd.read_file(nodes_path)
        print(f"[network_builder] Loaded {len(nodes_gdf)} nodes from:", nodes_path)
    except Exception:
        print("[network_builder] Could not load nodes file — reconstructing from edge endpoints.")
        nodes_gdf = None

    G = nx.DiGraph()

    for _, row in edges_gdf.iterrows():
        u = row.get("u") or row.get("from")
        v = row.get("v") or row.get("to")
        if u is None or v is None:
            continue

        # Get raw travel time and apply Lagos traffic congestion adjustment
        time_min_raw = float(row["time_min"]) if row["time_min"] is not None else 0.0
        time_min = time_min_raw * LAGOS_CONGESTION_FACTOR

        length_m  = float(row["length_m"])  if row["length_m"]  is not None else 0.0
        speed_kmh = float(row["speed_kmh"]) if row["speed_kmh"] is not None else 30.0

        # Adjust segment speed attribute to match the congestion factor
        adjusted_speed = speed_kmh / LAGOS_CONGESTION_FACTOR

        # Street name
        raw_name = row.get("name", "")
        street_name = str(raw_name).strip() if raw_name is not None else ""
        if street_name.lower() in ("nan", "none", ""):
            street_name = ""

        # Edge geometry
        geom = row.geometry
        if geom is not None and hasattr(geom, "coords"):
            geom_coords = list(geom.coords)
        else:
            geom_coords = []

        oneway_val = row.get("oneway", False)
        if isinstance(oneway_val, str):
            oneway = oneway_val.strip().lower() in ("true", "yes", "1")
        else:
            oneway = bool(oneway_val)

        # Extract highway tag
        highway_val = row.get("highway", "unclassified")
        if isinstance(highway_val, (list, tuple)):
            highway_val = highway_val[0] if len(highway_val) > 0 else "unclassified"
        highway_str = str(highway_val).lower().strip()

        edge_attrs = {
            "time_min":      max(time_min, 1e-6),
            "base_time_min": max(time_min, 1e-6),
            "length_m":      length_m,
            "speed_kmh":     adjusted_speed,
            "weight":        max(time_min, 1e-6),
            "name":          street_name,
            "geometry":      geom_coords,
            "highway":       highway_str,
            "wrong_way":     False,
        }

        G.add_edge(u, v, **edge_attrs)
        
        rev_attrs = dict(edge_attrs)
        rev_attrs["geometry"] = list(reversed(geom_coords))
        if oneway:
            rev_attrs["wrong_way"] = True
            rev_attrs["time_min"] = float('inf')
            rev_attrs["weight"] = float('inf')
        
        G.add_edge(v, u, **rev_attrs)

    print(f"[network_builder] Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Attach coordinates to nodes
    if nodes_gdf is not None:
        if "osmid" in nodes_gdf.columns:
            nodes_gdf = nodes_gdf.set_index("osmid")
        for node_id in G.nodes:
            if node_id in nodes_gdf.index:
                row = nodes_gdf.loc[node_id]
                G.nodes[node_id]["x"] = row.geometry.x
                G.nodes[node_id]["y"] = row.geometry.y

    # Fall back coordinates
    missing_coords = [n for n in G.nodes if "x" not in G.nodes[n]]
    if missing_coords:
        print(f"[network_builder] Reconstructing coordinates for {len(missing_coords)} nodes...")
        node_coords = {}
        for _, row in edges_gdf.iterrows():
            u = row.get("u") or row.get("from")
            v = row.get("v") or row.get("to")
            if row.geometry is not None and hasattr(row.geometry, "coords"):
                coords = list(row.geometry.coords)
                if u not in node_coords and coords:
                    node_coords[u] = coords[0]
                if v not in node_coords and coords:
                    node_coords[v] = coords[-1]
        for node_id, (x, y) in node_coords.items():
            if node_id in G.nodes:
                G.nodes[node_id]["x"] = x
                G.nodes[node_id]["y"] = y

    if nodes_gdf is None or len(missing_coords) > 0:
        node_records = []
        for node_id in G.nodes:
            nd = G.nodes[node_id]
            if "x" in nd and "y" in nd:
                node_records.append({
                    "osmid": node_id,
                    "x": nd["x"], "y": nd["y"],
                    "geometry": Point(nd["x"], nd["y"])
                })
        nodes_gdf = gpd.GeoDataFrame(node_records, crs=edges_gdf.crs).set_index("osmid")

    # Validate connectivity
    print("[network_builder] Checking strong connectivity...")
    if not nx.is_strongly_connected(G):
        components = list(nx.strongly_connected_components(G))
        sizes = sorted([len(c) for c in components], reverse=True)
        print(f"[LIMITATION] Graph not strongly connected: {len(components)} SCCs. Top: {sizes[:5]}")
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_scc).copy()
        scc_ids = set(largest_scc)
        nodes_gdf = nodes_gdf[nodes_gdf.index.isin(scc_ids)]
        print(f"[network_builder] After SCC: {len(G.nodes)} nodes, {len(G.edges)} edges")
    else:
        print("[network_builder] Graph is strongly connected.")

    # Save to Cache
    try:
        print(f"[network_builder] Saving built graph to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((G, nodes_gdf), f)
        print("[network_builder] Saved graph successfully.")
    except Exception as e:
        print(f"[network_builder] Error saving cache: {e}")

    print("[network_builder] Graph build complete.")
    return G, nodes_gdf


def path_to_detailed_coords(G: nx.DiGraph, path: list, exact_dest_coord: tuple = None) -> list:
    """Convert Dijkstra path into UTM coords following real OSM LineString curves."""
    all_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ed_dict = G.get_edge_data(u, v) or {}
        edge_data = ed_dict[0] if 0 in ed_dict else ed_dict
        
        if edge_data and edge_data.get("geometry"):
            seg = edge_data["geometry"]
            if all_coords and seg and seg[0] == all_coords[-1]:
                seg = seg[1:]
            all_coords.extend(seg)
        else:
            if "x" in G.nodes[u] and "x" in G.nodes[v]:
                pt_u = (G.nodes[u]["x"], G.nodes[u]["y"])
                pt_v = (G.nodes[v]["x"], G.nodes[v]["y"])
                if not all_coords or all_coords[-1] != pt_u:
                    all_coords.append(pt_u)
                all_coords.append(pt_v)

    if exact_dest_coord and len(all_coords) >= 2:
        try:
            # Create LineString from all coordinates
            line = LineString(all_coords)
            pt = Point(exact_dest_coord)
            
            # Find the distance along the line to the projected point
            proj_dist = line.project(pt)
            
            # If the projected distance is very small, or the line is very short, keep it as is
            if proj_dist > 0:
                truncated_coords = []
                accumulated = 0.0
                
                # Iterate through segments
                for i in range(1, len(all_coords)):
                    p1 = all_coords[i-1]
                    p2 = all_coords[i]
                    d = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                    
                    if accumulated + d >= proj_dist:
                        # The projected point falls on this segment
                        rem = proj_dist - accumulated
                        ratio = rem / d if d > 0 else 0
                        nx_coord = p1[0] + ratio * (p2[0] - p1[0])
                        ny_coord = p1[1] + ratio * (p2[1] - p1[1])
                        
                        truncated_coords.append(p1)
                        # Add the exact destination to ensure it terminates exactly there
                        
                        # Only add the projected point if it's not identical to p1
                        if ratio > 0.001:
                            truncated_coords.append((nx_coord, ny_coord))
                        
                        # Calculate distance between projected point and exact dest
                        snap_dist = math.sqrt((nx_coord - exact_dest_coord[0])**2 + (ny_coord - exact_dest_coord[1])**2)
                        
                        # Only append the exact destination if it's reasonably close (e.g. 500m)
                        # to prevent wild jumps across the map if snapping went wrong
                        if snap_dist < 500:
                            truncated_coords.append(exact_dest_coord)
                            
                        all_coords = truncated_coords
                        break
                    
                    truncated_coords.append(p1)
                    accumulated += d
                    
        except Exception as e:
            print(f"[network_builder] Warning: Route truncation failed: {e}")

    return all_coords


def snap_point_to_node(nodes_gdf: gpd.GeoDataFrame, point_geom) -> tuple:
    """Find the nearest graph node to a point geometry (returns (node_id, dist_m))."""
    distances = nodes_gdf.geometry.distance(point_geom)
    nearest_idx = distances.idxmin()
    dist_m = float(distances.min())
    return nearest_idx, dist_m


def get_node_coords(nodes_gdf: gpd.GeoDataFrame, node_id: int):
    """Return (x, y) coordinates of a node."""
    row = nodes_gdf.loc[node_id]
    return float(row.geometry.x), float(row.geometry.y)


if __name__ == "__main__":
    G, nodes = build_graph()
    print(f"\nFinal graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
