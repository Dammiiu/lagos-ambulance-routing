"""
Step A (fixed for osmnx 2.1.1): Re-download road network using graph_from_bbox
with a 2km buffer expansion beyond the place-name polygon boundary.

The osmnx 2.1.1 graph_from_place does NOT support buffer_dist.
We use graph_from_bbox instead, computing a generous bbox that covers
Ikeja + Yaba + Surulere plus a ~2km margin.

WGS84 extent for the three districts + ~0.02deg (~2km) margin:
  Surulere south:  ~6.49N
  Ikeja north:     ~6.63N
  West edge:       ~3.32E
  East edge:       ~3.41E
  With 0.02deg buffer: 6.47S, 6.65N, 3.30W, 3.43E
"""
import os
import warnings
warnings.filterwarnings('ignore')

import osmnx as ox
import geopandas as gpd
import networkx as nx

def clean_columns_for_gpkg(df):
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            )
    return df

print("=" * 60)
print("STEP A (osmnx 2.1.1): Road Network Download via graph_from_bbox")
print("=" * 60)

# Bounding box (WGS84): covers Ikeja + Yaba + Surulere with ~2km buffer
# osmnx 2.x bbox format is (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)
# or actually (north, south, east, west) — check:
import inspect
sig = inspect.signature(ox.graph_from_bbox)
print(f"graph_from_bbox signature: {sig}")

# osmnx 2.x bbox = (left, bottom, right, top) i.e. (west, south, east, north) in lon/lat
# Confirmed from osmnx 2.x docs: bbox: (left, bottom, right, top) = (west, south, east, north)
bbox = (3.30, 6.47, 3.43, 6.65)  # (west, south, east, north)
print(f"\nDownloading drive network for bbox {bbox}...")
print("This covers Ikeja/Yaba/Surulere + ~2km buffer. May take 3-6 minutes...")

G = ox.graph_from_bbox(bbox, network_type="drive", retain_all=False)
print(f"Downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

print("\nAdding speeds and travel times...")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

print("Projecting to EPSG:32631 (UTM Zone 31N)...")
G_proj = ox.project_graph(G, to_crs="EPSG:32631")
print(f"Projected graph: {len(G_proj.nodes)} nodes, {len(G_proj.edges)} edges")

print(f"\nConnectivity check...")
if not nx.is_strongly_connected(G_proj):
    components = list(nx.strongly_connected_components(G_proj))
    sizes = sorted([len(c) for c in components], reverse=True)
    print(f"[LIMITATION] Not strongly connected: {len(components)} components.")
    print(f"  Top sizes: {sizes[:5]}")
    largest = max(nx.strongly_connected_components(G_proj), key=len)
    G_proj = G_proj.subgraph(largest).copy()
    print(f"  Using largest SCC: {len(G_proj.nodes)} nodes, {len(G_proj.edges)} edges")
else:
    print("Strongly connected.")

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
print(f"\nGeoDataFrames: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")

# Add schema alias columns
if "speed_kph" in edges_gdf.columns:
    edges_gdf["speed_kmh"] = edges_gdf["speed_kph"]
else:
    edges_gdf["speed_kmh"] = 30.0
if "length" in edges_gdf.columns:
    edges_gdf["length_m"] = edges_gdf["length"]
else:
    edges_gdf["length_m"] = 0.0
if "travel_time" in edges_gdf.columns:
    edges_gdf["time_min"] = edges_gdf["travel_time"] / 60.0
else:
    edges_gdf["time_min"] = edges_gdf["length_m"] / (edges_gdf["speed_kmh"] * 1000.0 / 60.0)

edges_gdf["speed_kmh"] = edges_gdf["speed_kmh"].fillna(30.0)
edges_gdf["length_m"]  = edges_gdf["length_m"].fillna(0.0)
edges_gdf["time_min"]  = edges_gdf["time_min"].fillna(0.0)

print("\nKey edge stats:")
print(f"  speed_kmh: min={edges_gdf['speed_kmh'].min():.1f}, max={edges_gdf['speed_kmh'].max():.1f}, mean={edges_gdf['speed_kmh'].mean():.1f}")
print(f"  length_m:  min={edges_gdf['length_m'].min():.1f}, max={edges_gdf['length_m'].max():.1f}, mean={edges_gdf['length_m'].mean():.1f}")
print(f"  time_min:  min={edges_gdf['time_min'].min():.4f}, max={edges_gdf['time_min'].max():.2f}, mean={edges_gdf['time_min'].mean():.4f}")

edges_clean = clean_columns_for_gpkg(edges_gdf)
nodes_clean = clean_columns_for_gpkg(nodes_gdf)

os.makedirs("data", exist_ok=True)
print("\nSaving edges to data/road_network_final.gpkg ...")
edges_clean.to_file("data/road_network_final.gpkg", layer="edges", driver="GPKG")
print("Saving nodes to data/road_network_nodes.gpkg ...")
nodes_clean.to_file("data/road_network_nodes.gpkg", layer="nodes", driver="GPKG")

# Verify
verify_e = gpd.read_file("data/road_network_final.gpkg")
print(f"\nVerified edges: {len(verify_e)} rows")
print(f"Bounds: {verify_e.total_bounds}")
print(f"Columns: {list(verify_e.columns)}")

verify_n = gpd.read_file("data/road_network_nodes.gpkg")
print(f"Verified nodes: {len(verify_n)} rows")
print("\nStep A COMPLETE.")
