"""
Diagnostic + fix for candidate station geometry issue.
Run this from your project root: .\venv\Scripts\python.exe fix_stations.py
"""
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

TARGET_CRS = "EPSG:32631"  # UTM zone 31N
PLACES = ["Ikeja, Lagos, Nigeria", "Yaba, Lagos, Nigeria", "Surulere, Lagos, Nigeria"]

print("--- Re-downloading road network for bounding box reference ---")
G = ox.graph_from_place(PLACES, network_type="drive")
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges_proj = edges.to_crs(TARGET_CRS)
road_bounds = edges_proj.total_bounds  # [minx, miny, maxx, maxy]
print(f"Road network bounding box (UTM 31N): {road_bounds}")

print("\n--- Re-downloading candidate stations ---")
tags = {
    "amenity": ["hospital", "fire_station", "clinic"],
    "emergency": "ambulance_station",
}
stations = ox.features_from_place(PLACES, tags)
print(f"Raw features returned: {len(stations)}")
print(f"Geometry types:\n{stations.geom_type.value_counts()}")

print("\n--- Converting to point geometries safely ---")
# For polygons, use centroid; but compute centroid BEFORE reprojecting to a
# projected CRS incorrectly, and validate geometry first.
stations = stations[stations.geometry.is_valid & ~stations.geometry.is_empty].copy()
stations["geometry"] = stations.geometry.apply(
    lambda geom: geom.centroid if geom.geom_type in ("Polygon", "MultiPolygon") else geom
)
stations = stations.set_geometry("geometry")
stations = stations.set_crs("EPSG:4326", allow_override=True) if stations.crs is None else stations

stations_proj = stations.to_crs(TARGET_CRS)

print("\n--- Station coordinates (UTM 31N) before filtering ---")
for idx, row in stations_proj.iterrows():
    name = row.get("name", "unnamed")
    amenity = row.get("amenity", row.get("emergency", "unknown"))
    x, y = row.geometry.x, row.geometry.y
    print(f"  {name} [{amenity}]: x={x:.1f}, y={y:.1f}")

print("\n--- Filtering to stations within/near the road network extent ---")
buffer_m = 3000  # 3km buffer around road network bbox, generous margin
minx, miny, maxx, maxy = road_bounds
valid_mask = (
    (stations_proj.geometry.x >= minx - buffer_m) &
    (stations_proj.geometry.x <= maxx + buffer_m) &
    (stations_proj.geometry.y >= miny - buffer_m) &
    (stations_proj.geometry.y <= maxy + buffer_m)
)
stations_clean = stations_proj[valid_mask].copy()
stations_dropped = stations_proj[~valid_mask].copy()

print(f"Kept: {len(stations_clean)} stations")
print(f"Dropped as outliers: {len(stations_dropped)} stations")
if len(stations_dropped) > 0:
    print("Dropped station details:")
    for idx, row in stations_dropped.iterrows():
        name = row.get("name", "unnamed")
        print(f"  DROPPED: {name} at x={row.geometry.x:.1f}, y={row.geometry.y:.1f}")

os.makedirs("data", exist_ok=True)
# Keep only simple columns for GeoPackage export compatibility
keep_cols = [c for c in ["name", "amenity", "emergency", "geometry"] if c in stations_clean.columns]
stations_export = stations_clean[keep_cols].copy()
for col in stations_export.columns:
    if col != "geometry":
        stations_export[col] = stations_export[col].astype(str)

stations_export.to_file("data/candidate_stations_osm.gpkg", layer="stations", driver="GPKG")
print("\nSaved cleaned stations to data/candidate_stations_osm.gpkg")

print("\n--- Regenerating QA map ---")
os.makedirs("outputs", exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 10))
edges_proj.plot(ax=ax, color="lightgrey", linewidth=0.8, alpha=0.9, zorder=1)
stations_clean.plot(ax=ax, color="red", marker="^", markersize=180, edgecolor="black", zorder=5)

ax.set_title("Visual QA Map (Fixed) - Ikeja, Yaba, & Surulere\nRoad Network + Cleaned Candidate Stations",
              fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
ax.grid(True, linestyle="--", alpha=0.5)
legend_elements = [
    Line2D([0], [0], color="lightgrey", lw=2, label="Road Network"),
    Line2D([0], [0], marker="^", color="w", markeredgecolor="black", label="Candidate Stations (cleaned)",
           markerfacecolor="red", markersize=12),
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig("outputs/qa_map_fixed.png", dpi=300)
print("Saved outputs/qa_map_fixed.png")
