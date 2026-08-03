import geopandas as gpd
import sys

print("Checking data files:")
for name in ["candidate_stations_osm.gpkg", "incident_points.gpkg", "road_network_final.gpkg"]:
    try:
        gdf = gpd.read_file(f"data/{name}")
        print(f"\nFile: {name}")
        print(f"Shape: {gdf.shape}")
        print(f"Columns: {gdf.columns.tolist()}")
        print(f"CRS: {gdf.crs}")
        print(gdf.head(2))
    except Exception as e:
        print(f"Error reading {name}: {e}")

try:
    import osmnx as ox
    print("\nosmnx is installed. Version:", ox.__version__)
except ImportError:
    print("\nosmnx is NOT installed.")
