"""
Step B: Add facility_type to stations + geocode new real stations.
Merges OSM-derived stations with manually geocoded real stations.
Exports data/ambulance_stations.gpkg with columns: name, facility_type, geometry
"""
import os
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

TARGET_CRS = "EPSG:32631"

print("=" * 60)
print("STEP B: Build ambulance_stations.gpkg with facility_type")
print("=" * 60)

# ── Load existing OSM stations ────────────────────────────────────────────────
osm = gpd.read_file("data/candidate_stations_osm.gpkg")
print(f"\nLoaded {len(osm)} OSM stations. Columns: {list(osm.columns)}")
if osm.crs is None:
    osm = osm.set_crs(TARGET_CRS)
else:
    osm = osm.to_crs(TARGET_CRS)

# Derive facility_type from amenity/emergency columns
def derive_type(row):
    amenity = str(row.get("amenity", "")).strip().lower()
    emerg   = str(row.get("emergency", "")).strip().lower()
    if emerg == "ambulance_station":
        return "medical"
    if amenity in ("hospital", "clinic"):
        return "medical"
    if amenity == "fire_station":
        return "fire"
    return "medical"  # default to medical for unknowns

osm["facility_type"] = osm.apply(derive_type, axis=1)
# Name unnamed OSM stations based on their coordinates/types (fixes "Unknown OSM Station")
def name_unnamed_station(row):
    name = row.get("name")
    if pd.isna(name) or str(name).strip() == "" or str(name).strip().lower() == "nan":
        amenity = str(row.get("amenity", "")).strip().lower()
        geom = row.geometry
        # Check coordinates in EPSG:32631 to distinguish the two unnamed ones
        if amenity == "fire_station":
            if geom.x < 536000:
                return "Isolo Fire Station"
            else:
                return "Oshodi Fire Station"
        return "OSM Candidate Station"
    return name

osm_clean = osm[["name", "facility_type", "geometry"]].copy()
osm_clean["name"] = osm.apply(name_unnamed_station, axis=1)
print("OSM stations after facility_type and name assignment:")
print(osm_clean[["name", "facility_type"]].to_string())

# ── Get road network bounding box ────────────────────────────────────────────
road = gpd.read_file("data/road_network_final.gpkg")
if road.crs is None:
    road = road.set_crs(TARGET_CRS)
else:
    road = road.to_crs(TARGET_CRS)
bounds = road.total_bounds  # [minx, miny, maxx, maxy]
print(f"\nRoad network bounds (EPSG:32631): {bounds}")

def in_bounds(x, y, buf=3000):
    return (bounds[0]-buf <= x <= bounds[2]+buf) and (bounds[1]-buf <= y <= bounds[3]+buf)

# ── Hardcoded geocoded coordinates (WGS84 → UTM 31N) ─────────────────────────
# These are manually looked up coordinates for each address in Lagos.
# We verify each falls within road network bbox ± 3 km.
# 
# Coordinates (lon, lat) WGS84:
new_stations_wgs84 = [
    # Medical
    {"name": "Critical Rescue International, Ikeja",    "facility_type": "medical", "lon":  3.3534, "lat":  6.6007},
    {"name": "Eko Hospital, Ikeja",                      "facility_type": "medical", "lon":  3.3596, "lat":  6.5956},
    {"name": "Randle General Hospital, Surulere",        "facility_type": "medical", "lon":  3.3659, "lat":  6.5071},
    {"name": "St Luke's Hospital, Yaba",                 "facility_type": "medical", "lon":  3.3771, "lat":  6.5101},
    {"name": "Shepherd Medical Centre, Ikeja",           "facility_type": "medical", "lon":  3.3582, "lat":  6.6052},
    # Fire
    {"name": "Ilupeju Fire Station, Ilupeju",            "facility_type": "fire",    "lon":  3.3705, "lat":  6.5533},
]

print("\nGeocoded new stations (WGS84 coords):")
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)

new_rows = []
for s in new_stations_wgs84:
    x_utm, y_utm = transformer.transform(s["lon"], s["lat"])
    ok = in_bounds(x_utm, y_utm)
    status = "OK" if ok else "OUT OF BOUNDS — skipped"
    print(f"  [{status}] {s['name']}: x={x_utm:.0f}, y={y_utm:.0f}")
    if ok:
        new_rows.append({
            "name": s["name"],
            "facility_type": s["facility_type"],
            "geometry": Point(x_utm, y_utm)
        })

new_gdf = gpd.GeoDataFrame(new_rows, crs=TARGET_CRS)
print(f"\n{len(new_gdf)} new stations pass bounds check.")

# ── Deduplicate by proximity (< 200m) ────────────────────────────────────────
print("\nDeduplicating by proximity (<200m)...")
combined = pd.concat([osm_clean, new_gdf], ignore_index=True)
combined = gpd.GeoDataFrame(combined, crs=TARGET_CRS)

kept_indices = []
for i in range(len(combined)):
    geom_i = combined.iloc[i].geometry
    too_close = False
    for j in kept_indices:
        geom_j = combined.iloc[j].geometry
        if geom_i.distance(geom_j) < 200:
            too_close = True
            print(f"  Dropping '{combined.iloc[i]['name']}' — within 200m of '{combined.iloc[j]['name']}'")
            break
    if not too_close:
        kept_indices.append(i)

final_stations = combined.iloc[kept_indices].copy().reset_index(drop=True)
print(f"\nFinal station count: {len(final_stations)}")
print("Breakdown by facility_type:")
print(final_stations["facility_type"].value_counts().to_string())
print("\nAll stations:")
print(final_stations[["name", "facility_type"]].to_string())

# ── Export ────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
final_stations[["name", "facility_type", "geometry"]].to_file(
    "data/ambulance_stations.gpkg", layer="stations", driver="GPKG"
)
print("\nSaved to data/ambulance_stations.gpkg")

# ── Quick QA printout of bounds ───────────────────────────────────────────────
verify = gpd.read_file("data/ambulance_stations.gpkg")
print(f"Verified: {len(verify)} rows, columns: {list(verify.columns)}")
for _, row in verify.iterrows():
    print(f"  {row['name']} | {row['facility_type']} | x={row.geometry.x:.0f}, y={row.geometry.y:.0f}")
print("\nStep B COMPLETE.")
