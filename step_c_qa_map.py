"""
Step C: Color-coded QA map — medical (blue) vs fire (red)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

stations = gpd.read_file("data/ambulance_stations.gpkg")
incidents = gpd.read_file("data/incident_points.gpkg")
edges = gpd.read_file("data/road_network_final.gpkg")

print(f"Stations: {len(stations)} total")
print(stations["facility_type"].value_counts().to_string())
print(f"\nIncidents: {len(incidents)}")
print(incidents["type"].value_counts().to_string())
print(f"\nRoad network: {len(edges)} edges")
print(f"Bounds: {edges.total_bounds}")

fig, ax = plt.subplots(figsize=(14, 11))

# Road network
edges.plot(ax=ax, color="#cccccc", linewidth=0.5, alpha=0.7, zorder=1)

# Stations by type
med = stations[stations["facility_type"] == "medical"]
fire = stations[stations["facility_type"] == "fire"]
med.plot(ax=ax, color="#1565C0", marker="o", markersize=200, edgecolor="white", linewidth=1.5, zorder=5)
fire.plot(ax=ax, color="#B71C1C", marker="^", markersize=200, edgecolor="white", linewidth=1.5, zorder=5)

# Station labels
for _, row in stations.iterrows():
    ax.annotate(
        row["name"].split(",")[0][:25],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(3, 6), textcoords="offset points",
        fontsize=5, color="#222222",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
    )

# Incidents by type
inc_colors = {"Medical": "#2196F3", "RTA": "#FF9800", "Fire": "#F44336"}
for inc_type, color in inc_colors.items():
    subset = incidents[incidents["type"] == inc_type]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, marker="*", markersize=120, edgecolor="black", linewidth=0.5, zorder=6)

ax.set_title(
    "QA Map — Ambulance Stations & Simulated Incidents\nIkeja, Yaba & Surulere, Lagos State",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("UTM Easting (m)", fontsize=10)
ax.set_ylabel("UTM Northing (m)", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)

legend_elements = [
    mlines.Line2D([], [], color="#cccccc", lw=1.5, label="Road Network"),
    mlines.Line2D([], [], marker="o", color="w", markeredgecolor="white", 
                  markerfacecolor="#1565C0", markersize=12, label=f"Medical Station (n={len(med)})"),
    mlines.Line2D([], [], marker="^", color="w", markeredgecolor="white",
                  markerfacecolor="#B71C1C", markersize=12, label=f"Fire Station (n={len(fire)})"),
    mlines.Line2D([], [], marker="*", color="w", markeredgecolor="black",
                  markerfacecolor="#2196F3", markersize=11, label=f"Medical Incident (n={len(incidents[incidents['type']=='Medical'])})"),
    mlines.Line2D([], [], marker="*", color="w", markeredgecolor="black",
                  markerfacecolor="#FF9800", markersize=11, label=f"RTA Incident (n={len(incidents[incidents['type']=='RTA'])})"),
    mlines.Line2D([], [], marker="*", color="w", markeredgecolor="black",
                  markerfacecolor="#F44336", markersize=11, label=f"Fire Incident (n={len(incidents[incidents['type']=='Fire'])})"),
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, 
          facecolor="white", framealpha=0.95, fontsize=9, shadow=True)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/qa_map_final.png", dpi=250, bbox_inches="tight")
plt.close()
print("\nSaved outputs/qa_map_final.png")
print("Step C COMPLETE.")
