"""
run_all.py — Full pipeline runner (Steps D–G)
Regenerates all outputs: routing verification, service area, batch analysis,
charts. Uses the 2-leg route model for Medical/RTA incidents.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading graph (one-time build)...")
from src.network_builder import build_graph
from src.routing import (two_leg_route, find_closest_station,
                          straight_line_time,
                          snap_stations_to_graph, snap_incidents_to_graph)
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

os.makedirs("outputs", exist_ok=True)

G, nodes_gdf = build_graph()
print(f"Graph ready: {len(G.nodes)} nodes, {len(G.edges)} edges\n")

stations_gdf  = gpd.read_file("data/ambulance_stations.gpkg").to_crs("EPSG:32631")
stations_gdf  = snap_stations_to_graph(stations_gdf, nodes_gdf)
incidents_gdf = gpd.read_file("data/incident_points.gpkg").to_crs("EPSG:32631")
incidents_gdf = snap_incidents_to_graph(incidents_gdf, nodes_gdf)
incidents_gdf = incidents_gdf.reset_index(drop=True)

# =============================================================================
# STEP E: Real-data routing test (2-leg model, 6 sample incidents)
# =============================================================================
print("="*65)
print("STEP E: Real-data 2-leg routing test (6 sample incidents)")
print("="*65)

sample = incidents_gdf.groupby("type").apply(
    lambda x: x.head(2), include_groups=False
).reset_index(level=0)

print(f"\n{'ID':<4} {'Type':<8} {'Leg1 Station':<38} {'L1':>5}  {'Leg2 Hospital':<35} {'L2':>5}  {'Tot':>6}")
print("-"*106)
for _, row in sample.iterrows():
    r = two_leg_route(G, nodes_gdf, row["node_id"], row["type"], stations_gdf)
    if r:
        l2n = r["leg2_hospital_name"] or "—"
        l2t = f"{r['leg2_time_min']:.2f}" if r["has_leg2"] else "  —  "
        print(f"{row.name:<4} {row['type']:<8} {r['leg1_station_name']:<38} "
              f"{r['leg1_time_min']:>5.2f}  {l2n:<35} {l2t:>5}  {r['total_time_min']:>5.2f}m")
    else:
        print(f"{row.name:<4} {row['type']:<8} UNREACHABLE")

print("\nStep E COMPLETE.")

# =============================================================================
# STEP F: Service area analysis
# =============================================================================
print("\n" + "="*65)
print("STEP F: Service area analysis")
print("="*65)
from src.service_area import plot_service_area_map
edges_gdf = gpd.read_file("data/road_network_final.gpkg")
sa_results = plot_service_area_map(
    G, nodes_gdf, edges_gdf, stations_gdf, incidents_gdf,
    t_minutes_list=[5, 10, 15],
    output_path="outputs/service_area_map.png",
)
print("Step F COMPLETE.")

# =============================================================================
# STEP G: Batch analysis — all 20 incidents (single-leg for CSV/charts,
#         2-leg totals also stored)
# =============================================================================
print("\n" + "="*65)
print("STEP G: Batch analysis — all 20 incidents")
print("="*65)

TYPE_COLORS = {"Medical":"#1d4ed8","RTA":"#d97706","Fire":"#b91c1c"}

records = []
print(f"\n{'ID':<4} {'Type':<8} {'Station':<38} {'Fac':<8} "
      f"{'Net':>6} {'SL':>6} {'Overhead':>9}")
print("-"*88)

for i, inc in incidents_gdf.iterrows():
    inc_type = inc["type"]
    inc_node = inc["node_id"]
    inc_geom = inc.geometry

    r2 = two_leg_route(G, nodes_gdf, inc_node, inc_type, stations_gdf)

    if r2 is None:
        records.append({
            "incident_id": i, "incident_type": inc_type,
            "nearest_station": "UNREACHABLE", "facility_type": "N/A",
            "network_time_min": float("nan"),
            "straight_line_time_min": float("nan"),
            "percent_improvement": float("nan"),
            "leg2_hospital": "N/A",
            "leg2_time_min": float("nan"),
            "total_time_min": float("nan"),
        })
        print(f"{i:<4} {inc_type:<8} UNREACHABLE")
        continue

    s_geom = r2["leg1_station_geom"]
    sl = straight_line_time(s_geom, inc_geom, speed_kmh=12.0)
    net = r2["leg1_time_min"]
    pct = ((net - sl) / sl * 100) if sl > 0 else float("nan")

    records.append({
        "incident_id": i,
        "incident_type": inc_type,
        "nearest_station": r2["leg1_station_name"],
        "facility_type": r2["leg1_facility_type"],
        "network_time_min": round(net, 3),
        "straight_line_time_min": round(sl, 3),
        "percent_improvement": round(pct, 1),
        "leg2_hospital": r2["leg2_hospital_name"] if r2["has_leg2"] else "N/A",
        "leg2_time_min": round(r2["leg2_time_min"], 3) if r2["has_leg2"] else float("nan"),
        "total_time_min": round(r2["total_time_min"], 3),
    })
    print(f"{i:<4} {inc_type:<8} {r2['leg1_station_name']:<38} "
          f"{r2['leg1_facility_type']:<8} {net:>6.2f}m {sl:>6.2f}m {pct:>+8.1f}%")

results_df = pd.DataFrame(records)
results_df.to_csv("outputs/results.csv", index=False)
print(f"\nSaved: outputs/results.csv")

valid = results_df.dropna(subset=["network_time_min"])
print(f"\n{'='*65}")
print(f"SUMMARY (n={len(valid)})")
print(f"  Mean network time (Leg 1): {valid['network_time_min'].mean():.2f} min")
print(f"  Mean straight-line time:   {valid['straight_line_time_min'].mean():.2f} min")
print(f"  Mean overhead:             {valid['percent_improvement'].mean():+.1f}%")
has_l2 = valid.dropna(subset=["leg2_time_min"])
if len(has_l2):
    print(f"  Mean total (2-leg) time:   {has_l2['total_time_min'].mean():.2f} min")
    print(f"  Incidents with Leg 2:      {len(has_l2)}/{len(valid)}")

# ── Chart 1: Comparison bar chart ─────────────────────────────────────────────
valid_s = valid.sort_values("incident_id")
x = np.arange(len(valid_s))
w = 0.35
bar_cols = [TYPE_COLORS.get(t,"#888") for t in valid_s["incident_type"]]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#0b1120")
ax.set_facecolor("#0d1828")

b1 = ax.bar(x - w/2, valid_s["network_time_min"], w,
            color=bar_cols, alpha=0.9, edgecolor="#1e2d40", linewidth=0.6,
            label="Network (Dijkstra)")
b2 = ax.bar(x + w/2, valid_s["straight_line_time_min"], w,
            color="#475569", alpha=0.75, edgecolor="#1e2d40", linewidth=0.6,
            hatch="//", label="Straight-line baseline")

mean_net = valid_s["network_time_min"].mean()
ax.axhline(mean_net, color="#38bdf8", linestyle="--", linewidth=1.8, alpha=0.9)
ax.text(len(x)-0.5, mean_net+0.08,
        f"Mean Leg 1: {mean_net:.2f} min",
        color="#38bdf8", fontsize=9, ha="right", va="bottom")

legend_patches = [
    mpatches.Patch(facecolor="#1d4ed8", label="Medical incident"),
    mpatches.Patch(facecolor="#d97706", label="RTA incident"),
    mpatches.Patch(facecolor="#b91c1c", label="Fire incident"),
    mpatches.Patch(facecolor="#475569", hatch="//", label="Straight-line baseline"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
          facecolor="#0d1828", edgecolor="#1e2d40", labelcolor="white")

ax.set_xlabel("Incident ID", fontsize=11, color="#94a3b8")
ax.set_ylabel("Response Time (minutes)", fontsize=11, color="#94a3b8")
ax.set_title(
    "Network-Optimised (Dijkstra) vs Straight-Line Response Times\n"
    "Model 1 & 5 Comparison — All 20 Incidents · Ikeja, Yaba & Surulere, Lagos",
    fontsize=13, fontweight="bold", color="#e2e8f0", pad=14
)
ax.set_xticks(x)
ax.set_xticklabels([f"#{int(r)}" for r in valid_s["incident_id"]],
                   fontsize=8, rotation=45, color="#94a3b8")
ax.tick_params(colors="#4a5568")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2d40")
ax.grid(axis="y", linestyle="--", alpha=0.25, color="#334155")
ax.set_ylim(0, max(valid_s["network_time_min"].max(),
                   valid_s["straight_line_time_min"].max()) * 1.3)
plt.tight_layout()
plt.savefig("outputs/comparison_chart.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: outputs/comparison_chart.png")

# ── Chart 2: Response time histogram ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0b1120")
ax.set_facecolor("#0d1828")

bins = np.arange(0, valid["network_time_min"].max() + 2, 1.5)
bottom = np.zeros(len(bins)-1)
for itype in ["Medical","RTA","Fire"]:
    vals = valid[valid["incident_type"]==itype]["network_time_min"].values
    if len(vals) > 0:
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(bins[:-1]+0.75, counts, width=1.5, bottom=bottom,
               color=TYPE_COLORS.get(itype,"#888"), alpha=0.85,
               label=f"{itype} incidents", edgecolor="#1e2d40", linewidth=0.5)
        bottom += counts

for t_thresh, c, lbl in [(5,"#22c55e","5 min"),(10,"#f59e0b","10 min"),(15,"#ef4444","15 min")]:
    ax.axvline(t_thresh, color=c, linestyle="--", linewidth=2, alpha=0.85,
               label=f"{lbl} threshold")

ax.set_xlabel("Network Response Time — Leg 1 (minutes)", fontsize=11, color="#94a3b8")
ax.set_ylabel("Number of Incidents", fontsize=11, color="#94a3b8")
ax.set_title(
    "Distribution of Ambulance Response Times\n"
    "Model 1 — Dijkstra Optimal Routing · All 20 Simulated Incidents",
    fontsize=13, fontweight="bold", color="#e2e8f0", pad=14
)
ax.legend(fontsize=9, facecolor="#0d1828", edgecolor="#1e2d40", labelcolor="white")
ax.tick_params(colors="#4a5568")
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2d40")
ax.grid(axis="y", linestyle="--", alpha=0.25, color="#334155")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("outputs/response_time_hist.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: outputs/response_time_hist.png")

print("\n" + "="*65)
print("ALL STEPS D-G COMPLETE — outputs/ is ready")
print("="*65)

import shutil
shutil.make_archive("outputs", "zip", "outputs")
print("Created outputs.zip for easy download.")

print("\nTo launch the app:")
print("  venv\\Scripts\\streamlit run src\\app.py")
