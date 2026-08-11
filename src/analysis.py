"""
src/analysis.py

Step G — Batch analysis across all 20 incidents.

Results table columns:
    incident_id, incident_type, nearest_station, facility_type,
    network_time_min, straight_line_time_min, percent_improvement

Exports:
    outputs/results.csv
    outputs/comparison_chart.png   (bar chart: network vs straight-line per incident)
    outputs/response_time_hist.png (histogram of network response times)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def run_analysis(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.network_builder import build_graph
    from src.routing import (find_closest_station, straight_line_time,
                              snap_stations_to_graph, snap_incidents_to_graph)

    print("=" * 60)
    print("STEP G: Batch Analysis — 20 Incidents")
    print("=" * 60)

    # Load and build
    G, nodes_gdf = build_graph()
    
    stations_gdf = gpd.read_file("data/ambulance_stations.gpkg").to_crs("EPSG:32631")
    stations_gdf = snap_stations_to_graph(stations_gdf, nodes_gdf)
    print(f"\nStations: {len(stations_gdf)} ({dict(stations_gdf['facility_type'].value_counts())})")
    print(stations_gdf[["name","facility_type","snap_dist_m"]].to_string())

    incidents_gdf = gpd.read_file("data/incident_points.gpkg").to_crs("EPSG:32631")
    incidents_gdf = snap_incidents_to_graph(incidents_gdf, nodes_gdf)
    incidents_gdf = incidents_gdf.reset_index(drop=True)
    print(f"\nIncidents: {len(incidents_gdf)} ({dict(incidents_gdf['type'].value_counts())})")

    # Batch routing
    records = []
    print(f"\n{'ID':<4} {'Type':<8} {'Station':<35} {'FacType':<8} {'NetTime':>8} {'SLTime':>8} {'Improv%':>8}")
    print("-" * 85)

    for i, inc_row in incidents_gdf.iterrows():
        inc_type = inc_row["type"]
        inc_node = inc_row["node_id"]
        inc_geom = inc_row.geometry

        result = find_closest_station(G, nodes_gdf, inc_node, inc_type, stations_gdf)

        if result is None:
            record = {
                "incident_id": i,
                "incident_type": inc_type,
                "nearest_station": "UNREACHABLE",
                "facility_type": "N/A",
                "network_time_min": float("nan"),
                "straight_line_time_min": float("nan"),
                "percent_improvement": float("nan"),
            }
            print(f"{i:<4} {inc_type:<8} {'UNREACHABLE':<35} {'N/A':<8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            # Get station geometry for straight-line calculation
            s_row = stations_gdf[stations_gdf["name"] == result["station_name"]].iloc[0]
            sl_time = straight_line_time(s_row.geometry, inc_geom, speed_kmh=30.0)
            net_time = result["network_time_min"]
            
            # Percent improvement = how much longer network route is vs straight line
            # (network is always >= straight line; this shows the overhead)
            if sl_time > 0:
                pct_diff = ((net_time - sl_time) / sl_time) * 100.0
            else:
                pct_diff = float("nan")

            record = {
                "incident_id": i,
                "incident_type": inc_type,
                "nearest_station": result["station_name"],
                "facility_type": result["facility_type"],
                "network_time_min": round(net_time, 3),
                "straight_line_time_min": round(sl_time, 3),
                "percent_improvement": round(pct_diff, 1),
            }
            print(f"{i:<4} {inc_type:<8} {result['station_name']:<35} {result['facility_type']:<8} "
                  f"{net_time:>8.2f} {sl_time:>8.2f} {pct_diff:>7.1f}%")

        records.append(record)

    results_df = pd.DataFrame(records)

    # Export CSV
    csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary stats
    valid = results_df.dropna(subset=["network_time_min"])
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total incidents processed: {len(results_df)}")
    print(f"Successfully routed: {len(valid)}")
    print(f"Mean network response time: {valid['network_time_min'].mean():.2f} min")
    print(f"Median network response time: {valid['network_time_min'].median():.2f} min")
    print(f"Min / Max: {valid['network_time_min'].min():.2f} / {valid['network_time_min'].max():.2f} min")
    print(f"Mean straight-line time: {valid['straight_line_time_min'].mean():.2f} min")
    print(f"Mean network overhead vs straight-line: {valid['percent_improvement'].mean():.1f}%")
    print(f"\nBy incident type:")
    for inc_type, grp in valid.groupby("incident_type"):
        print(f"  {inc_type}: n={len(grp)}, mean_net={grp['network_time_min'].mean():.2f} min, "
              f"mean_sl={grp['straight_line_time_min'].mean():.2f} min")

    # --- Chart 1: Comparison bar chart ---
    fig, ax = plt.subplots(figsize=(16, 7))
    
    valid_sorted = valid.sort_values("incident_id")
    x = np.arange(len(valid_sorted))
    width = 0.38
    
    type_colors = {"Medical": "#1565C0", "RTA": "#E65100", "Fire": "#B71C1C"}
    bar_colors = [type_colors.get(t, "#888888") for t in valid_sorted["incident_type"]]
    
    bars_net = ax.bar(x - width/2, valid_sorted["network_time_min"], width,
                       label="Network (Dijkstra) Time", color=bar_colors, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    bars_sl  = ax.bar(x + width/2, valid_sorted["straight_line_time_min"], width,
                       label="Straight-Line Baseline Time", color="#90A4AE", alpha=0.85,
                       edgecolor="black", linewidth=0.5, hatch="//")
    
    # Colour legend patches
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor="#1565C0", label="Medical incident (Network)"),
        mpatches.Patch(facecolor="#E65100", label="RTA incident (Network)"),
        mpatches.Patch(facecolor="#B71C1C", label="Fire incident (Network)"),
        mpatches.Patch(facecolor="#90A4AE", hatch="//", label="Straight-line baseline"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              frameon=True, facecolor="white", framealpha=0.9)
    
    ax.set_xlabel("Incident ID", fontsize=11)
    ax.set_ylabel("Response Time (minutes)", fontsize=11)
    ax.set_title(
        "Network-Optimised vs Straight-Line Response Times\n"
        "All 20 Simulated Incidents — Ikeja, Yaba & Surulere",
        fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{int(r)}" for r in valid_sorted["incident_id"]], fontsize=8, rotation=45)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False, nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(valid_sorted["network_time_min"].max(), valid_sorted["straight_line_time_min"].max()) * 1.2)
    
    # Add mean line
    mean_net = valid_sorted["network_time_min"].mean()
    ax.axhline(mean_net, color="#0D47A1", linestyle="--", linewidth=1.5,
               label=f"Mean network time: {mean_net:.2f} min")
    ax.text(len(x)-0.5, mean_net + 0.1, f"Mean: {mean_net:.2f} min",
            color="#0D47A1", fontsize=9, ha="right", va="bottom")
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart_path}")
    
    # --- Chart 2: Histogram of network response times ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(0, valid["network_time_min"].max() + 2, 1.5)
    
    # Stacked histogram by type
    data_by_type = {}
    for t in ["Medical", "RTA", "Fire"]:
        subset = valid[valid["incident_type"] == t]["network_time_min"]
        if len(subset) > 0:
            data_by_type[t] = subset.values
    
    colors_hist = {"Medical": "#1565C0", "RTA": "#E65100", "Fire": "#B71C1C"}
    bottom = np.zeros(len(bins)-1)
    
    for inc_type, vals in data_by_type.items():
        counts, _ = np.histogram(vals, bins=bins)
        ax.bar(bins[:-1] + 0.75, counts, width=1.5, bottom=bottom,
               color=colors_hist[inc_type], alpha=0.8, label=f"{inc_type} incidents",
               edgecolor="black", linewidth=0.5)
        bottom += counts
    
    # Threshold lines
    ax.axvline(5,  color="#4CAF50", linestyle="--", linewidth=2, alpha=0.8, label="5 min target")
    ax.axvline(10, color="#FF9800", linestyle="--", linewidth=2, alpha=0.8, label="10 min threshold")
    ax.axvline(15, color="#F44336", linestyle="--", linewidth=2, alpha=0.8, label="15 min critical")
    
    ax.set_xlabel("Network Response Time (minutes)", fontsize=11)
    ax.set_ylabel("Number of Incidents", fontsize=11)
    ax.set_title(
        "Distribution of Network Response Times\n"
        "Dijkstra Optimal Routing — All 20 Incidents",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, frameon=True, facecolor="white")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "response_time_hist.png")
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {hist_path}")
    
    print("\nStep G COMPLETE.")
    return results_df


if __name__ == "__main__":
    run_analysis()
