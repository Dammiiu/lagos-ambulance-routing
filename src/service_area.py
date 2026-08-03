"""
src/service_area.py

Service Area Analysis Model (Step F):

SA_t(s) = {v in V | d(s,v) <= t}

Computes reachable nodes within t minutes from each station.
Runs for t = 5, 10, 15 minutes.
Exports coverage map to outputs/service_area_map.png.

Public API:
    compute_service_area(G, nodes_gdf, station_node, t_minutes) -> set of node IDs
    compute_all_service_areas(G, nodes_gdf, stations_gdf, t_minutes_list) -> dict
    plot_service_area_map(G, nodes_gdf, edges_gdf, stations_gdf, incidents_gdf, output_path)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union


# ── Core model ─────────────────────────────────────────────────────────────────

def compute_service_area(G: nx.DiGraph, nodes_gdf: gpd.GeoDataFrame, 
                          station_node, t_minutes: float) -> set:
    """
    SA_t(s) = {v in V | d(s,v) <= t}
    
    Uses Dijkstra single-source shortest paths from station_node,
    returns the set of node IDs reachable within t_minutes.
    """
    lengths = nx.single_source_dijkstra_path_length(G, station_node, weight="time_min")
    reachable = {node for node, dist in lengths.items() if dist <= t_minutes}
    return reachable


def compute_all_service_areas(G: nx.DiGraph, nodes_gdf: gpd.GeoDataFrame,
                               stations_gdf: gpd.GeoDataFrame,
                               t_minutes_list: list = None):
    """
    Compute service areas for all stations at each time threshold.
    
    Returns
    -------
    dict: {
        t: {
            "medical": set of reachable node IDs (union over all medical stations),
            "fire":    set of reachable node IDs (union over all fire stations),
            "per_station": {station_name: set of node IDs}
        }
    }
    """
    if t_minutes_list is None:
        t_minutes_list = [5, 10, 15]
    
    results = {}
    for t in t_minutes_list:
        print(f"  Computing service areas for t={t} min...")
        per_station = {}
        medical_union = set()
        fire_union = set()
        
        for _, srow in stations_gdf.iterrows():
            s_node = srow.get("node_id")
            if s_node is None or s_node not in G.nodes:
                continue
            
            sa = compute_service_area(G, nodes_gdf, s_node, t)
            sname = srow.get("name", str(s_node))
            per_station[sname] = sa
            
            if srow.get("facility_type") == "medical":
                medical_union |= sa
            elif srow.get("facility_type") == "fire":
                fire_union |= sa
        
        results[t] = {
            "medical": medical_union,
            "fire":    fire_union,
            "per_station": per_station,
        }
        
        total_nodes = len(G.nodes)
        print(f"    t={t}: medical covers {len(medical_union)}/{total_nodes} nodes ({100*len(medical_union)/total_nodes:.1f}%)")
        print(f"    t={t}: fire    covers {len(fire_union)}/{total_nodes} nodes ({100*len(fire_union)/total_nodes:.1f}%)")
    
    return results


def nodes_to_polygon(nodes_gdf: gpd.GeoDataFrame, node_set: set, buffer_m: float = 150):
    """
    Convert a set of node IDs to a polygon (buffered union of node points).
    Used for visualizing coverage zones.
    """
    valid_nodes = [nid for nid in node_set if nid in nodes_gdf.index]
    if not valid_nodes:
        return None
    
    points = nodes_gdf.loc[valid_nodes].geometry
    buffered = points.buffer(buffer_m)
    polygon = unary_union(buffered)
    return polygon


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_service_area_map(G, nodes_gdf, edges_gdf, stations_gdf, incidents_gdf,
                           t_minutes_list=None, output_path="outputs/service_area_map.png"):
    """
    Generate color-coded service area coverage map.
    
    Shows:
    - Road network (light grey)
    - Coverage zones at t=5,10,15 min (nested polygons, lightest=15)
    - Stations (circles, blue=medical, red=fire)
    - Incidents (star markers, colored by type)
    """
    if t_minutes_list is None:
        t_minutes_list = [5, 10, 15]
    
    sa_results = compute_all_service_areas(G, nodes_gdf, stations_gdf, t_minutes_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle(
        "Ambulance Service Area Coverage — Ikeja, Yaba & Surulere\n"
        "Service Area SA$_t$(s) = {v ∈ V | d(s,v) ≤ t}",
        fontsize=15, fontweight="bold", y=0.98
    )
    
    # Colour schemes
    medical_colors = {5: "#0a4c9e", 10: "#3a7bc8", 15: "#a8c8f0"}
    fire_colors    = {5: "#8b0000", 10: "#cc2222", 15: "#ffaaaa"}
    type_titles    = ["Medical/RTA Coverage", "Fire Coverage"]
    fac_types      = ["medical", "fire"]
    color_schemes  = [medical_colors, fire_colors]
    
    for ax_idx, (ax, fac_type, colors, title) in enumerate(
        zip(axes, fac_types, color_schemes, type_titles)
    ):
        # Road network
        edges_gdf.plot(ax=ax, color="#e0e0e0", linewidth=0.5, alpha=0.8, zorder=1)
        
        # Coverage zones — plot largest first (most transparent)
        for t in sorted(t_minutes_list, reverse=True):
            node_set = sa_results[t][fac_type]
            poly = nodes_to_polygon(nodes_gdf, node_set, buffer_m=200)
            if poly is not None:
                gdf_poly = gpd.GeoDataFrame({"geometry": [poly]}, crs=nodes_gdf.crs)
                gdf_poly.plot(ax=ax, color=colors[t], alpha=0.25, zorder=2)
        
        # Draw coverage zone outlines
        for t in sorted(t_minutes_list):
            node_set = sa_results[t][fac_type]
            poly = nodes_to_polygon(nodes_gdf, node_set, buffer_m=200)
            if poly is not None:
                gdf_poly = gpd.GeoDataFrame({"geometry": [poly]}, crs=nodes_gdf.crs)
                gdf_poly.boundary.plot(ax=ax, color=colors[t], linewidth=1.5, 
                                       alpha=0.7, zorder=3, label=f"t={t} min")
        
        # Stations
        med_stations = stations_gdf[stations_gdf["facility_type"] == "medical"]
        fire_stations = stations_gdf[stations_gdf["facility_type"] == "fire"]
        if len(med_stations) > 0:
            med_stations.plot(ax=ax, color="#1565C0", marker="o", markersize=120,
                              edgecolor="white", linewidth=1.5, zorder=6)
        if len(fire_stations) > 0:
            fire_stations.plot(ax=ax, color="#B71C1C", marker="^", markersize=120,
                               edgecolor="white", linewidth=1.5, zorder=6)
        
        # Incidents
        inc_colors = {"Medical": "#2196F3", "RTA": "#FF9800", "Fire": "#F44336"}
        for inc_type, inc_color in inc_colors.items():
            subset = incidents_gdf[incidents_gdf["type"] == inc_type]
            if len(subset) > 0:
                subset.plot(ax=ax, color=inc_color, marker="*", markersize=80,
                            edgecolor="black", linewidth=0.5, zorder=7)
        
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("UTM Easting (m)", fontsize=9)
        ax.set_ylabel("UTM Northing (m)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Legend
        legend_handles = []
        for t in sorted(t_minutes_list):
            patch = mpatches.Patch(facecolor=colors[t], alpha=0.5, 
                                   edgecolor=colors[t], label=f"Coverage ≤{t} min")
            legend_handles.append(patch)
        
        import matplotlib.lines as mlines
        legend_handles += [
            mlines.Line2D([], [], color="#1565C0", marker="o", linestyle="None",
                          markersize=10, markeredgecolor="white", label="Medical Station"),
            mlines.Line2D([], [], color="#B71C1C", marker="^", linestyle="None",
                          markersize=10, markeredgecolor="white", label="Fire Station"),
            mlines.Line2D([], [], color="#2196F3", marker="*", linestyle="None",
                          markersize=10, markeredgecolor="k", label="Medical Incident"),
            mlines.Line2D([], [], color="#FF9800", marker="*", linestyle="None",
                          markersize=10, markeredgecolor="k", label="RTA Incident"),
            mlines.Line2D([], [], color="#F44336", marker="*", linestyle="None",
                          markersize=10, markeredgecolor="k", label="Fire Incident"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
                  frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[service_area] Saved: {output_path}")
    return sa_results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.network_builder import build_graph
    from src.routing import snap_stations_to_graph, snap_incidents_to_graph
    
    print("=" * 60)
    print("STEP F: Service Area Analysis")
    print("=" * 60)
    
    G, nodes_gdf = build_graph()
    edges_gdf = gpd.read_file("data/road_network_final.gpkg")
    
    stations_gdf = gpd.read_file("data/ambulance_stations.gpkg")
    stations_gdf = stations_gdf.to_crs("EPSG:32631")
    stations_gdf = snap_stations_to_graph(stations_gdf, nodes_gdf)
    
    incidents_gdf = gpd.read_file("data/incident_points.gpkg")
    incidents_gdf = incidents_gdf.to_crs("EPSG:32631")
    incidents_gdf = snap_incidents_to_graph(incidents_gdf, nodes_gdf)
    
    print("\nComputing service areas for t = 5, 10, 15 minutes...")
    os.makedirs("outputs", exist_ok=True)
    sa_results = plot_service_area_map(
        G, nodes_gdf, edges_gdf, stations_gdf, incidents_gdf,
        t_minutes_list=[5, 10, 15],
        output_path="outputs/service_area_map.png"
    )
    print("\nStep F COMPLETE.")
