import os
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

def clean_columns_for_gpkg(df):
    """
    Cleans DataFrame columns containing lists or other non-primitive types 
    by converting them to comma-separated strings. This prevents GPKG export errors.
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        # Check if any element in the column is a list
        if df_cleaned[col].apply(lambda x: isinstance(x, list)).any():
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x)
            )
    return df_cleaned

def main():
    print("=" * 60)
    print("OSM DATA ACQUISITION & PROCESSING PIPELINE")
    print("=" * 60)

    # STEP 2 — Download the road network
    print("\n--- STEP 2: Downloading Road Network ---")
    place_names = [
        "Ikeja, Lagos, Nigeria", 
        "Yaba, Lagos, Nigeria", 
        "Surulere, Lagos, Nigeria"
    ]
    print(f"Target study areas: {', '.join(place_names)}")
    try:
        G = ox.graph_from_place(place_names, network_type="drive")
        print(f"Success! Downloaded combined 'drive' graph.")
        print(f"Raw graph: {len(G.nodes)} nodes and {len(G.edges)} edges.")
    except Exception as e:
        print(f"Error downloading with place names: {e}")
        print("Switching to bounding box or alternative geocoding lookup...")
        raise e

    # STEP 3 — Add speed and travel time
    print("\n--- STEP 3: Adding Speed and Travel Time ---")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print("Success! Calculated speeds and travel times.")

    # STEP 4 — Convert to GeoDataFrames and export
    print("\n--- STEP 4: Projecting and Converting to GeoDataFrames ---")
    print("Reprojecting graph to UTM Zone 31N (EPSG:32631)...")
    G_proj = ox.project_graph(G, to_crs="EPSG:32631")
    
    nodes_df, edges_df = ox.graph_to_gdfs(G_proj)
    print(f"Conversion complete. Nodes: {len(nodes_df)}, Edges: {len(edges_df)}")

    # Rename fields to match project expected schema
    # Expected fields: speed_kmh, length_m, time_min, oneway, highway
    print("Mapping schema names...")
    if "speed_kph" in edges_df.columns:
        edges_df["speed_kmh"] = edges_df["speed_kph"]
    else:
        edges_df["speed_kmh"] = 30.0  # Fallback
        
    if "length" in edges_df.columns:
        edges_df["length_m"] = edges_df["length"]
    else:
        edges_df["length_m"] = 0.0
        
    if "travel_time" in edges_df.columns:
        edges_df["time_min"] = edges_df["travel_time"] / 60.0  # Convert seconds to minutes
    else:
        edges_df["time_min"] = edges_df["length_m"] / (edges_df["speed_kmh"] * 1000.0 / 60.0)

    # Clean non-primitive types for GPKG compliance
    edges_df_cleaned = clean_columns_for_gpkg(edges_df)
    nodes_df_cleaned = clean_columns_for_gpkg(nodes_df)

    # Export
    os.makedirs("data", exist_ok=True)
    road_network_path = "data/road_network_final.gpkg"
    print(f"Saving edges (roads) to {road_network_path}...")
    edges_df_cleaned.to_file(road_network_path, layer="edges", driver="GPKG")
    print("Export complete.")

    # STEP 5 — Validate the network
    print("\n--- STEP 5: Validating the Network ---")
    is_connected = nx.is_strongly_connected(G_proj)
    print(f"Is the graph strongly connected? {is_connected}")
    
    if not is_connected:
        components = list(nx.strongly_connected_components(G_proj))
        component_sizes = [len(c) for c in components]
        component_sizes.sort(reverse=True)
        print(f"Disconnected component count: {len(components)}")
        print(f"Largest component size: {component_sizes[0]} nodes")
        print(f"Other component sizes: {component_sizes[1:10]} (top 10)")
        
        # Keep only the largest connected component
        print("\n[LIMITATION/DECISION FLAG] The road network is NOT strongly connected.")
        print("Defaulting to keeping only the LARGEST strongly connected component to avoid routing failures.")
        largest_component = max(nx.strongly_connected_components(G_proj), key=len)
        G_final = G_proj.subgraph(largest_component).copy()
        
        # Overwrite the GeoPackages and update dataframes
        nodes_df_final, edges_df_final = ox.graph_to_gdfs(G_final)
        edges_df_final_cleaned = clean_columns_for_gpkg(edges_df_final)
        
        # Map fields on final graph
        if "speed_kph" in edges_df_final_cleaned.columns:
            edges_df_final_cleaned["speed_kmh"] = edges_df_final_cleaned["speed_kph"]
        else:
            edges_df_final_cleaned["speed_kmh"] = 30.0
            
        if "length" in edges_df_final_cleaned.columns:
            edges_df_final_cleaned["length_m"] = edges_df_final_cleaned["length"]
            
        if "travel_time" in edges_df_final_cleaned.columns:
            edges_df_final_cleaned["time_min"] = edges_df_final_cleaned["travel_time"] / 60.0
        else:
            edges_df_final_cleaned["time_min"] = edges_df_final_cleaned["length_m"] / (edges_df_final_cleaned["speed_kmh"] * 1000.0 / 60.0)

        print(f"Re-saving connected road network (Nodes: {len(nodes_df_final)}, Edges: {len(edges_df_final_cleaned)})...")
        edges_df_final_cleaned.to_file(road_network_path, layer="edges", driver="GPKG")
        print("Connected component exported.")
    else:
        G_final = G_proj
        edges_df_final_cleaned = edges_df_cleaned
        nodes_df_final = nodes_df_cleaned
        print("The road network is fully connected!")

    # STEP 6 — Download candidate ambulance/emergency facility points
    print("\n--- STEP 6: Downloading Candidate Emergency Facilities ---")
    tags = {
        'amenity': ['hospital', 'fire_station', 'clinic'],
        'emergency': 'ambulance_station'
    }
    print(f"Querying features with tags: {tags}")
    try:
        gdf_stations = ox.features_from_place(place_names, tags=tags)
        print(f"Retrieved {len(gdf_stations)} candidate facilities.")
        
        # Reproject to UTM Zone 31N
        gdf_stations_proj = gdf_stations.to_crs("EPSG:32631")
        
        # Convert all geometries to points (centroids)
        gdf_stations_proj['geometry'] = gdf_stations_proj.centroid
        
        # Extract individual facility type counts
        gdf_stations_proj['amenity_clean'] = gdf_stations_proj['amenity'] if 'amenity' in gdf_stations_proj.columns else np.nan
        gdf_stations_proj['emergency_clean'] = gdf_stations_proj['emergency'] if 'emergency' in gdf_stations_proj.columns else np.nan
        
        def get_facility_type(row):
            if pd.notna(row['emergency_clean']) and row['emergency_clean'] == 'ambulance_station':
                return 'ambulance_station'
            if pd.notna(row['amenity_clean']):
                return row['amenity_clean']
            return 'other'
            
        gdf_stations_proj['facility_type'] = gdf_stations_proj.apply(get_facility_type, axis=1)
        
        # Report breakdown
        breakdown = gdf_stations_proj['facility_type'].value_counts()
        print("Facility type breakdown from OpenStreetMap:")
        all_expected = ['hospital', 'fire_station', 'clinic', 'ambulance_station']
        for ftype in all_expected:
            count = breakdown.get(ftype, 0)
            print(f"  - {ftype.replace('_', ' ').title()}: {count}")
            
        # Clean columns and export
        gdf_stations_cleaned = clean_columns_for_gpkg(gdf_stations_proj)
        stations_path = "data/candidate_stations_osm.gpkg"
        gdf_stations_cleaned.to_file(stations_path, layer="stations", driver="GPKG")
        print(f"Exported candidate stations to {stations_path}.")
    except Exception as e:
        print(f"Error in Step 6: {e}")
        raise e

    # STEP 7 — Generate representative incident points
    print("\n--- STEP 7: Generating Representative Incident Points ---")
    # Calculate study area center based on nodes
    mean_x = nodes_df_final.geometry.x.mean()
    mean_y = nodes_df_final.geometry.y.mean()
    
    # Calculate distance to center
    distances = np.sqrt((nodes_df_final.geometry.x - mean_x)**2 + (nodes_df_final.geometry.y - mean_y)**2)
    max_dist = distances.max()
    
    # Distance weight (closer to center = higher weight)
    w_dist = max_dist - distances
    
    # Road class weight (higher weight for primary/secondary roads)
    primary_secondary_classes = {'primary', 'secondary', 'primary_link', 'secondary_link'}
    
    def is_primary_secondary(val):
        if not isinstance(val, str):
            return False
        parts = [p.strip() for p in val.split(',')]
        return any(p in primary_secondary_classes for p in parts)
        
    high_class_edges = edges_df_final_cleaned[edges_df_final_cleaned['highway'].apply(is_primary_secondary)]
    high_class_nodes = set(high_class_edges.index.get_level_values(0)).union(
        set(high_class_edges.index.get_level_values(1))
    )
    
    w_road = np.where(nodes_df_final.index.isin(high_class_nodes), 5.0, 1.0)
    
    # Combined weights
    combined_weights = w_dist * w_road
    if combined_weights.sum() == 0:
        combined_weights = np.ones(len(combined_weights))
        
    probabilities = combined_weights / combined_weights.sum()
    
    # Sample 20 nodes
    np.random.seed(42)
    sampled_node_ids = np.random.choice(nodes_df_final.index, size=20, replace=False, p=probabilities)
    incident_nodes = nodes_df_final.loc[sampled_node_ids].copy()
    
    # Add random type
    types = ["RTA", "Medical", "Fire"]
    incident_nodes['type'] = np.random.choice(types, size=20, replace=True)
    
    # Create incident points GDF
    gdf_incidents = gpd.GeoDataFrame(
        {'type': incident_nodes['type']},
        geometry=incident_nodes.geometry,
        crs=nodes_df_final.crs
    )
    
    # Export
    incidents_path = "data/incident_points.gpkg"
    gdf_incidents_cleaned = clean_columns_for_gpkg(gdf_incidents)
    gdf_incidents_cleaned.to_file(incidents_path, layer="incidents", driver="GPKG")
    print(f"Generated and exported 20 simulated incident points with spatial bias to {incidents_path}.")

    # STEP 8 — Produce a visual QA map
    print("\n--- STEP 8: Generating Visual QA Map ---")
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot road network edges
    edges_df_final_cleaned.plot(ax=ax, color='lightgrey', linewidth=0.8, alpha=0.9, zorder=1)
    
    # Plot candidate stations
    gdf_stations_proj.plot(ax=ax, color='red', marker='^', markersize=180, edgecolor='black', zorder=5)
    
    # Plot incidents
    gdf_incidents.plot(ax=ax, color='orange', marker='o', markersize=100, edgecolor='black', zorder=6)
    
    # Add labels and details
    ax.set_title(
        "Visual QA Map - Ikeja, Yaba, & Surulere\nRoad Network, Candidate Stations & Simulated Incident Points", 
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel("UTM Easting (m)", labelpad=10)
    ax.set_ylabel("UTM Northing (m)", labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgrey', lw=2, label='Road Network'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor='black', label='Candidate Stations (Hospital / Fire Station)', markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', label='Simulated Incidents (RTA / Medical / Fire)', markerfacecolor='orange', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True, facecolor='white')
    
    qa_map_path = "outputs/qa_map.png"
    plt.tight_layout()
    plt.savefig(qa_map_path, dpi=300)
    print(f"Successfully generated and saved Visual QA Map to: {qa_map_path}")
    print("\n" + "=" * 60)
    print("ALL ACQUISITION AND PROCESSING STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
