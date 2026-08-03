# GIS-Based Optimal Ambulance Routing and Emergency Response Time Analysis

This project implements a GIS-based system for optimal ambulance routing and emergency response time analysis using Dijkstra's Algorithm and Network Analysis.

## Project Structure
- `data/`: Input GeoPackage files.
- `src/`: Source code for network construction, routing, service area, and analysis.
- `notebooks/`: Jupyter notebooks for exploration.
- `outputs/`: Maps, metrics, and comparison results.
- `tests/`: Unit tests for verification.

## Methodology
1. **Route Optimization**: Dijkstra's Algorithm for shortest travel-time.
2. **Closest Facility Analysis**: Identifying the optimal station based on actual network travel time.
3. **Service Area Analysis**: Computing coverage polygons for 5, 10, and 15-minute thresholds.
4. **Comparison Model**: Evaluating network-optimized response time against straight-line distance estimates.
