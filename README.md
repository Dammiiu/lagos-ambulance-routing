# GIS-Based Optimal Ambulance Routing and Emergency Response Time Analysis

Final year project for B.Sc. Surveying and Geoinformatics — Abiola Ajimobi Technical University, Ibadan.

This system optimises ambulance routing and analyses emergency response coverage for Lagos road networks using Dijkstra's Algorithm and network analysis.

## Project Structure
- `data/` — Input GeoPackage files (road network, stations, incidents)
- `src/` — Application source code
  - `app.py` — Main Streamlit web application
  - `network_builder.py` — Road network graph construction
  - `routing.py` — Dijkstra routing and closest facility logic
  - `service_area.py` — Service area computation (5, 10, 15 min zones)
  - `navigation.py` — Route simulation and turn-by-turn directions
  - `analysis.py` — Batch analysis and comparison model

## Key Features
1. **Route Optimisation** — Shortest travel-time routing via Dijkstra's Algorithm
2. **Closest Facility** — Automatic station selection based on network travel time
3. **Service Area Coverage** — 5, 10, and 15-minute coverage zone mapping
4. **Two-Leg Routing** — Station → Incident → Hospital
5. **Siren Mode** — One-way road override for emergency vehicles
6. **Live Vehicle Tracking** — Animated ambulance simulation along routes

## Running
```bash
pip install -r requirements.txt
streamlit run src/app.py
```
