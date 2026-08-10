"""
src/app.py — GIS-Based Optimal Ambulance Routing & Emergency Response Time Analysis
Makanjuola, Thomas Oluwadamilare (125/21/1/0180)
Abiola Ajimobi Technical University, Ibadan — 2026

Features:
  ✓ Two-leg routing: dispatch → incident (Leg 1) + incident → hospital (Leg 2, Medical/RTA)
  ✓ Configurable vehicle start — station / browser geolocation / map-click / manual coords
  ✓ Turn-by-turn directions (src/navigation.py) for both legs
  ✓ Journey Simulation: manual scrubber + auto-advance (st_autorefresh)
  ✓ Live Navigation Mode: real device GPS polling, off-route detection & re-routing
  ✓ Navigation-quality route rendering: thick halo lines + directional arrows
  ✓ Real road geometry (LineString curves)
  ✓ Map controls: Fullscreen, Measure, Recenter, Live Compass overlay
  ✓ Voice turn announcements (Web Speech API)
  ✓ 3D Pydeck tab (CARTO_DARK, no Mapbox token needed)
  ✓ Double-cache loading (pickle) — sub-second startup
  ✓ Lagos congestion factor 2.2 on network weights
  ✓ 5 / 10 / 15 min coverage zones
  ✓ Sidebar inputs fully readable (dark backgrounds, white text)
  ✓ Responsive layout (mobile / tablet / laptop)

Run: streamlit run src/app.py
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, math, pickle
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import streamlit as st
import streamlit.components.v1 as components
# The geocoding function for Priority 2
global geocode_lagos
@st.cache_data(ttl=3600, show_spinner=False)
def geocode_lagos(query):
    time.sleep(1.1)  # Respect Nominatim limits for uncached queries
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "viewbox": "3.0,6.3,3.7,6.8",
        "bounded": 1,
        "limit": 5
    }
    headers = {"User-Agent": "AATU-Ambulance-Routing-App/1.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import time
from st_keyup import st_keyup
from shapely.geometry import Point
from pyproj import Transformer
from datetime import datetime
import folium
from folium import plugins
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
import pydeck as pdk

def create_vehicle_arrow(lat, lon, bearing, size_deg=0.00035):
    """Generates a dynamic 3D directional chevron polygon for the vehicle marker"""
    ang = math.radians(-bearing + 90)
    # Front tip
    f_lat = lat + size_deg * math.sin(ang)
    f_lon = lon + size_deg * math.cos(ang)
    # Back left
    bl_ang = ang + 2.45
    bl_lat = lat + (size_deg * 0.75) * math.sin(bl_ang)
    bl_lon = lon + (size_deg * 0.75) * math.cos(bl_ang)
    # Center indent
    c_lat = lat + (size_deg * 0.25) * math.sin(ang + math.pi)
    c_lon = lon + (size_deg * 0.25) * math.cos(ang + math.pi)
    # Back right
    br_ang = ang - 2.45
    br_lat = lat + (size_deg * 0.75) * math.sin(br_ang)
    br_lon = lon + (size_deg * 0.75) * math.cos(br_ang)
    return [[f_lon, f_lat], [bl_lon, bl_lat], [c_lon, c_lat], [br_lon, br_lat], [f_lon, f_lat]]

# ── Load project modules (bypasses stale pycache) ─────────────────────────────
import importlib.util as _ilu

def _load(name, relpath):
    _p   = os.path.join(_ROOT, relpath)
    _s   = _ilu.spec_from_file_location(name, _p)
    _m   = _ilu.module_from_spec(_s)
    sys.modules[name] = _m
    _s.loader.exec_module(_m)
    return _m

_nb  = _load("src.network_builder", "src/network_builder.py")
_rt  = _load("src.routing",         "src/routing.py")
_nav = _load("src.navigation",      "src/navigation.py")
_sa  = _load("src.service_area",    "src/service_area.py")

build_graph              = _nb.build_graph
snap_point_to_node       = _nb.snap_point_to_node
path_to_detailed_coords  = _nb.path_to_detailed_coords

two_leg_route            = _rt.two_leg_route
shortest_path            = _rt.shortest_path
straight_line_time       = _rt.straight_line_time
snap_stations_to_graph   = _rt.snap_stations_to_graph
snap_incidents_to_graph  = _rt.snap_incidents_to_graph
INCIDENT_TO_FACILITY     = _rt.INCIDENT_TO_FACILITY

bearing_deg              = _nav.bearing_deg
cardinal_direction       = _nav.cardinal_direction
generate_directions      = _nav.generate_directions
format_directions_html   = _nav.format_directions_html
route_total_distance     = _nav.route_total_distance
interpolate_route        = _nav.interpolate_route
utm_to_ll_list           = _nav.utm_to_ll_list

compute_all_service_areas = _sa.compute_all_service_areas
nodes_to_polygon          = _sa.nodes_to_polygon

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GIS Ambulance Routing | Lagos State",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Priority 1 fix: readable sidebar inputs + dark dropdown popover
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0b1120; }
.main .block-container { padding-top:0.8rem; max-width:100%; padding-left:1rem; padding-right:1rem; }

/* ── Banner ── */
.top-banner {
    background: linear-gradient(135deg, #070f1e 0%, #0d1e36 50%, #153258 100%);
    border: 1.5px solid rgba(59, 130, 246, 0.4);
    border-radius: 14px;
    padding: 1.2rem 2rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}
.top-banner .logo { font-size: 2.5rem; filter: drop-shadow(0 2px 8px rgba(59,130,246,0.4)); }
.top-banner h1 { color: #fff; font-size: 1.4rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
.top-banner p  { color: #94a3b8; font-size: 0.78rem; margin: 0.2rem 0 0; }
.top-banner .uni {
    margin-left: auto;
    text-align: right;
    color: #e2e8f0;
    font-size: 0.8rem;
    line-height: 1.5;
    white-space: nowrap;
    border-left: 3px solid #3b82f6;
    padding-left: 1.2rem;
}

/* ── KPI cards ── */
.kpi-row { display:flex; gap:0.55rem; margin-bottom:0.7rem; }
.kpi-card { flex:1;border-radius:10px;padding:0.7rem 0.85rem;
    background:linear-gradient(135deg,#0d2547,#142f5e);
    border:1px solid rgba(66,153,225,0.18);box-shadow:0 4px 14px rgba(0,0,0,0.28); }
.kpi-card.green  { background:linear-gradient(135deg,#0a2e14,#0f4a22);border-color:rgba(72,199,116,0.28); }
.kpi-card.amber  { background:linear-gradient(135deg,#2e1a00,#4a2d00);border-color:rgba(246,173,85,0.28); }
.kpi-card.red    { background:linear-gradient(135deg,#2e0a0a,#4a1010);border-color:rgba(245,101,101,0.28); }
.kpi-card.purple { background:linear-gradient(135deg,#1a0a2e,#2d1050);border-color:rgba(159,122,234,0.28); }
.kpi-val   { font-size:1.45rem;font-weight:800;color:#fff; }
.kpi-label { font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.48);margin-top:0.1rem; }

/* ── Result panel ── */
.result-panel { background:linear-gradient(180deg,#0a1828,#0d1f38);
    border:1px solid rgba(66,153,225,0.18);border-radius:12px;
    padding:1rem;margin-bottom:0.55rem;box-shadow:0 6px 22px rgba(0,0,0,0.38); }
.result-panel h4 { color:#63b3ed;font-size:0.68rem;text-transform:uppercase;
    letter-spacing:0.1em;margin:0 0 0.55rem; }

/* ── Leg cards ── */
.leg-card { border-radius:10px;padding:0.75rem 0.9rem;margin-bottom:0.45rem; }
.leg-card.leg1  { background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.22); }
.leg-card.leg2  { background:rgba(251,146,60,0.07);border:1px solid rgba(251,146,60,0.22); }
.leg-card.fire  { background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.18); }
.leg-title  { font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em; }
.leg-title.blue   { color:#56cff8; }
.leg-title.orange { color:#fb923c; }
.leg-title.red    { color:#f87171; }
.leg-title.purple { color:#a78bfa; }
.leg-name { font-size:0.86rem;font-weight:600;color:#e2e8f0;margin:0.18rem 0 0.08rem; }
.leg-time { font-size:1.2rem;font-weight:800;color:#fff; }
.leg-unit { font-size:0.6rem;color:rgba(255,255,255,0.38);margin-left:0.12rem; }

/* ── Badges ── */
.badge { display:inline-block;padding:0.16rem 0.5rem;border-radius:20px;
    font-size:0.66rem;font-weight:700;margin-right:0.28rem; }
.badge-medical { background:#1e40af;color:#bfdbfe; }
.badge-rta     { background:#92400e;color:#fde68a; }
.badge-fire    { background:#7f1d1d;color:#fecaca; }
.badge-geo     { background:#3b0764;color:#e9d5ff; }
.badge-live    { background:#064e3b;color:#6ee7b7; }

/* ── Misc ── */
.info-box { background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.07);
    border-radius:8px;padding:0.6rem 0.82rem;margin:0.3rem 0;font-size:0.76rem;color:#cbd5e1; }
.rating-pill { display:inline-block;padding:0.22rem 0.65rem;border-radius:20px;font-size:0.73rem;font-weight:700; }
.rating-excellent { background:rgba(72,199,116,0.14);color:#48c774;border:1px solid rgba(72,199,116,0.28); }
.rating-good      { background:rgba(246,173,85,0.14); color:#f6ad55;border:1px solid rgba(246,173,85,0.28); }
.rating-moderate  { background:rgba(237,137,54,0.14); color:#ed8936;border:1px solid rgba(237,137,54,0.28); }
.rating-critical  { background:rgba(245,101,101,0.14);color:#f56565;border:1px solid rgba(245,101,101,0.28); }
.dir-box { background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
    border-radius:10px;padding:0.65rem;max-height:280px;overflow-y:auto; }
.sim-panel { background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.18);
    border-radius:10px;padding:0.8rem;margin-top:0.45rem; }
.live-panel { background:rgba(6,78,59,0.12);border:1px solid rgba(52,211,153,0.25);
    border-radius:10px;padding:0.8rem;margin-top:0.45rem; }
.sim-stat  { text-align:center;padding:0.35rem; }
.sim-val   { font-size:1.15rem;font-weight:800;color:#a78bfa; }
.live-val  { font-size:1.15rem;font-weight:800;color:#34d399; }
.sim-label { font-size:0.6rem;color:rgba(255,255,255,0.42);text-transform:uppercase;letter-spacing:0.07em; }
.click-info { background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);
    border-radius:8px;padding:0.5rem 0.7rem;font-size:0.76rem;color:#6ee7b7;margin-top:0.3rem; }
.geo-denied { background:rgba(185,28,28,0.08);border:1px solid rgba(185,28,28,0.25);
    border-radius:8px;padding:0.55rem 0.8rem;font-size:0.74rem;color:#f87171;margin-top:0.3rem; }
.dispatch-box {
    background: linear-gradient(135deg, rgba(13,31,56,0.95), rgba(6,20,38,0.95));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
}

/* ── Sidebar stats cards ── */
.stat-group {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.stat-header {
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 0.8rem;
}
.bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    width: 100%;
    margin-top: 4px;
    margin-bottom: 12px;
}
.med-bar {
    background: #3b82f6;
    height: 100%;
}
.fire-bar {
    background: #ef4444;
    height: 100%;
}
.badge-value {
    font-weight: 700;
    color: #f8fafc;
}

/* ── Mobile & Responsive Design (consolidated) ── */
@media (max-width: 768px) {
    .main .block-container { padding: 0.4rem 0.3rem; }
    .top-banner { flex-direction: column; text-align: center; gap: 0.5rem; padding: 0.8rem 0.6rem; }
    .top-banner h1 { font-size: 1rem; }
    .top-banner p  { font-size: 0.68rem; }
    .top-banner .logo { font-size: 1.8rem; }
    .top-banner .uni { text-align: center; margin-left: 0; border-left: none; padding-left: 0;
        border-top: 2px solid #3b82f6; padding-top: 0.5rem; white-space: normal; font-size: 0.72rem; }
    .kpi-row { flex-direction: column; gap: 0.35rem; }
    .kpi-card { width: 100% !important; margin-bottom: 0; }
    .kpi-val { font-size: 1.15rem; }
    .result-panel { padding: 0.65rem; margin-bottom: 0.35rem; }
    .leg-card { padding: 0.55rem 0.65rem; }
    .dir-box { max-height: 180px; font-size: 0.72rem; }
    .sim-stat { padding: 6px !important; }
    .sim-val, .live-val { font-size: 1rem; }
    /* Increase touch targets for buttons */
    .stButton > button { padding: 0.65rem 0.8rem !important; font-size: 0.9rem !important;
        min-height: 44px !important; }
    /* Fix Streamlit columns on mobile */
    div[data-testid="column"] { width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; }
    /* Sidebar full width on mobile */
    div[data-testid="stSidebar"] { min-width: 100vw !important; max-width: 100vw !important; }
    /* Map containers */
    div[data-testid="stDeckGlJsonChart"] { height: 65vh !important; }
    iframe { max-height: 65vh !important; }
    .pydeck-legend-container { display: none !important; }
}

/* ── Prevent Streamlit rerun indicators & stale element dimming during tracking ── */
[data-testid="stStatusWidget"] { visibility: hidden !important; height: 0 !important; position: absolute !important; }
[data-testid="stAppViewContainer"] [data-stale="true"] { opacity: 1 !important; transition: none !important; }
.st-emotion-cache-1kyxreq, .st-emotion-cache-1gzhbcq, .st-emotion-cache-1629p8f { opacity: 1 !important; transition: none !important; }

/* ── Make placeholders extremely visible ── */
::placeholder { color: #ffffff !important; opacity: 0.9 !important; font-weight: 600 !important; }
input::placeholder, textarea::placeholder { color: #ffffff !important; opacity: 0.9 !important; font-weight: 600 !important; }


/* ── Sidebar base ── */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#061020,#0a1828) !important;
    border-right:1px solid rgba(255,255,255,0.055) !important;
}
/* General sidebar text — but NOT input elements (those get their own rules) */
[data-testid="stSidebar"] > div > div { color:#e2e8f0; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stToggle label {
    color:#94a3b8 !important; font-size:0.78rem !important;
}
[data-testid="stSidebar"] h3 {
    color:#90cdf4 !important; font-size:0.73rem !important;
    text-transform:uppercase; letter-spacing:0.08em;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown { color:#cbd5e1 !important; }

/* ── PRIORITY 1: Sidebar input elements — dark backgrounds, white text ── */
/* Selectbox / dropdown container */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #12243d !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdown"],
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #e2e8f0 !important;
    background-color: transparent !important;
}
/* Number / text inputs */
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background-color: #12243d !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 6px !important;
    caret-color: #90cdf4 !important;
}
[data-testid="stSidebar"] input[type="number"]:focus,
[data-testid="stSidebar"] input[type="text"]:focus {
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 0 2px rgba(66,153,225,0.2) !important;
}
/* Input wrappers */
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: #12243d !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 6px !important;
}

/* ── PRIORITY 1: GLOBAL dark dropdown popover (portal outside sidebar DOM) ── */
[data-baseweb="popover"] {
    background-color: #0d1f38 !important;
}
[data-baseweb="popover"] * {
    color: #e2e8f0 !important;
}
[role="listbox"] {
    background-color: #0d1f38 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 8px !important;
}
[role="listbox"] li,
[role="option"] {
    background-color: #0d1f38 !important;
    color: #e2e8f0 !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background-color: #1a3a6a !important;
    color: #fff !important;
}
/* Selectbox dropdown menu list */
[data-baseweb="menu"] {
    background-color: #0d1f38 !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li {
    background-color: transparent !important;
    color: #e2e8f0 !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #1a3a6a !important;
    color: #fff !important;
}

/* Sidebar primary button */
[data-testid="stSidebar"] [data-testid="stButton"] {
    position: sticky;
    bottom: 20px;
    z-index: 999;
}
[data-testid="stSidebar"] .stButton>button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:white !important;border:none;border-radius:8px;font-weight:600;
    box-shadow:0 8px 16px rgba(37,99,235,0.6);transition:all 0.2s;
}
[data-testid="stSidebar"] .stButton>button:hover {
    transform: translateY(-2px);
    box-shadow:0 10px 20px rgba(37,99,235,0.8);
}

/* Tablet breakpoint */
@media (max-width: 1024px) {
    .stApp .row-widget { flex-direction: column !important; }
    div[data-testid="column"] { width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; }
    .top-banner h1 { font-size: 1.15rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Coordinate transformers ────────────────────────────────────────────────────
_to_wgs84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
_to_utm   = Transformer.from_crs("EPSG:4326",  "EPSG:32631", always_xy=True)

def utm_to_ll(x, y):
    lon, lat = _to_wgs84.transform(x, y)
    return lat, lon

def ll_to_utm(lat, lon):
    x, y = _to_utm.transform(lon, lat)
    return x, y

def response_rating(t):
    if t <= 5:  return "Excellent", "rating-excellent"
    if t <= 10: return "Good",      "rating-good"
    if t <= 15: return "Moderate",  "rating-moderate"
    return           "Critical",  "rating-critical"

def _dist_m(lat1, lon1, lat2, lon2):
    """Quick Euclidean distance in metres via UTM projection."""
    x1, y1 = ll_to_utm(lat1, lon1)
    x2, y2 = ll_to_utm(lat2, lon2)
    return math.hypot(x2 - x1, y2 - y1)

def _nearest_route_point(current_ll, route_ll):
    """Return index of closest point on route to current_ll."""
    clat, clon = current_ll
    best_i, best_d = 0, float("inf")
    for i, (rlat, rlon) in enumerate(route_ll):
        d = _dist_m(clat, clon, rlat, rlon)
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


# ── Priority 3: Startup Data Validation ────────────────────────────────────────
@st.cache_data(show_spinner=False)
def validate_startup_data():
    """Validates existence and schema of required GeoPackages before building graph."""
    required = {
        "data/ambulance_stations.gpkg": ["facility_type", "name", "geometry"],
        "data/incident_points.gpkg": ["type", "geometry"],
        "data/road_network_final.gpkg": ["speed_kmh", "length_m", "time_min", "highway", "geometry"]
    }
    for file_path, required_cols in required.items():
        if not os.path.exists(file_path):
            return f"Missing required file: {file_path}"
        try:
            # Read just 1 row to validate schema and non-emptiness quickly
            gdf = gpd.read_file(file_path, rows=1)
            if len(gdf) == 0:
                return f"File is empty (0 rows): {file_path}"
            missing = [c for c in required_cols if c not in gdf.columns]
            if missing:
                return f"File {file_path} missing required columns: {', '.join(missing)}"
        except Exception as e:
            return f"Corrupt or invalid GeoPackage {file_path}: {e}"
    return None
# ─────────────────────────────────────────────────────────────────────────────

# ── Double-Cache data loader ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading and preparing routing network...")
def load_data():
    validation_error = validate_startup_data()
    if validation_error:
        raise ValueError(f"Data Validation Failed: {validation_error}")
        
    cache_file = "data/app_prepared_data.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print("[load_data] Cache loaded.")
            return data
        except Exception as e:
            print(f"[load_data] Cache error: {e}. Recomputing...")

    G, nodes_gdf = build_graph()
    stations_gdf  = gpd.read_file("data/ambulance_stations.gpkg").to_crs("EPSG:32631")
    stations_gdf  = snap_stations_to_graph(stations_gdf, nodes_gdf)
    incidents_gdf = gpd.read_file("data/incident_points.gpkg").to_crs("EPSG:32631")
    incidents_gdf = snap_incidents_to_graph(incidents_gdf, nodes_gdf)
    incidents_gdf = incidents_gdf.reset_index(drop=True)
    sa_data  = compute_all_service_areas(G, nodes_gdf, stations_gdf, [5, 10, 15])
    def _project_poly(poly):
        if poly is None:
            return None
        try:
            return gpd.GeoSeries([poly], crs="EPSG:32631").to_crs("EPSG:4326").iloc[0]
        except Exception:
            return None

    sa_polys = {
        t: {
            "medical": _project_poly(nodes_to_polygon(nodes_gdf, sa_data[t]["medical"], 150)),
            "fire":    _project_poly(nodes_to_polygon(nodes_gdf, sa_data[t]["fire"],    150)),
        }
        for t in [5, 10, 15]
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump((G, nodes_gdf, stations_gdf, incidents_gdf, sa_data, sa_polys), f)
        print("[load_data] Cache saved.")
    except Exception as e:
        print(f"[load_data] Cache save error: {e}")
    return G, nodes_gdf, stations_gdf, incidents_gdf, sa_data, sa_polys


# ── Route geometry helper ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def path_to_ll_via_geometry(_G, path: list, exact_dest_coord: tuple = None) -> list:
    utm_coords = path_to_detailed_coords(_G, path, exact_dest_coord)
    return [(lat, lon) for x, y in utm_coords for lon, lat in [_to_wgs84.transform(x, y)]]


# ── GEO_HTML redundant block removed ──


# ══════════════════════════════════════════════════════════════════════════════
# MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
INC_COLORS = {"Medical": "#3b82f6", "RTA": "#f59e0b", "Fire": "#ef4444"}
TILE_OPTIONS = {
    "🌑 Dark (Carto)":   "CartoDB dark_matter",
    "🗺️ Street (OSM)":  "OpenStreetMap",
    "🛰️ Light (Carto)": "CartoDB positron",
}


def build_navigable_map(stations_gdf, incidents_gdf, sa_polys,
                        show_cov, cov_t, tile_name,
                        incident_ll=None, vehicle_ll=None,
                        click_mode=None, bearing=0.0, card_dir="N",
                        recenter_ll=None, zoom_start=13):
    tile   = TILE_OPTIONS.get(tile_name, "CartoDB dark_matter")
    center = recenter_ll if recenter_ll else (vehicle_ll if vehicle_ll else [6.565, 3.375])

    m = folium.Map(
        location=center, zoom_start=zoom_start, tiles=tile,
        prefer_canvas=True, control_scale=True,
        attribution_control=False,
    )
    folium.TileLayer("CartoDB dark_matter", name="Dark",   show=False).add_to(m)
    folium.TileLayer("OpenStreetMap",       name="Street", show=False).add_to(m)
    folium.TileLayer("CartoDB positron",    name="Light",  show=False).add_to(m)

    # Add Live Traffic Tile Layer if TOMTOM_API_KEY is configured
    tomtom_api_key = st.secrets.get("TOMTOM_API_KEY")
    if tomtom_api_key:
        folium.TileLayer(
            tiles=f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{{z}}/{{x}}/{{y}}.png?key={tomtom_api_key}",
            attr="TomTom Traffic",
            name="Live Traffic Flow",
            overlay=True,
            control=True,
            opacity=0.8
        ).add_to(m)

    plugins.Fullscreen(position="topright", title="Fullscreen", force_separate_button=True).add_to(m)
    plugins.MeasureControl(position="bottomright", primary_length_unit="meters").add_to(m)
    plugins.Geocoder(position="topleft").add_to(m)

    # Coverage zones
    if show_cov and cov_t in sa_polys:
        for ftype, color in [("medical", "#1d4ed8"), ("fire", "#b91c1c")]:
            poly = sa_polys[cov_t].get(ftype)
            if poly:
                gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
                folium.GeoJson(
                    gdf.__geo_interface__,
                    style_function=lambda x, c=color: {
                        "fillColor": c, "fillOpacity": 0.10,
                        "color": c, "weight": 1.5, "opacity": 0.38,
                    },
                    name=f"{ftype.title()} coverage ≤{cov_t} min",
                ).add_to(m)

    med_group  = folium.FeatureGroup("🏥 Medical Stations", show=True).add_to(m)
    fire_group = folium.FeatureGroup("🚒 Fire Stations",    show=True).add_to(m)
    inc_group  = folium.FeatureGroup("⚡ Incidents",         show=True).add_to(m)

    for _, srow in stations_gdf.iterrows():
        lat, lon = utm_to_ll(srow.geometry.x, srow.geometry.y)
        ftype  = srow.get("facility_type", "medical")
        is_med = ftype == "medical"
        popup_html = (
            f"<div style='font-family:Inter,sans-serif;min-width:160px'>"
            f"<b style='color:{'#2563eb' if is_med else '#dc2626'};font-size:13px'>"
            f"{'🏥' if is_med else '🚒'} {srow['name']}</b><br>"
            f"<span style='font-size:11px;color:#64748b'>{ftype.title()} Station</span></div>"
        )
        marker = folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=srow["name"],
            icon=folium.Icon(
                color="blue" if is_med else "red",
                icon="plus-sign" if is_med else "fire",
                prefix="glyphicon",
            ),
        )
        (med_group if is_med else fire_group).add_child(marker)

    for i, inc in incidents_gdf.iterrows():
        lat, lon = utm_to_ll(inc.geometry.x, inc.geometry.y)
        itype  = inc["type"]
        icolor = INC_COLORS.get(itype, "#888")
        popup_html = (
            f"<div style='font-family:Inter,sans-serif'>"
            f"<b>Incident #{i}</b><br>"
            f"Type: <b style='color:{icolor}'>{itype}</b><br>"
            f"<span style='font-size:10px;color:#64748b'>{lat:.4f}°N, {lon:.4f}°E</span></div>"
        )
        folium.CircleMarker(
            [lat, lon], radius=7,
            color="#000", weight=1.2,
            fill=True, fill_color=icolor, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=175),
            tooltip=f"#{i} — {itype}",
        ).add_to(inc_group)

    if click_mode:
        label = "incident location" if click_mode == "incident" else "vehicle position"
        click_html = (
            f'<div style="position:fixed;top:14px;left:50%;transform:translateX(-50%);'
            f'z-index:9999;background:rgba(6,18,38,0.92);color:#a7f3d0;'
            f'padding:8px 16px;border-radius:20px;font-family:Inter,sans-serif;'
            f'font-size:12px;font-weight:600;border:1px solid rgba(52,211,153,0.3);'
            f'box-shadow:0 4px 16px rgba(0,0,0,0.5);pointer-events:none">'
            f'🖱️ Click anywhere on the map to set the {label}</div>'
        )
        m.get_root().html.add_child(folium.Element(click_html))

    compass_html = f"""
    <div style="position:absolute;top:10px;right:50px;z-index:9999;
                background:rgba(6,18,38,0.92);color:#e2e8f0;
                width:34px;height:34px;border-radius:50%;
                border:1px solid rgba(255,255,255,0.18);
                box-shadow:0 3px 8px rgba(0,0,0,0.5);
                display:flex;align-items:center;justify-content:center;
                pointer-events:auto;"
          title="Vehicle Heading: {bearing:.0f}° {card_dir}">
      <div style="transform:rotate({bearing:.1f}deg);font-size:19px;line-height:1;
                  transition:transform 0.4s cubic-bezier(0.4,0,0.2,1);">🧭</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(compass_html))

    legend = """
    <div style="position:absolute;top:10px;left:50px;z-index:9999;
                background:rgba(6,18,38,0.93);color:#e2e8f0;
                padding:10px 14px;border-radius:10px;
                border:1px solid rgba(255,255,255,0.1);
                font-family:Inter,sans-serif;font-size:10.5px;min-width:175px;">
      <b style="font-size:11.5px;color:#90cdf4">Map Legend</b><br><br>
      <span style="color:#3b82f6">&#9679;</span>&ensp;Medical Station<br>
      <span style="color:#ef4444">&#9650;</span>&ensp;Fire Station<br>
      <span style="color:#3b82f6;opacity:.7">&#9679;</span>&ensp;Medical Incident<br>
      <span style="color:#f59e0b;opacity:.7">&#9679;</span>&ensp;RTA Incident<br>
      <span style="color:#ef4444;opacity:.7">&#9679;</span>&ensp;Fire Incident<br>
      <hr style="border-color:rgba(255,255,255,0.1);margin:5px 0">
      <span style="color:#4285f4">&#9472;&#9472;</span>&ensp;Leg 1: Dispatch → Scene<br>
      <span style="color:#fb923c">&#9472;&#9472;</span>&ensp;Leg 2: Scene → Hospital<br>
      <span style="color:#1d4ed8;opacity:.45">&#9632;</span>&ensp;Medical coverage<br>
      <span style="color:#b91c1c;opacity:.45">&#9632;</span>&ensp;Fire coverage
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    return m


def add_dynamic_markers(m, incident_ll, vehicle_ll, bearing, dest_ll=None):
    import folium
    if incident_ll:
        ilat, ilon = incident_ll
        folium.Marker(
            [ilat, ilon],
            icon=folium.DivIcon(
                html='<div style="font-size:26px;transform:translate(-50%,-50%);'
                     'filter:drop-shadow(0 2px 4px #000)">⚡</div>',
                icon_size=(36, 36), icon_anchor=(18, 18),
            ),
            popup="Active Incident Location",
            tooltip="Active Incident Location",
        ).add_to(m)

    if vehicle_ll:
        vlat, vlon = vehicle_ll
        folium.Marker(
            [vlat, vlon],
            icon=folium.DivIcon(
                html=f'''<div style="position: relative; width: 52px; height: 52px;">
  <style>
    @keyframes gps-pulse {{
      0% {{ transform: scale(0.85); opacity: 0.8; }}
      50% {{ transform: scale(1.35); opacity: 0.3; }}
      100% {{ transform: scale(0.85); opacity: 0.8; }}
    }}
  </style>
  <div style="position: absolute; top: 50%; left: 50%; width: 44px; height: 44px; margin: -22px; border-radius: 50%; background: rgba(59, 130, 246, 0.22); border: 1.5px solid rgba(59, 130, 246, 0.5); box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); animation: gps-pulse 2.2s infinite ease-in-out; transform-origin: 50% 50%;"></div>
  <div style="position: absolute; top: 50%; left: 50%; width: 0; height: 0; margin-left: -9px; margin-top: -20px; border-left: 9px solid transparent; border-right: 9px solid transparent; border-bottom: 18px solid #2563eb; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5)); transform: rotate({bearing}deg); transform-origin: 50% 100%;"></div>
  <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));">🚑</div>
</div>''',
                icon_size=(52, 52), icon_anchor=(26, 26),
            ),
            popup="Vehicle Current Position",
            tooltip="Vehicle Current Position",
        ).add_to(m)

    if dest_ll:
        dlat, dlon = dest_ll
        folium.Marker(
            [dlat, dlon],
            icon=folium.DivIcon(
                html='<div style="font-size:26px;transform:translate(-50%,-50%);'
                     'filter:drop-shadow(0 2px 4px #000)">🏁</div>',
                icon_size=(36, 36), icon_anchor=(18, 18),
            ),
            popup="Custom Destination (Hospital Overwrite)",
            tooltip="Custom Destination (Hospital Overwrite)",
        ).add_to(m)

# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY 3: Navigation-quality route overlay
# ══════════════════════════════════════════════════════════════════════════════
def overlay_route(m, result_2leg, G, sim_progress=None, exact_dest_coord=None, exact_h_coord=None):
    """Draw route with Google-Maps-style thick halo lines + directional arrows."""
    if not result_2leg:
        return m, [], []

    if "leg1_ll_direct" in result_2leg:
        leg1_ll = result_2leg["leg1_ll_direct"]
    else:
        leg1_ll = path_to_ll_via_geometry(G, result_2leg["leg1_path"], exact_dest_coord)

    if len(leg1_ll) >= 2:
        # Leg 1 thick green halo
        folium.PolyLine(
            leg1_ll, color="#34d399", weight=14, opacity=0.6,
        ).add_to(m)
        # Leg 1 core (dark green)
        folium.PolyLine(
            leg1_ll, color="#065f46", weight=6, opacity=1.0,
            tooltip=f"Leg 1 — Dispatch to Scene: {result_2leg['leg1_time_min']:.2f} min",
        ).add_to(m)

    sg = result_2leg["leg1_station_geom"]
    s_ll = utm_to_ll(sg.x, sg.y)
    folium.Marker(
        s_ll,
        icon=folium.Icon(color="darkblue", icon="home", prefix="glyphicon"),
        popup=folium.Popup(
            f"<b style='color:#2563eb'>DISPATCH STATION</b><br>"
            f"{result_2leg['leg1_station_name']}<br>"
            f"<b>Leg 1: {result_2leg['leg1_time_min']:.2f} min</b>",
            max_width=210),
        tooltip="Dispatch Station",
    ).add_to(m)

    if leg1_ll:
        inc_ll = leg1_ll[-1]
        
        # ── PART A3 FIX: Visual Overlap Offset ──
        # If the incident snapped to the exact same node as the dispatch station, slightly offset it visually
        if abs(inc_ll[0] - s_ll[0]) < 1e-6 and abs(inc_ll[1] - s_ll[1]) < 1e-6:
            inc_ll = (inc_ll[0], inc_ll[1] + 0.00015)
            
        folium.Marker(
            inc_ll,
            icon=folium.Icon(color="orange", icon="exclamation-sign", prefix="glyphicon"),
            popup=folium.Popup("<b>INCIDENT SCENE</b>", max_width=140),
            tooltip="Incident Scene",
        ).add_to(m)

    leg2_ll = []
    if result_2leg.get("has_leg2"):
        if "leg2_ll_direct" in result_2leg:
            leg2_ll = result_2leg["leg2_ll_direct"]
        elif result_2leg.get("leg2_path"):
            leg2_ll = path_to_ll_via_geometry(G, result_2leg["leg2_path"], exact_h_coord)
            
        if len(leg2_ll) >= 2:
            # Leg 2 thin orange core (stands out clearly over the green halo)
            folium.PolyLine(
                leg2_ll, color="#fbbf24", weight=6, opacity=1.0,
                tooltip=f"Leg 2 — Scene to Hospital: {result_2leg['leg2_time_min']:.2f} min",
            ).add_to(m)
        hg = result_2leg["leg2_hospital_geom"]
        h_ll = utm_to_ll(hg.x, hg.y)
        
        # ── PART A3 FIX: Visual Overlap Offset ──
        # Offset hospital marker if it overlaps the incident or station
        if leg1_ll:
            if abs(h_ll[0] - inc_ll[0]) < 1e-6 and abs(h_ll[1] - inc_ll[1]) < 1e-6:
                h_ll = (h_ll[0] - 0.00015, h_ll[1])
        if abs(h_ll[0] - s_ll[0]) < 1e-6 and abs(h_ll[1] - s_ll[1]) < 1e-6:
            h_ll = (h_ll[0] - 0.00015, h_ll[1] + 0.00015)

        folium.Marker(
            h_ll,
            icon=folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
            popup=folium.Popup(
                f"<b style='color:#dc2626'>RECEIVING HOSPITAL</b><br>"
                f"{result_2leg['leg2_hospital_name']}<br>"
                f"<b>Leg 2: {result_2leg['leg2_time_min']:.2f} min</b>",
                max_width=220),
            tooltip="Receiving Hospital",
        ).add_to(m)

    veh_origin = result_2leg.get("vehicle_origin_ll")
    if veh_origin:
        folium.PolyLine(
            [veh_origin, s_ll], color="#a78bfa", weight=2.5,
            opacity=0.55, dash_array="5 8",
            tooltip="Vehicle to assigned station area",
        ).add_to(m)
        folium.Marker(
            veh_origin,
            icon=folium.DivIcon(
                html='<div style="font-size:20px;transform:translate(-50%,-50%)">🚑</div>',
                icon_size=(32, 32), icon_anchor=(16, 16),
            ),
            popup="Vehicle Start Position",
            tooltip="Vehicle Start Position",
        ).add_to(m)

    all_ll = leg1_ll + leg2_ll
    if sim_progress is not None and sim_progress > 0 and len(all_ll) >= 2:
        total_pts = len(all_ll)
        idx = min(int(sim_progress * (total_pts - 1)), total_pts - 1)
        veh_ll = all_ll[idx]
        
        # Calculate bearing dynamically for simulation progress marker
        v_bearing = 0.0
        next_idx = min(idx + 1, total_pts - 1)
        if idx < next_idx:
            lat1, lon1 = all_ll[idx]
            lat2, lon2 = all_ll[next_idx]
            x1, y1 = ll_to_utm(lat1, lon1)
            x2, y2 = ll_to_utm(lat2, lon2)
            v_bearing = bearing_deg(x1, y1, x2, y2)

        folium.Marker(
            veh_ll,
            icon=folium.DivIcon(
                html=f'''<div style="position: relative; width: 52px; height: 52px;">
  <style>
    @keyframes gps-pulse {{
      0% {{ transform: scale(0.85); opacity: 0.8; }}
      50% {{ transform: scale(1.35); opacity: 0.3; }}
      100% {{ transform: scale(0.85); opacity: 0.8; }}
    }}
  </style>
  <div style="position: absolute; top: 50%; left: 50%; width: 44px; height: 44px; margin: -22px; border-radius: 50%; background: rgba(59, 130, 246, 0.22); border: 1.5px solid rgba(59, 130, 246, 0.5); box-shadow: 0 0 10px rgba(59, 130, 246, 0.4); animation: gps-pulse 2.2s infinite ease-in-out; transform-origin: 50% 50%;"></div>
  <div style="position: absolute; top: 50%; left: 50%; width: 0; height: 0; margin-left: -9px; margin-top: -20px; border-left: 9px solid transparent; border-right: 9px solid transparent; border-bottom: 18px solid #2563eb; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5)); transform: rotate({v_bearing}deg); transform-origin: 50% 100%;"></div>
  <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));">🚑</div>
</div>''',
                icon_size=(52, 52), icon_anchor=(26, 26),
            ),
            tooltip=f"Simulation — {sim_progress*100:.0f}% of route",
        ).add_to(m)

    if all_ll:
        lats = [c[0] for c in all_ll]
        lons = [c[1] for c in all_ll]
        pad  = 0.005
        m.fit_bounds([[min(lats)-pad, min(lons)-pad], [max(lats)+pad, max(lons)+pad]])

    return m, leg1_ll, leg2_ll


def compute_remaining_time_and_dist(result, prog, G, live_speed_kmh=None, all_ll=None):
    """Accurately calculates remaining distance and time based on actual paths and dynamic speed."""
    if all_ll is None:
        all_ll = st.session_state.get("leg1_ll", []) + st.session_state.get("leg2_ll", [])
        
    if not all_ll:
        return 0.0, 0.0
        
    total_dist = 0.0
    segments = []
    # Calculate great-circle distances between all coordinate pairs
    for i in range(len(all_ll) - 1):
        lat1, lon1 = all_ll[i]
        lat2, lon2 = all_ll[i+1]
        x1, y1 = ll_to_utm(lat1, lon1)
        x2, y2 = ll_to_utm(lat2, lon2)
        d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        segments.append(d)
        total_dist += d
        
    target_dist = total_dist * prog
    rem_dist = max(0.0, total_dist - target_dist)
    
    if live_speed_kmh is not None and live_speed_kmh > 0:
        # speed is km/h, dist is meters. Time = (dist/1000) / speed * 60
        # Clamp speed so it doesn't predict crazy ETAs if stopped
        eff_speed = max(live_speed_kmh, 5.0) 
        rem_time = (rem_dist / 1000.0) / eff_speed * 60.0 # in minutes
    else:
        # Fallback to static total time proportional to distance remaining
        tot_time = result.get("total_time_min", 0.0)
        rem_time = tot_time * (1.0 - prog)
        
    return rem_dist, rem_time


# ══════════════════════════════════════════════════════════════════════════════
# 3D PYDECK MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_3d_pydeck_chart(
    result, G, stations_gdf, incidents_gdf, sa_polys,
    sim_progress=0.0, vehicle_ll=None, bearing=0.0,
    is_navigating=False, tile_choice="OpenStreetMap", force_2d=False, exact_dest_coord=None
):
    camera_follow = False
    layers = []

    if "Street" in tile_choice or "Light" in tile_choice or tile_choice in ["OpenStreetMap", "CartoDB positron"]:
        map_style = pdk.map_styles.LIGHT
    else:
        map_style = pdk.map_styles.CARTO_DARK

    if is_navigating:
        # We must use Carto as the base to prevent Mapbox 401 Unauthorized blank screens,
        # but we layer a free Esri Satellite TileLayer on top to fulfill the user's request!
        layers.insert(0, pdk.Layer(
            "TileLayer",
            data=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            opacity=1.0,
            pickable=False
        ))
        
        # Add a Street Labels layer on top of the satellite imagery so streets are identifiable
        layers.append(pdk.Layer(
            "TileLayer",
            data="https://a.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}.png",
            opacity=0.8,
            pickable=False
        ))

    # Add Live Traffic Tile Layer if TOMTOM_API_KEY is configured
    tomtom_api_key = st.secrets.get("TOMTOM_API_KEY")
    if tomtom_api_key:
        layers.append(pdk.Layer(
            "TileLayer",
            data=f"https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{{z}}/{{x}}/{{y}}.png?key={tomtom_api_key}",
            opacity=0.7,
            pickable=False,
            tile_size=256,
            max_requests=-1,
        ))

    # Always show service areas (if any) so the map doesn't look empty
    poly_data = []
    for t in sa_polys.keys():
        for ftype, col_rgba in [("medical", [29, 78, 216, 40]), ("fire", [185, 28, 28, 40])]:
            poly = sa_polys[t].get(ftype)
            if poly:
                sim_poly = poly.simplify(35.0)
                geoms = [sim_poly] if sim_poly.geom_type == "Polygon" else (
                    list(sim_poly.geoms) if sim_poly.geom_type == "MultiPolygon" else [])
                for g in geoms:
                    coords_wgs = [list(utm_to_ll(x, y))[::-1] for x, y in g.exterior.coords]
                    poly_data.append({
                        "polygon":   coords_wgs,
                        "color":     col_rgba,
                        "elevation": 45 if ftype == "medical" else 75,
                        "name":      f"{ftype.title()} ≤{t} min coverage",
                    })
        if poly_data:
            layers.append(pdk.Layer(
                "PolygonLayer", data=pd.DataFrame(poly_data),
                get_polygon="polygon", get_fill_color="color", get_elevation="elevation",
                extruded=True, wireframe=True,
                get_line_color=[255, 255, 255, 50], get_line_width=2, pickable=True,
            ))

    station_data = []
    station_sign_data = []
    for _, srow in stations_gdf.iterrows():
        lat, lon = utm_to_ll(srow.geometry.x, srow.geometry.y)
        ftype = srow.get("facility_type", "medical")
        color = [37, 99, 235, 240] if ftype == "medical" else [220, 38, 38, 240]
        symbol = "🏥" if ftype == "medical" else "🚒"
        station_data.append({"name": srow["name"], "type": f"{ftype.title()} Station",
                              "position": [lon, lat, 10], "color": color, "radius": 40})
        station_sign_data.append({
            "position": [lon, lat, 35],
            "text": f"{symbol} {srow['name']}",
            "color": [255, 255, 255, 255],
            "name": srow["name"],
            "type": f"{ftype.title()} Station"
        })

    layers.append(pdk.Layer(
        "ScatterplotLayer", data=pd.DataFrame(station_data),
        get_position="position", get_fill_color="color", get_radius="radius",
        radius_min_pixels=6, pickable=True,
    ))
    if station_sign_data:
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame(station_sign_data),
            get_position="position",
            get_text="text",
            get_size=18,
            size_min_pixels=16,
            size_max_pixels=32,
            get_color="color",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            background=True,
            get_background_color=[15, 23, 42, 230],
            font_family="'Inter', sans-serif",
            font_weight="bold",
            pickable=True,
        ))

    cmap = {"Medical": [59,130,246,230], "RTA": [245,158,11,230], "Fire": [239,68,68,230]}
    inc_symbols = {"Medical": "🩺", "RTA": "💥", "Fire": "🔥"}
    inc_data = []
    inc_sign_data = []
    for i, inc in incidents_gdf.iterrows():
        lat, lon = utm_to_ll(inc.geometry.x, inc.geometry.y)
        itype = inc["type"]
        symbol = inc_symbols.get(itype, "🚨")
        inc_data.append({"name": f"Incident #{i}", "type": f"{itype} Incident",
                          "position": [lon, lat, 10],
                          "color": cmap.get(itype, [156,163,175,230]), "radius": 25})
        inc_sign_data.append({
            "position": [lon, lat, 22],
            "text": symbol,
            "color": [255, 255, 255, 255],
            "name": f"Incident #{i}",
            "type": f"{itype} Incident"
        })
                          
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=pd.DataFrame(inc_data),
        get_position="position", get_fill_color="color", get_radius="radius",
        radius_min_pixels=4, pickable=True,
    ))
    if inc_sign_data:
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame(inc_sign_data),
            get_position="position",
            get_text="text",
            get_size=24,
            size_min_pixels=18,
            size_max_pixels=36,
            get_color="color",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            background=True,
            get_background_color=[185, 28, 28, 230],
            font_family="'Inter', sans-serif",
            font_weight="bold",
            pickable=True,
        ))

    if result:
        # Precompute leg1_ll and leg2_ll are stored in session state, but we fall back if not found
        hg = result.get("leg2_hospital_geom")
        exact_h_coord = (hg.x, hg.y) if hg else None
        if "leg1_ll_direct" in result:
            leg1_ll = result["leg1_ll_direct"]
        else:
            leg1_ll = st.session_state.get("leg1_ll") or path_to_ll_via_geometry(G, result["leg1_path"], exact_dest_coord)
            
        leg2_ll = []
        if result.get("has_leg2"):
            if "leg2_ll_direct" in result:
                leg2_ll = result["leg2_ll_direct"]
            elif result.get("leg2_path"):
                leg2_ll = st.session_state.get("leg2_ll") or path_to_ll_via_geometry(G, result["leg2_path"], exact_h_coord)
        
        all_ll = leg1_ll + leg2_ll
        if len(all_ll) >= 2:
            total_pts = len(all_ll)
            cur_i = min(int(sim_progress * (total_pts - 1)), total_pts - 1)
            
            if is_navigating:
                traveled_ll = all_ll[:cur_i+1]
                
                if len(traveled_ll) >= 2:
                    layers.append(pdk.Layer(
                        "PathLayer",
                        data=pd.DataFrame([{"path": [[lon, lat, 15] for lat, lon in traveled_ll], "name": "Traveled"}]),
                        get_path="path", get_color=[156, 163, 175, 180], get_width=10,
                        width_min_pixels=5, pickable=False,
                    ))
                
                # Split remaining route into Leg 1 and Leg 2
                if cur_i < len(leg1_ll) - 1:
                    remaining_leg1 = leg1_ll[cur_i:]
                    remaining_leg2 = leg2_ll
                else:
                    remaining_leg1 = []
                    leg2_start_i = cur_i - (len(leg1_ll) - 1)
                    remaining_leg2 = leg2_ll[leg2_start_i:] if leg2_ll else []
                
                if len(remaining_leg1) >= 2:
                    layers.append(pdk.Layer(
                        "PathLayer",
                        data=pd.DataFrame([{"path": [[lon, lat, 15] for lat, lon in remaining_leg1], "name": "Remaining Leg 1", "type": "Active Route"}]),
                        get_path="path", get_color=[66, 133, 244, 255], get_width=12,
                        width_min_pixels=6, pickable=True,
                    ))
                if len(remaining_leg2) >= 2:
                    layers.append(pdk.Layer(
                        "PathLayer",
                        data=pd.DataFrame([{"path": [[lon, lat, 25] for lat, lon in remaining_leg2], "name": "Remaining Leg 2", "type": "Active Route"}]),
                        get_path="path", get_color=[251, 146, 60, 255], get_width=12,
                        width_min_pixels=6, pickable=True,
                    ))
            else:
                if len(leg1_ll) >= 2:
                    layers.append(pdk.Layer(
                        "PathLayer",
                        data=pd.DataFrame([{"path": [[lon, lat, 15] for lat, lon in leg1_ll], "name": f"Leg 1: {result['leg1_time_min']:.2f} min", "type": "Planned Route"}]),
                        get_path="path", get_color=[66,133,244,230], get_width=10,
                        width_min_pixels=5, pickable=True,
                    ))
                if leg2_ll and len(leg2_ll) >= 2:
                    layers.append(pdk.Layer(
                        "PathLayer",
                        data=pd.DataFrame([{"path": [[lon, lat, 25] for lat, lon in leg2_ll], "name": f"Leg 2: {result['leg2_time_min']:.2f} min", "type": "Planned Route"}]),
                        get_path="path", get_color=[251,146,60,230], get_width=10,
                        width_min_pixels=5, pickable=True,
                    ))

    # Draw custom destination if present
    clicked_dest_lat = st.session_state.get("clicked_dest_lat")
    clicked_dest_lon = st.session_state.get("clicked_dest_lon")
    if clicked_dest_lat and clicked_dest_lon:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"position": [clicked_dest_lon, clicked_dest_lat, 10], "color": [249, 115, 22, 240], "radius": 40, "name": "Custom Destination", "type": "Hospital Overwrite"}]),
            get_position="position", get_fill_color="color", get_radius="radius",
            radius_min_pixels=6, pickable=True,
        ))
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame([{"position": [clicked_dest_lon, clicked_dest_lat, 22], "text": "🏁 Custom Destination", "color": [255, 255, 255, 255], "name": "Custom Destination", "type": "Hospital Overwrite"}]),
            get_position="position", get_text="text", get_size=20,
            size_min_pixels=14, size_max_pixels=28, get_color="color",
            get_alignment_baseline="'center'", get_text_anchor="'middle'",
            background=True, get_background_color=[15, 23, 42, 230],
            font_family="'Inter', sans-serif", font_weight="bold", pickable=True,
        ))

    if vehicle_ll:
        vlat, vlon = vehicle_ll
        
        # Pulse accuracy halo beneath vehicle
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{"position": [vlon, vlat, 30], "color": [59, 130, 246, 50], "radius": 120}]),
            get_position="position", get_fill_color="color", get_radius="radius",
            radius_min_pixels=15, pickable=False,
        ))
        
        # High-visibility 3D directional arrow (➤) rotated by heading
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame([{"position": [vlon, vlat, 30], "text": "➤", "bearing": bearing - 90, "color": [37, 99, 235, 255]}]),
            get_position="position",
            get_text="text",
            get_angle="bearing",
            get_size=42,
            get_color="color",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            pickable=False,
        ))

        # Vehicle Floating billboard sign
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame([{"position": [vlon, vlat, 30], "text": "🚑", "color": [255, 255, 255, 255]}]),
            get_position="position",
            get_text="text",
            get_size=22,
            get_color="color",
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            pickable=True,
        ))

    center_lat = vehicle_ll[0] if vehicle_ll else 6.565
    center_lon = vehicle_ll[1] if vehicle_ll else 3.375
    
    if is_navigating:
        camera_follow = st.session_state.get("camera_follow", True)
        
        # Calculate dynamic bearing
        if camera_follow:
            # We want the map to physically rotate so the vehicle always points "Up" (like Google Maps)
            cam_bearing = bearing
            st.session_state["prev_cam_bearing"] = bearing
            
            view_state = pdk.ViewState(
                latitude=center_lat, longitude=center_lon,
                zoom=16.5 if force_2d else 18.5, 
                pitch=0 if force_2d else 75, 
                bearing=0 if force_2d else cam_bearing,
                transition_duration=0, # Remove transition to avoid jerky fighting with Streamlit reruns
            )
        else:
            view_state = pdk.ViewState(
                latitude=center_lat, longitude=center_lon,
                zoom=16.5 if force_2d else 18.0, 
                pitch=0 if force_2d else 75, 
                bearing=st.session_state.get("prev_cam_bearing", 0)
            )
    else:
        view_state = pdk.ViewState(
            latitude=center_lat, longitude=center_lon,
            zoom=12.8, pitch=45, bearing=bearing,
        )

    deck = pdk.Deck(
        layers=layers, initial_view_state=view_state,
        map_style=map_style,
        tooltip={"html": "<b>{name}</b><br/><i>{type}</i>", "style": {"backgroundColor": "#0d1f38", "color": "white"}},
    )
    
    # Use dynamic key during navigation follow mode to force Pydeck context to mount/re-center on the vehicle
    pdk_key = "pydeck_navigation_chart"
    if is_navigating and camera_follow:
        pdk_key = f"pydeck_chart_nav_{center_lat:.6f}_{center_lon:.6f}"
        
    st.pydeck_chart(deck, use_container_width=True, height=800, key=pdk_key)

    # Overlay 3D map legend using absolute negative-margin container positioning
    tomtom_key_present = bool(st.secrets.get("TOMTOM_API_KEY"))
    live_traffic_html = '<br><span style="color: #34d399;">&#9679;</span>&ensp;TomTom Live Traffic Active' if tomtom_key_present else ''
    
    st.markdown(f"""
    <div class="pydeck-legend-container" style="position: relative; margin-top: -800px; height: 800px; pointer-events: none; z-index: 1000;">
        <div style="position: absolute; top: 20px; right: 20px; 
                    background: rgba(6,18,38,0.92); color: #e2e8f0; 
                    padding: 10px 14px; border-radius: 8px; 
                    border: 1px solid rgba(255,255,255,0.1); 
                    font-family: Inter, sans-serif; font-size: 10.5px; 
                    min-width: 165px; pointer-events: auto;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
          <b style="font-size: 11px; color: #90cdf4;">3D Map Legend</b><br><br>
          <span style="color: #3b82f6;">&#9679;</span>&ensp;Medical Station (Blue)<br>
          <span style="color: #ef4444;">&#9650;</span>&ensp;Fire Station (Red)<br>
          <span style="color: #10b981;">&#9679;</span>&ensp;Active Vehicle (Green)<br>
          <span style="color: #3b82f6; opacity: .7;">&#9679;</span>&ensp;Medical Incident<br>
          <span style="color: #f59e0b; opacity: .7;">&#9679;</span>&ensp;RTA Incident<br>
          <span style="color: #ef4444; opacity: .7;">&#9679;</span>&ensp;Fire Incident<br>
          <hr style="border-color: rgba(255,255,255,0.1); margin: 5px 0;">
          <span style="color: #4285f4;">&#9472;&#9472;</span>&ensp;Leg 1: Dispatch → Scene<br>
          <span style="color: #fb923c;">&#9472;&#9472;</span>&ensp;Leg 2: Scene → Hospital<br>
          <span style="color: rgba(29,78,216,0.35);">&#9632;</span>&ensp;Medical coverage<br>
          <span style="color: rgba(185,28,28,0.35);">&#9632;</span>&ensp;Fire coverage{live_traffic_html}
        </div>
    </div>
    <div style="height: 0px; margin-top: 0px;"></div>
    """, unsafe_allow_html=True)


LAGOS_AREAS = {
    "Ikeja": {"lat": 6.5965, "lon": 3.3421, "display_name": "Ikeja, Lagos, Nigeria"},
    "Yaba": {"lat": 6.5147, "lon": 3.3831, "display_name": "Yaba, Lagos Mainland, Lagos, Nigeria"},
    "Lekki": {"lat": 6.4698, "lon": 3.5852, "display_name": "Lekki, Eti Osa, Lagos, Nigeria"},
    "Victoria Island": {"lat": 6.4281, "lon": 3.4219, "display_name": "Victoria Island, Eti Osa, Lagos, Nigeria"},
    "Surulere": {"lat": 6.4952, "lon": 3.3508, "display_name": "Surulere, Lagos, Nigeria"},
    "Ikoyi": {"lat": 6.4526, "lon": 3.4385, "display_name": "Ikoyi, Eti Osa, Lagos, Nigeria"},
    "Maryland": {"lat": 6.5746, "lon": 3.3664, "display_name": "Maryland, Ikeja, Lagos, Nigeria"},
    "Oshodi": {"lat": 6.5517, "lon": 3.3435, "display_name": "Oshodi, Oshodi-Isolo, Lagos, Nigeria"},
    "Festac Town": {"lat": 6.4682, "lon": 3.2842, "display_name": "Festac Town, Amuwo-Odofin, Lagos, Nigeria"},
    "Gbagada": {"lat": 6.5558, "lon": 3.3888, "display_name": "Gbagada, Kosofe, Lagos, Nigeria"},
    "Ajah": {"lat": 6.4667, "lon": 3.5667, "display_name": "Ajah, Eti Osa, Lagos, Nigeria"},
    "Ikorodu": {"lat": 6.6194, "lon": 3.5105, "display_name": "Ikorodu, Lagos, Nigeria"},
    "Mushin": {"lat": 6.5348, "lon": 3.3444, "display_name": "Mushin, Lagos, Nigeria"},
    "Agege": {"lat": 6.6177, "lon": 3.3228, "display_name": "Agege, Lagos, Nigeria"},
    "Apapa": {"lat": 6.4447, "lon": 3.3637, "display_name": "Apapa, Lagos, Nigeria"},
    "Badagry": {"lat": 6.4316, "lon": 2.8876, "display_name": "Badagry, Lagos, Nigeria"},
    "Epe": {"lat": 6.5841, "lon": 3.9834, "display_name": "Epe, Lagos, Nigeria"}
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Priority 4: Access Control (TEMPORARILY DISABLED) ──────────────────────
    # To re-enable password protection, uncomment the block below:
    # if "password_correct" not in st.session_state:
    #     st.session_state["password_correct"] = False
    # if not st.session_state["password_correct"]:
    #     st.markdown("<h2 style='text-align: center; margin-top: 50px;'>🔒 Restricted Access</h2>", unsafe_allow_html=True)
    #     pwd = st.text_input("Enter system password:", type="password", key="pwd_input")
    #     if pwd:
    #         secret_pwd = st.secrets.get("password")
    #         if not secret_pwd:
    #             st.error("⚠️ Application password not configured in Streamlit Secrets.")
    #         elif pwd == secret_pwd:
    #             st.session_state["password_correct"] = True
    #             st.rerun()
    #         else:
    #             st.error("Incorrect password.")
    #     st.stop()
    # ───────────────────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="top-banner">
      <div class="logo">🚑</div>
      <div>
        <h1>GIS-Based Optimal Ambulance Routing &amp; Emergency Response Analysis</h1>
        <p>Dijkstra Network Routing · Two-Leg Dispatch · Turn-by-Turn Navigation · 3D Visualization · Lagos State, Nigeria</p>
      </div>
      <div class="uni">
        <strong style="color: #fff;">Makanjuola, Thomas Oluwadamilare</strong><br>
        <span style="color: #94a3b8; font-weight: 500;">Matric No:</span> <strong style="color: #38bdf8;">125/21/1/0180</strong><br>
        <span style="color: #60a5fa; font-weight: 600;">Abiola Ajimobi Technical University, Ibadan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Initialising — loading pre-prepared routing network..."):
        try:
            G, nodes_gdf, stations_gdf, incidents_gdf, sa_data, sa_polys = load_data()
        except Exception as e:
            st.error(f"**Initialisation failed:** {e}")
            st.stop()

    total_nodes = len(G.nodes)
    n_med  = len(stations_gdf[stations_gdf["facility_type"] == "medical"])
    n_fire = len(stations_gdf[stations_gdf["facility_type"] == "fire"])
    med_10, fire_10 = len(sa_data[10]["medical"]), len(sa_data[10]["fire"])
    med_15, fire_15 = len(sa_data[15]["medical"]), len(sa_data[15]["fire"])

    # ── Session state defaults ─────────────────────────────────────────────────
    defaults = {
        "result": None, "inc_geom": None, "inc_type": None,
        "leg1_ll": [], "leg2_ll": [],
        "sim_progress": 0.0, "sim_tracking": False,
        "voice_enabled": False, "last_spoken_step": None,
        "clicked_inc_lat": None, "clicked_inc_lon": None,
        "clicked_veh_lat": None, "clicked_veh_lon": None,
        "clicked_dest_lat": None, "clicked_dest_lon": None,
        "custom_dest_node": None,
        "map_clicked_lat": None, "map_clicked_lon": None,
        "click_mode": None,
        "geo_lat": None, "geo_lon": None,
        "geo_granted": False,          # True once user has detected location
        "live_nav_mode": False,        # Priority 4: live GPS mode
        "live_veh_lat": None, "live_veh_lon": None,
        "live_off_route": False,
        "recenter_trigger": False,
        "orig_inc_node": None,         # For live re-routing
        "route_version": 0,            # Incremented each time a new route is computed
        "last_rendered_map_key": None,
        "last_processed_click": None,
        "blocked_edges": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Handle map click before building anything else (prevents rendering lag and blank map on rerun) ──
    last_key = st.session_state.get("last_rendered_map_key")
    if last_key and last_key in st.session_state:
        map_val = st.session_state[last_key]
        if map_val and map_val.get("last_clicked"):
            clicked = map_val["last_clicked"]
            clat, clon = clicked["lat"], clicked["lng"]
            last_processed = st.session_state.get("last_processed_click")
            if last_processed != (clat, clon):
                st.session_state["last_processed_click"] = (clat, clon)
                st.session_state["map_clicked_lat"] = clat
                st.session_state["map_clicked_lon"] = clon

    # ── Auto-refresh tick — sim tracking OR live nav ───────────────────────────
    if st.session_state["sim_tracking"] and not st.session_state["live_nav_mode"] and st.session_state.get("result"):
        # Refresh every 1000ms for buttery-smooth vehicle tracking on the map
        st_autorefresh(interval=1000, key="nav_autorefresh_tick")
        import time
        now = time.time()
        if "sim_last_update" not in st.session_state:
            st.session_state["sim_last_update"] = now
            
        dt = now - st.session_state["sim_last_update"]
        st.session_state["sim_last_update"] = now
        
        # Avoid huge jumps if the browser slept or paused
        if dt > 3.0:
            dt = 1.0
            
        total_time_min = st.session_state["result"].get("total_time_min", 5.0)
        total_duration_sec = max(total_time_min * 60.0, 1.0)
        
        # 1x, 2x, 5x, etc. simulation speed multiplier
        sim_speed_multiplier = float(st.session_state.get("sim_speed_sel", 5.0))
        prog_increment = (dt / total_duration_sec) * sim_speed_multiplier
        
        next_prog = min(1.0, st.session_state["sim_progress"] + prog_increment)
        st.session_state["sim_progress"] = next_prog
        if next_prog >= 1.0:
            st.session_state["sim_tracking"] = False

    # Priority 4: live nav polls GPS every 4 seconds
    if st.session_state["live_nav_mode"]:
        st_autorefresh(interval=4000, key="live_nav_autorefresh")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Map Settings")
        tile_choice = st.selectbox("Basemap style", list(TILE_OPTIONS.keys()), index=1)
        show_cov = st.toggle("Coverage zones", value=True)
        cov_t = st.radio("Threshold", [5, 10, 15], index=1,
                         format_func=lambda x: f"{x} min", horizontal=True)
        
    with st.sidebar:
        st.markdown("---")
        med_5 = 100*len(sa_data[5]['medical'])/total_nodes
        fire_5 = 100*len(sa_data[5]['fire'])/total_nodes
        med_10_pct = 100*med_10/total_nodes
        fire_10_pct = 100*fire_10/total_nodes
        med_15_pct = 100*med_15/total_nodes
        fire_15_pct = 100*fire_15/total_nodes

        import textwrap
        stats_html = textwrap.dedent(f"""<div class="stat-group">
            <div class="stat-header">📊 System Overview</div>
            <div class="stat-row">
                <span>🛣️ Road Network</span>
                <span class="badge-value">{len(G.nodes):,} nodes</span>
            </div>
            <div class="stat-row">
                <span>🏥 Medical Stations</span>
                <span class="badge-value" style="color:#93c5fd">{n_med}</span>
            </div>
            <div class="stat-row">
                <span>🚒 Fire Stations</span>
                <span class="badge-value" style="color:#fca5a5">{n_fire}</span>
            </div>
            <div class="stat-row">
                <span>🚨 Simulated Incidents</span>
                <span class="badge-value">{len(incidents_gdf)}</span>
            </div>
        </div>
        <div class="stat-group">
            <div class="stat-header">⏱️ Network Coverage Analysis</div>
            <div class="stat-row" style="margin-bottom: 0;">
                <span>5-Min Medical coverage</span>
                <span class="badge-value" style="color:#60a5fa">{med_5:.0f}%</span>
            </div>
            <div class="bar-container"><div class="med-bar" style="width: {med_5}%"></div></div>
            <div class="stat-row" style="margin-bottom: 0;">
                <span>5-Min Fire coverage</span>
                <span class="badge-value" style="color:#f87171">{fire_5:.0f}%</span>
            </div>
            <div class="bar-container"><div class="fire-bar" style="width: {fire_5}%"></div></div>
            <div class="stat-row" style="margin-bottom: 0;">
                <span>10-Min Medical coverage</span>
                <span class="badge-value" style="color:#60a5fa">{med_10_pct:.0f}%</span>
            </div>
            <div class="bar-container"><div class="med-bar" style="width: {med_10_pct}%"></div></div>
            <div class="stat-row" style="margin-bottom: 0;">
                <span>10-Min Fire coverage</span>
                <span class="badge-value" style="color:#f87171">{fire_10_pct:.0f}%</span>
            </div>
            <div class="bar-container"><div class="fire-bar" style="width: {fire_10_pct}%"></div></div>
        </div>""")
        st.markdown(stats_html, unsafe_allow_html=True)
    
    # ── Emergency Dispatch Panel (Main Screen Top, Easily Accessible) ──
    st.markdown("### 🚑 Emergency Dispatch Panel")
    with st.container():
        st.markdown("<div class='dispatch-box'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        with col1:
            st.markdown("#### 🚨 Incident Location")
            inc_mode = st.radio(
                "Set incident by:",
                ["Preset list", "Click on map", "Enter coordinates"],
                key="inc_mode_sel"
            )
            inc_node, inc_geom, inc_type, inc_lat, inc_lon = None, None, "Medical", None, None

            if inc_mode == "Preset list":
                st.session_state["click_mode"] = None
                inc_labels = [f"#{i} — {r['type']}" for i, r in incidents_gdf.iterrows()]
                sel = st.selectbox("Select incident:", inc_labels)
                sel_idx  = int(sel.split("—")[0].strip().replace("#",""))
                inc_row  = incidents_gdf.loc[sel_idx]
                inc_node = inc_row["node_id"]
                inc_geom = inc_row.geometry
                inc_type = inc_row["type"]
                inc_lat, inc_lon = utm_to_ll(inc_geom.x, inc_geom.y)
                st.caption(f"📍 {inc_lat:.5f}°N, {inc_lon:.5f}°E")

            elif inc_mode == "Click on map":
                st.session_state["click_mode"] = "incident"
                st.markdown("""
                <div class="click-info">
                  🖱️ Set target to 'Incident Location' then click the map.
                </div>""", unsafe_allow_html=True)
                if st.session_state.get("clicked_inc_lat"):
                    inc_lat = st.session_state["clicked_inc_lat"]
                    inc_lon = st.session_state["clicked_inc_lon"]
                    cx, cy  = ll_to_utm(inc_lat, inc_lon)
                    inc_geom = Point(cx, cy)
                    inc_node, snap_d = snap_point_to_node(nodes_gdf, inc_geom)
                    st.success(f"📍 {inc_lat:.5f}°N, {inc_lon:.5f}°E (snapped {snap_d:.0f}m)")
                    if snap_d > 600:
                        st.warning("Far from road network — outside study area?")
                else:
                    st.info("No map click yet — click on the map first.")

            else:  # Enter coordinates
                st.session_state["click_mode"] = None
                c1, c2 = st.columns(2)
                with c1:
                    default_lat = st.session_state.get("clicked_inc_lat") or 6.565
                    inc_lat = st.number_input("Lat (°N)", value=float(default_lat),
                                              format="%.5f", step=0.001, key="inc_lat_in")
                with c2:
                    default_lon = st.session_state.get("clicked_inc_lon") or 3.375
                    inc_lon = st.number_input("Lon (°E)", value=float(default_lon),
                                              format="%.5f", step=0.001, key="inc_lon_in")
                cx, cy  = ll_to_utm(inc_lat, inc_lon)
                inc_geom = Point(cx, cy)
                inc_node, snap_d = snap_point_to_node(nodes_gdf, inc_geom)
                if snap_d > 600:
                    st.warning(f"Nearest road: {snap_d:.0f}m away")

            type_opts = ["Medical", "RTA", "Fire"]
            inc_type = st.selectbox(
                "Incident type:", type_opts,
                index=type_opts.index(inc_type) if inc_type in type_opts else 0,
            )

        with col2:
            st.markdown("#### 🚑 Vehicle Position")
            veh_mode = st.radio(
                "Vehicle location:",
                ["At dispatch station", "📍 Use my location",
                 "Click on map", "Enter coordinates"],
                key="veh_mode_sel"
            )
            veh_node = None
            veh_origin_ll = None

            if veh_mode == "📍 Use my location":
                if "geo_consented" not in st.session_state:
                    st.session_state["geo_consented"] = False
                    
                if not st.session_state["geo_consented"]:
                    st.info("ℹ️ Your live location is only used to compute the fastest route during this session.")
                    if st.button("Grant Location Access"):
                        st.session_state["geo_consented"] = True
                        st.rerun()
                else:
                    try:
                        from streamlit_geolocation import streamlit_geolocation
                        gps_data = streamlit_geolocation()
                    except ImportError:
                        st.error("Missing dependency: pip install streamlit-geolocation")
                        gps_data = None
        
                    if gps_data and gps_data.get("latitude") and gps_data.get("longitude"):
                        st.session_state["geo_lat"] = gps_data["latitude"]
                        st.session_state["geo_lon"] = gps_data["longitude"]
                        st.session_state["geo_granted"] = True
                    else:
                        st.session_state["geo_granted"] = False
        
                    if st.session_state.get("geo_granted") and st.session_state.get("geo_lat"):
                        glat = st.session_state["geo_lat"]
                        glon = st.session_state["geo_lon"]
                        st.success(f"📍 Location active: {glat:.5f}°N, {glon:.5f}°E")
                        vx, vy = ll_to_utm(glat, glon)
                        veh_node, vd = snap_point_to_node(nodes_gdf, Point(vx, vy))
                        veh_origin_ll = (glat, glon)
                        if vd > 600:
                            st.warning(f"Vehicle snapped {vd:.0f}m to nearest road")
                        else:
                            st.caption(f"✅ On road (±{vd:.0f}m)")
                    else:
                        err_msg = gps_data.get("error") if gps_data else None
                        if err_msg:
                            st.error(f"❌ GPS Error: {err_msg}")
                        else:
                            st.info("⏳ Waiting for device GPS authorization...")

            elif veh_mode == "Click on map":
                st.session_state["click_mode"] = "vehicle"
                st.markdown("""
                <div class="click-info">
                  🖱️ Set target to 'Vehicle Start Position', then click the map.
                </div>""", unsafe_allow_html=True)
                if st.session_state.get("clicked_veh_lat"):
                    vlat = st.session_state["clicked_veh_lat"]
                    vlon = st.session_state["clicked_veh_lon"]
                    vx, vy = ll_to_utm(vlat, vlon)
                    veh_node, vd = snap_point_to_node(nodes_gdf, Point(vx, vy))
                    veh_origin_ll = (vlat, vlon)
                    st.success(f"🚑 {vlat:.5f}°N, {vlon:.5f}°E (snapped {vd:.0f}m)")
                else:
                    st.info("Click the map to place the vehicle.")

            elif veh_mode == "Enter coordinates":
                if st.session_state.get("click_mode") == "vehicle":
                    st.session_state["click_mode"] = None
                vc1, vc2 = st.columns(2)
                with vc1:
                    default_vlat = st.session_state.get("clicked_veh_lat") or 6.565
                    vlat = st.number_input("Vehicle Lat", value=float(default_vlat),
                                           format="%.5f", step=0.001, key="veh_lat_in")
                with vc2:
                    default_vlon = st.session_state.get("clicked_veh_lon") or 3.370
                    vlon = st.number_input("Vehicle Lon", value=float(default_vlon),
                                           format="%.5f", step=0.001, key="veh_lon_in")
                vx, vy = ll_to_utm(vlat, vlon)
                veh_node, vd = snap_point_to_node(nodes_gdf, Point(vx, vy))
                veh_origin_ll = (vlat, vlon)
                if vd > 600:
                    st.warning(f"Vehicle snapped {vd:.0f}m to nearest road")
                else:
                    st.success(f"Vehicle on road (±{vd:.0f}m)")

        with col3:
            st.markdown("#### ⚙️ Advanced Routing")
            siren_mode = st.toggle("🚨 Siren Mode (Right-of-way)", value=False, help="Allows emergency vehicles to cautiously travel against one-way restrictions.")
            
            time_of_day = st.selectbox(
                "🕐 Time of day (Static Model):",
                ["Off-peak", "Peak morning", "Peak evening"],
                index=0,
                help="Applies time-of-day traffic congestion multipliers to the network."
            )
            
            all_facilities = stations_gdf["name"].tolist()
            unavailable_facilities = st.multiselect(
                "Mark facilities as unavailable:",
                all_facilities,
                default=st.session_state.get("unavailable_facilities", []),
                help="These facilities will be temporarily excluded from dispatch."
            )
            st.session_state["unavailable_facilities"] = unavailable_facilities

        st.markdown("---")
        go = st.button("🚀  Dispatch — Find Optimal Route",
                       use_container_width=True, type="primary",
                       disabled=(inc_node is None))

        # Let's find if simulation is already in progress.
        # If so, the start point of Leg 1 or Leg 2 should be the current vehicle location!
        sim_prog = st.session_state.get("sim_progress", 0.0)
        current_veh_node = None
        leg1_ll_s = st.session_state.get("leg1_ll", [])
        leg2_ll_s = st.session_state.get("leg2_ll", [])
        all_ll_s = leg1_ll_s + leg2_ll_s
        cur_i = 0
        if sim_prog > 0.0 and all_ll_s:
            total_pts = len(all_ll_s)
            cur_i = min(int(sim_prog * (total_pts - 1)), total_pts - 1)
            vlat, vlon = all_ll_s[cur_i]
            vx, vy = ll_to_utm(vlat, vlon)
            current_veh_node, vd = snap_point_to_node(nodes_gdf, Point(vx, vy))

        if go and inc_node is not None:
            with st.spinner("Running Time-Aware Dijkstra routing..."):
                # ── Advanced Routing Modifiers ──
                G_working = G.copy() if (siren_mode or time_of_day != "Off-peak") else G
                if G_working is not G:
                    for u, v, d in G_working.edges(data=True):
                        base_t = d.get("base_time_min", d.get("time_min", float('inf')))
                        if base_t == float('inf') and not d.get("wrong_way"):
                            continue
                            
                        # Apply time of day penalty
                        if time_of_day == "Peak morning":
                            penalty = 1.3 if d.get("highway") in ["primary", "secondary", "trunk", "motorway"] else 1.5
                            d["time_min"] = base_t * penalty
                        elif time_of_day == "Peak evening":
                            penalty = 1.4 if d.get("highway") in ["primary", "secondary", "trunk", "motorway"] else 1.6
                            d["time_min"] = base_t * penalty
                        else:
                            d["time_min"] = base_t
                            
                        # Apply Siren Mode (wrong way)
                        if d.get("wrong_way"):
                            if siren_mode:
                                d["time_min"] = d.get("base_time_min", 1.0) * 2.5 # Apply right-of-way caution penalty
                            else:
                                d["time_min"] = float('inf')
                
                # Apply Facility Availability
                active_stations = stations_gdf
                if unavailable_facilities:
                    active_stations = stations_gdf[~stations_gdf["name"].isin(unavailable_facilities)]

                if active_stations.empty:
                    st.error("🚨 All facilities of this type are marked as unavailable.")
                    result = None
                else:
                    prev_result = st.session_state.get("result")
                    if current_veh_node is not None and prev_result:
                        # Vehicle is already moving, reroute from its current location
                        is_on_leg2 = False
                        if prev_result.get("has_leg2") and len(leg1_ll_s) > 0:
                            if cur_i >= len(leg1_ll_s):
                                is_on_leg2 = True
                        
                        if is_on_leg2:
                            h_node = prev_result.get("leg2_hospital_node")
                            if h_node:
                                h_path, h_time = shortest_path(G_working, current_veh_node, h_node, blocked_edges=st.session_state["blocked_edges"])
                                if h_time < float("inf"):
                                    result = dict(prev_result)
                                    result["leg1_path"] = []
                                    result["leg1_time_min"] = 0.0
                                    result["leg2_path"] = h_path
                                    result["leg2_time_min"] = h_time
                                    result["total_time_min"] = h_time
                                else:
                                    result = None
                            else:
                                result = None
                        else:
                            veh_path, veh_time = shortest_path(G_working, current_veh_node, inc_node, blocked_edges=st.session_state["blocked_edges"])
                            if veh_time < float("inf"):
                                result = dict(prev_result)
                                result["leg1_path"] = veh_path
                                result["leg1_time_min"] = veh_time
                                result["total_time_min"] = veh_time + result.get("leg2_time_min", 0.0)
                                
                                # Recalculate Leg 2 if needed (it could also be blocked)
                                if prev_result.get("has_leg2") and prev_result.get("leg2_hospital_node"):
                                    h_node = prev_result["leg2_hospital_node"]
                                    h_path, h_time = shortest_path(G_working, inc_node, h_node, blocked_edges=st.session_state["blocked_edges"])
                                    if h_time < float("inf"):
                                        result["leg2_path"] = h_path
                                        result["leg2_time_min"] = h_time
                                        result["total_time_min"] = veh_time + h_time
                            else:
                                result = None
                    else:
                        # Standard routing from station
                        result = two_leg_route(G_working, nodes_gdf, inc_node, inc_type, active_stations, blocked_edges=st.session_state["blocked_edges"], custom_hospital_node=st.session_state.get("custom_dest_node"))

            if result is None:
                st.error("No reachable facility found — try a different location or clear active filters.")
                st.session_state["result"] = None
            elif (result.get("leg1_time_min", 0) > 0 and len(result.get("leg1_path", [])) == 0) or \
                 (result.get("leg2_time_min", 0) > 0 and len(result.get("leg2_path", [])) == 0):
                st.error("🚨 Route Connectivity Error: The generated path is broken or disconnected.")
                st.session_state["result"] = None
                result = None
            else:
                if current_veh_node is None:
                    if veh_node is not None and veh_mode not in ("At dispatch station",):
                        veh_path, veh_time = shortest_path(G_working, veh_node, inc_node, blocked_edges=st.session_state["blocked_edges"])
                        if veh_time < float("inf"):
                            result["leg1_path"]      = veh_path
                            result["leg1_time_min"]  = veh_time
                            result["total_time_min"] = veh_time + result.get("leg2_time_min", 0)
                        result["vehicle_origin_ll"] = veh_origin_ll
                    else:
                        result["vehicle_origin_ll"] = None
                        
                # Detect if Siren Mode wrong-way was used
                used_wrong_way = False
                if siren_mode:
                    all_path = result.get("leg1_path", []) + result.get("leg2_path", [])
                    for j in range(len(all_path) - 1):
                        u, v = all_path[j], all_path[j+1]
                        if G_working.has_edge(u, v) and G_working[u][v].get("wrong_way"):
                            used_wrong_way = True
                            break
                result["used_wrong_way"] = used_wrong_way
                
                # --- LIVE TRAFFIC EXTERNAL ROUTING ---
                tomtom_key = st.secrets.get("TOMTOM_API_KEY")
                if tomtom_key and result and not used_wrong_way:
                    try:
                        # Leg 1
                        sg = result.get("leg1_station_geom")
                        if current_veh_node is None and sg:
                            start_ll = utm_to_ll(sg.x, sg.y)
                        elif current_veh_node is not None:
                            start_ll = utm_to_ll(*get_node_coords(nodes_gdf, current_veh_node))
                        elif veh_node is not None and veh_mode not in ("At dispatch station",):
                            start_ll = veh_origin_ll
                        else:
                            start_ll = utm_to_ll(sg.x, sg.y) if sg else None
                            
                        end_ll = utm_to_ll(inc_geom.x, inc_geom.y) if inc_geom else None
                        
                        if start_ll and end_ll:
                            url1 = f"https://api.tomtom.com/routing/1/calculateRoute/{start_ll[0]},{start_ll[1]}:{end_ll[0]},{end_ll[1]}/json"
                            res1 = requests.get(url1, params={"key": tomtom_key, "traffic": "true", "travelMode": "car"}, timeout=5).json()
                            if "routes" in res1 and res1["routes"]:
                                r1 = res1["routes"][0]
                                result["leg1_ll_direct"] = [(pt["latitude"], pt["longitude"]) for pt in r1["legs"][0]["points"]]
                                result["leg1_time_min"] = r1["summary"]["travelTimeInSeconds"] / 60.0

                        # Leg 2
                        if result.get("has_leg2") and result.get("leg2_hospital_geom"):
                            start2_ll = end_ll
                            hg = result.get("leg2_hospital_geom")
                            end2_ll = utm_to_ll(hg.x, hg.y) if hg else None
                            if start2_ll and end2_ll:
                                url2 = f"https://api.tomtom.com/routing/1/calculateRoute/{start2_ll[0]},{start2_ll[1]}:{end2_ll[0]},{end2_ll[1]}/json"
                                res2 = requests.get(url2, params={"key": tomtom_key, "traffic": "true", "travelMode": "car"}, timeout=5).json()
                                if "routes" in res2 and res2["routes"]:
                                    r2 = res2["routes"][0]
                                    result["leg2_ll_direct"] = [(pt["latitude"], pt["longitude"]) for pt in r2["legs"][0]["points"]]
                                    result["leg2_time_min"] = r2["summary"]["travelTimeInSeconds"] / 60.0
                                    
                        result["total_time_min"] = result.get("leg1_time_min", 0) + result.get("leg2_time_min", 0)
                        result["live_traffic_active"] = True
                    except Exception as e:
                        print(f"TomTom routing failed, falling back to local NetworkX: {e}")
                        result["live_traffic_active"] = False

                st.session_state.update({
                    "result": result,
                    "inc_geom": inc_geom,
                    "inc_type": inc_type,
                    "inc_node_stored": inc_node,
                    "sim_progress": 0.0,
                    "sim_tracking": st.session_state.get("sim_tracking", False),
                    "live_nav_mode": False,
                    "live_off_route": False,
                    "last_spoken_step": None,
                    "orig_inc_node": inc_node,
                    "route_version": st.session_state.get("route_version", 0) + 1,
                })
                
                # Precompute leg1_ll and leg2_ll immediately
                exact_dest = (inc_geom.x, inc_geom.y) if inc_geom else None
                if result.get("live_traffic_active") and "leg1_ll_direct" in result:
                    st.session_state["leg1_ll"] = result["leg1_ll_direct"]
                else:
                    st.session_state["leg1_ll"] = path_to_ll_via_geometry(G, result["leg1_path"], exact_dest)
                
                if result.get("has_leg2"):
                    if result.get("live_traffic_active") and "leg2_ll_direct" in result:
                        raw_leg2 = result["leg2_ll_direct"]
                    elif result.get("leg2_path"):
                        hg = result["leg2_hospital_geom"]
                        exact_h_coord = (hg.x, hg.y) if hg else None
                        raw_leg2 = path_to_ll_via_geometry(G, result["leg2_path"], exact_h_coord)
                    else:
                        raw_leg2 = []
                    # Add microscopic geographical offset (+0.00005 deg) so Leg 2 does not Z-fight or overlap exactly in 3D
                    st.session_state["leg2_ll"] = [[lat + 0.00005, lon - 0.00005] for lat, lon in raw_leg2]
                else:
                    st.session_state["leg2_ll"] = []



    # ── MAIN LAYOUT ────────────────────────────────────────────────────────────
    map_col, panel_col = st.columns([2.5, 1], gap="small")

    with panel_col:
        # Dynamic Rerouting Panel
        st.markdown("---")
        st.markdown("### 🚧 Dynamic Rerouting")
        st.markdown("<div style='font-size:0.75rem;color:#94a3b8;margin-bottom:8px'>Simulate blocked roads to instantly reroute.</div>", unsafe_allow_html=True)
        
        current_route_edges = []
        res = st.session_state.get("result")
        if res and res.get("leg1_path"):
            p = res["leg1_path"]
            for i in range(len(p)-1):
                u, v = p[i], p[i+1]
                ed_dict = G.get_edge_data(u, v) or {}
                ed = ed_dict[0] if 0 in ed_dict else ed_dict
                ename = ed.get("name", "") or "Unnamed Road"
                if ename and ename != "Unnamed Road":
                    current_route_edges.append((u, v, ename))
        
        if current_route_edges:
            edge_labels = [f"{name} (Nodes: {u}-{v})" for u, v, name in current_route_edges[:10]]
            edge_map = {f"{name} (Nodes: {u}-{v})": (u, v) for u, v, name in current_route_edges[:10]}
            blocked_sel = st.selectbox("Block upcoming street:", ["-- Select Street --"] + edge_labels)
            if st.button("Block & Reroute", use_container_width=True) and blocked_sel != "-- Select Street --":
                u, v = edge_map[blocked_sel]
                st.session_state["blocked_edges"].append((u, v))
                st.session_state["route_version"] += 1
                go = True  # force recalculation
        
        if st.session_state.get("blocked_edges"):
            st.markdown(f"<div style='color:#ef4444;font-size:0.7rem;'>Currently blocked: {len(st.session_state['blocked_edges'])} road(s)</div>", unsafe_allow_html=True)
            if st.button("Clear Blocks"):
                st.session_state["blocked_edges"] = []
                # Not forcing auto reroute here, wait for click
                
        result       = st.session_state.get("result")
        inc_geom     = st.session_state.get("inc_geom") or inc_geom
        inc_type_disp = st.session_state.get("inc_type") or inc_type

        if result:
            l1t = result["leg1_time_min"]
            l2t = result.get("leg2_time_min", 0.0)
            tot = result["total_time_min"]
            rating_txt, rating_cls = response_rating(l1t)

            sg_geom = result["leg1_station_geom"]
            sl_t    = straight_line_time(sg_geom, inc_geom, 12.0) if inc_geom else 0
            ovhd    = ((l1t - sl_t) / sl_t * 100) if sl_t > 0 else 0

            kpi_cls   = "green" if l1t <= 5 else "amber" if l1t <= 10 else "red"
            badge_cls = {"Medical":"badge-medical","RTA":"badge-rta","Fire":"badge-fire"}.get(inc_type_disp,"badge-medical")
            is_custom = bool(result.get("vehicle_origin_ll"))

            # Build badge HTML as variables to avoid 4-space markdown code-block bug
            _badge_type   = f'<span class="badge {badge_cls}">{inc_type_disp}</span>'
            _badge_custom = '<span class="badge badge-geo">Custom Vehicle</span>' if is_custom else ''
            _badge_siren  = '<span class="badge badge-rta" style="background:#dc2626;color:#fee2e2">⚠️ Wrong Way Used</span>' if result.get("used_wrong_way") else ''
            _badge_traffic = '<span class="badge badge-live" style="background:#064e3b;color:#6ee7b7">🟢 Live Traffic Routing</span>' if result.get("live_traffic_active") else ''
            _leg_label    = "2-Leg" if result["has_leg2"] else "1-Leg"
            _badge_leg    = f'<span class="badge badge-medical" style="background:#1e3a5f;color:#90cdf4">{_leg_label} Response</span>'
            _kpi2 = (
                f'<div class="kpi-card amber">'
                f'<div class="kpi-val">{l2t:.2f}<span class="kpi-label"> min</span></div>'
                f'<div class="kpi-label">Leg 2 — Hospital</div></div>'
            ) if result["has_leg2"] else ""

            _result_html = (
                f'<div class="result-panel">'
                f'<div style="margin-bottom:0.5rem">{_badge_type}{_badge_custom}{_badge_siren}{_badge_traffic}{_badge_leg}</div>'
                f'<h4 style="margin:0 0 0.4rem;font-size:.75rem;color:#90cdf4">RESPONSE TIMES (TCF=2.2 Adjusted)</h4>'
                f'<div class="kpi-row">'
                f'<div class="kpi-card {kpi_cls}">'
                f'<div class="kpi-val">{l1t:.2f}<span class="kpi-label"> min</span></div>'
                f'<div class="kpi-label">Leg 1 — Scene arrival</div></div>'
                f'{_kpi2}</div>'
                f'<div class="kpi-row">'
                f'<div class="kpi-card purple">'
                f'<div class="kpi-val">{tot:.2f}<span class="kpi-label"> min</span></div>'
                f'<div class="kpi-label">Total chain</div></div>'
                f'<div class="kpi-card">'
                f'<div class="kpi-val">{sl_t:.2f}<span class="kpi-label"> min</span></div>'
                f'<div class="kpi-label">Straight-line ref</div></div></div>'
                f'<span class="rating-pill {rating_cls}">{rating_txt}</span>'
                f'<span style="font-size:.68rem;color:#64748b;margin-left:.4rem">overhead {ovhd:+.1f}%</span>'
                f'</div>'
            )
            st.markdown(_result_html, unsafe_allow_html=True)

            # Leg 1 card
            _leg1_title = "Vehicle" if is_custom else "Dispatch"
            _leg1_html = (
                f'<div class="leg-card leg1">'
                f'<div class="leg-title blue">🔵 LEG 1 — {_leg1_title} → Scene</div>'
                f'<div class="leg-name">{result["leg1_station_name"]}</div>'
                f'<div><span class="leg-time">{l1t:.2f}</span>'
                f'<span class="leg-unit">min · {len(result["leg1_path"])} nodes</span></div>'
                f'</div>'
            )
            st.markdown(_leg1_html, unsafe_allow_html=True)

            if result["has_leg2"]:
                _leg2_html = (
                    f'<div style="text-align:center;color:rgba(255,255,255,.15);font-size:.82rem;margin:.15rem 0">↓ patient loaded ↓</div>'
                    f'<div class="leg-card leg2">'
                    f'<div class="leg-title orange">🟠 LEG 2 — Scene → Hospital</div>'
                    f'<div class="leg-name">{result["leg2_hospital_name"]}</div>'
                    f'<div><span class="leg-time">{l2t:.2f}</span>'
                    f'<span class="leg-unit">min · {len(result["leg2_path"])} nodes</span></div>'
                    f'</div>'
                )
                st.markdown(_leg2_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="leg-card fire">'
                    '<div class="leg-title red">🔴 Fire — Scene only (no patient transport)</div>'
                    '</div>',
                    unsafe_allow_html=True)

            dirs1 = generate_directions(G, result["leg1_path"])
            dirs2 = (generate_directions(G, result["leg2_path"])
                     if result.get("has_leg2") and result.get("leg2_path") else [])
            all_dirs = dirs1 + dirs2

            st.markdown("---")
            show_dir = st.toggle("📋 Turn-by-turn directions", value=False, key="show_dirs")
            if show_dir:
                st.markdown(
                    f'<div class="dir-box">{format_directions_html(dirs1, "LEG 1 — Dispatch to Scene")}</div>',
                    unsafe_allow_html=True)
                if dirs2:
                    st.markdown(
                        f'<div class="dir-box" style="margin-top:.45rem">'
                        f'{format_directions_html(dirs2, "LEG 2 — Scene to Hospital")}</div>',
                        unsafe_allow_html=True)

            # ── PRIORITY 4: Mode selector — Live Nav vs Journey Simulation ────
            st.markdown("---")
            has_geo = st.session_state.get("geo_granted") and st.session_state.get("geo_lat")
            nav_modes = ["🎬 Journey Simulation — manual preview of computed route"]
            if has_geo:
                nav_modes.insert(0, "🧭 Live Navigation Mode — uses your device's real GPS position")
            nav_mode_sel = st.radio("Navigation mode:", nav_modes, index=0, key="nav_mode_radio")
            live_mode = "Live Navigation Mode" in nav_mode_sel

            # ── LIVE NAVIGATION MODE (Priority 4) ────────────────────────────
            if live_mode:
                st.markdown(
                    '<div class="live-panel">'
                    '<div style="font-size:.68rem;font-weight:700;color:#34d399;text-transform:uppercase;'
                    'letter-spacing:.08em;margin-bottom:.4rem">🧭 Live Navigation Mode — Real GPS Tracking</div>'
                    '<div style="font-size:.65rem;color:#6ee7b7;margin-bottom:.5rem">'
                    'Vehicle position updated from your device GPS every 4 seconds.</div>',
                    unsafe_allow_html=True)

                if not st.session_state["live_nav_mode"]:
                    if st.button("▶ Start Live Navigation", use_container_width=True, key="btn_live_start"):
                        st.session_state["live_nav_mode"] = True
                        st.session_state["live_off_route"] = False
                        st.rerun()
                else:
                    if st.button("⏹ Stop Live Navigation", use_container_width=True, key="btn_live_stop"):
                        st.session_state["live_nav_mode"] = False
                        st.session_state["camera_follow"] = True
                        st.rerun()

                    # ── On each autorefresh: read GPS and update progress ────
                    live_lat = st.session_state.get("geo_lat")
                    live_lon = st.session_state.get("geo_lon")

                    if live_lat and live_lon:
                        st.session_state["live_veh_lat"] = live_lat
                        st.session_state["live_veh_lon"] = live_lon

                        all_route_ll = (st.session_state.get("leg1_ll", []) +
                                        st.session_state.get("leg2_ll", []))

                        if all_route_ll:
                            near_i, near_d = _nearest_route_point(
                                (live_lat, live_lon), all_route_ll
                            )
                            live_prog = near_i / max(len(all_route_ll) - 1, 1)
                            st.session_state["sim_progress"] = live_prog

                            OFF_ROUTE_THRESH = 150  # metres
                            
                            import time
                            cur_time = time.time()
                            if "last_reroute_time" not in st.session_state:
                                st.session_state["last_reroute_time"] = cur_time
                                
                            needs_reroute = False
                            if near_d > OFF_ROUTE_THRESH:
                                needs_reroute = True
                                st.session_state["live_off_route"] = True
                            elif cur_time - st.session_state["last_reroute_time"] > 60:
                                needs_reroute = True
                                st.session_state["last_reroute_time"] = cur_time
                                
                            if needs_reroute:
                                # Re-route from current position to original incident
                                orig_inc = st.session_state.get("orig_inc_node")
                                if orig_inc:
                                    tomtom_key = st.secrets.get("TOMTOM_API_KEY")
                                    tomtom_success = False
                                    if tomtom_key:
                                        try:
                                            # We need original incident Lat/Lon.
                                            inc_geom_tmp = st.session_state.get("inc_geom")
                                            if inc_geom_tmp:
                                                end_ll = utm_to_ll(inc_geom_tmp.x, inc_geom_tmp.y)
                                                url = f"https://api.tomtom.com/routing/1/calculateRoute/{live_lat},{live_lon}:{end_ll[0]},{end_ll[1]}/json"
                                                res = requests.get(url, params={"key": tomtom_key, "traffic": "true", "travelMode": "car"}, timeout=5).json()
                                                if "routes" in res and res["routes"]:
                                                    r = res["routes"][0]
                                                    new_result = dict(result)
                                                    pts = [(pt["latitude"], pt["longitude"]) for pt in r["legs"][0]["points"]]
                                                    new_result["leg1_ll_direct"] = pts
                                                    new_result["leg1_time_min"] = r["summary"]["travelTimeInSeconds"] / 60.0
                                                    new_result["total_time_min"] = new_result["leg1_time_min"] + new_result.get("leg2_time_min", 0)
                                                    new_result["leg1_station_name"] = "Live Traffic Re-route"
                                                    new_result["vehicle_origin_ll"] = (live_lat, live_lon)
                                                    new_result["live_traffic_active"] = True
                                                    
                                                    st.session_state["result"] = new_result
                                                    st.session_state["leg1_ll"] = pts
                                                    result = new_result
                                                    st.session_state["live_off_route"] = False
                                                    tomtom_success = True
                                        except Exception:
                                            pass
                                            
                                    if not tomtom_success:
                                        lx, ly = ll_to_utm(live_lat, live_lon)
                                        cur_node, _ = snap_point_to_node(nodes_gdf, Point(lx, ly))
                                        repath, retime = shortest_path(G, cur_node, orig_inc)
                                        if retime < float("inf"):
                                            new_result = dict(result)
                                            new_result["leg1_path"]      = repath
                                            new_result["leg1_time_min"]  = retime
                                            new_result["total_time_min"] = (retime +
                                                new_result.get("leg2_time_min", 0))
                                            new_result["leg1_station_name"] = "Live Re-route"
                                            new_result["vehicle_origin_ll"] = (live_lat, live_lon)
                                            st.session_state["result"] = new_result
                                            st.session_state["leg1_ll"] = path_to_ll_via_geometry(G, repath)
                                            result = new_result
                                            st.session_state["live_off_route"] = False
                            else:
                                st.session_state["live_off_route"] = False

                    if st.session_state.get("live_off_route"):
                        st.warning("⚠️ Off route — recalculating…")
                        
                    # Calculate real vehicle speed dynamically
                    # time module already imported at top level
                    cur_time = time.time()
                    live_speed = None
                    if live_lat and live_lon:
                        if "last_gps_time" in st.session_state:
                            dt = cur_time - st.session_state["last_gps_time"]
                            if dt > 1.0: # At least 1 second passed
                                p_lat = st.session_state["last_gps_lat"]
                                p_lon = st.session_state["last_gps_lon"]
                                px, py = ll_to_utm(p_lat, p_lon)
                                cx, cy = ll_to_utm(live_lat, live_lon)
                                dist_m = math.sqrt((cx-px)**2 + (cy-py)**2)
                                live_speed = (dist_m / dt) * 3.6 # m/s to km/h
                        
                        st.session_state["last_gps_lat"] = live_lat
                        st.session_state["last_gps_lon"] = live_lon
                        st.session_state["last_gps_time"] = cur_time
                        
                    if live_speed is not None:
                        prev_speed = st.session_state.get("live_speed_kmh", 40.0)
                        if live_speed < 160: # Ignore absurd GPS jumps
                            st.session_state["live_speed_kmh"] = prev_speed * 0.7 + live_speed * 0.3
                            
                    prog = st.session_state["sim_progress"]
                    eff_speed_kmh = st.session_state.get("live_speed_kmh", 40.0)
                    rem_dist, rem_time = compute_remaining_time_and_dist(result, prog, G, live_speed_kmh=eff_speed_kmh, all_ll=all_route_ll)

                    livecols = st.columns(3)
                    with livecols[0]:
                        st.markdown(f'<div class="sim-stat"><div class="live-val">{rem_dist/1000:.2f}</div>'
                                    f'<div class="sim-label">km left</div></div>', unsafe_allow_html=True)
                    with livecols[1]:
                        st.markdown(f'<div class="sim-stat"><div class="live-val">{rem_time:.1f}</div>'
                                    f'<div class="sim-label">min left</div></div>', unsafe_allow_html=True)
                    with livecols[2]:
                        st.markdown(f'<div class="sim-stat"><div class="live-val">{prog*100:.0f}%</div>'
                                    f'<div class="sim-label">done</div></div>', unsafe_allow_html=True)

                    if live_lat:
                        st.markdown(
                            f"<div style='font-size:.7rem;color:#6ee7b7;margin-top:.3rem'>"
                            f"📍 Live GPS: <b>{live_lat:.5f}°N, {live_lon:.5f}°E</b></div>",
                            unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # ── JOURNEY SIMULATION DETAILS (READ-ONLY) ────────────────────────
            else:
                st.markdown(
                    '<div class="sim-panel">'
                    '<div style="font-size:.68rem;font-weight:700;color:#a78bfa;text-transform:uppercase;'
                    'letter-spacing:.08em;margin-bottom:.4rem">🎬 Journey simulation stats</div>'
                    '<div style="font-size:.65rem;color:#475569;margin-bottom:.5rem">'
                    'Real-time simulation status (use controls above map)</div>',
                    unsafe_allow_html=True)

                prog = st.session_state.get("sim_progress", 0.0)
                leg1_ll_s = st.session_state.get("leg1_ll", [])
                leg2_ll_s = st.session_state.get("leg2_ll", [])
                all_ll_s  = leg1_ll_s + leg2_ll_s
                total_pts = max(len(all_ll_s), 1)

                leg1_utm = path_to_detailed_coords(G, result["leg1_path"])
                leg2_utm = (path_to_detailed_coords(G, result["leg2_path"])
                            if result.get("has_leg2") and result.get("leg2_path") else [])
                all_utm  = leg1_utm + leg2_utm
                total_dist_m = route_total_distance(all_utm) if all_utm else 0
                rem_dist = total_dist_m * (1 - prog)
                rem_time = tot * (1 - prog)

                cur_street = ""
                path_len   = len(result["leg1_path"])
                if path_len >= 2:
                    eidx = min(int(prog * (path_len - 1)), path_len - 2)
                    u, v = result["leg1_path"][eidx], result["leg1_path"][eidx+1]
                    ed_dict = G.get_edge_data(u, v) or {}
                    ed = ed_dict[0] if 0 in ed_dict else ed_dict
                    cur_street = ed.get("name", "") or "unnamed road"

                cur_idx = min(int(prog * (total_pts - 1)), total_pts - 1)
                on_leg  = "LEG 1 — En route to scene" if cur_idx < len(leg1_ll_s) else "LEG 2 — Transporting patient"

                # Speech Synthesis removed from stats sidebar

                cols3 = st.columns(3)
                with cols3[0]:
                    st.markdown(f'<div class="sim-stat"><div class="sim-val">{rem_dist/1000:.2f}</div>'
                                f'<div class="sim-label">km left</div></div>', unsafe_allow_html=True)
                with cols3[1]:
                    st.markdown(f'<div class="sim-stat"><div class="sim-val">{rem_time:.1f}</div>'
                                f'<div class="sim-label">min left</div></div>', unsafe_allow_html=True)
                with cols3[2]:
                    st.markdown(f'<div class="sim-stat"><div class="sim-val">{prog*100:.0f}%</div>'
                                f'<div class="sim-label">done</div></div>', unsafe_allow_html=True)

                st.markdown(
                    f"<div style='font-size:.7rem;color:#94a3b8;margin-top:.35rem'>"
                    f"📍 <b>{on_leg}</b><br>"
                    f"🛣️ <i>{cur_street or 'unnamed road'}</i></div>",
                    unsafe_allow_html=True)

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    is_trk = st.session_state["sim_tracking"]
                    trk_label = "⏸ Pause" if is_trk else "▶ Start Tracking"
                    if st.button(trk_label, use_container_width=True, key="btn_trk"):
                        st.session_state["sim_tracking"] = not is_trk
                        if not is_trk:
                            st.session_state["recenter_trigger"] = True
                        st.rerun()
                with bc2:
                    if st.button("↺ Reset", use_container_width=True, key="btn_rst"):
                        st.session_state["sim_progress"]    = 0.0
                        st.session_state["sim_tracking"]    = False
                        st.session_state["last_spoken_step"] = None
                        st.session_state["custom_dest_node"] = None
                        st.session_state["clicked_dest_lat"] = None
                        st.session_state["clicked_dest_lon"] = None
                        st.session_state["map_clicked_lat"] = None
                        st.session_state["map_clicked_lon"] = None
                        st.rerun()
                with bc3:
                    if st.button("🎯 Recenter", use_container_width=True, key="btn_rcnt"):
                        st.session_state["recenter_trigger"] = True
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-panel" style="text-align:center;padding:2.5rem 1rem">
              <div style="font-size:2.5rem;margin-bottom:.8rem">🚑</div>
              <div style="color:#475569;font-size:.84rem">
                Set an incident location and click<br>
                <b style="color:#90cdf4">Dispatch — Find Optimal Route</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:.7rem 0'>",
                    unsafe_allow_html=True)


    # ── MAP DISPLAY ───────────────────────────────────────────────────────────
    with map_col:
        click_mode = st.session_state.get("click_mode")
        leg1_ll_s  = st.session_state.get("leg1_ll", [])
        leg2_ll_s  = st.session_state.get("leg2_ll", [])
        all_ll_s   = leg1_ll_s + leg2_ll_s
        sim_prog   = st.session_state.get("sim_progress", 0.0)

        vehicle_cur_ll = None
        veh_bearing    = 30.0
        card_dir       = "N"
        cur_street     = ""

        is_navigating = st.session_state.get("live_nav_mode") or sim_prog > 0

        # For live mode: use real GPS position as vehicle
        if st.session_state.get("live_nav_mode") and st.session_state.get("live_veh_lat"):
            vehicle_cur_ll = (st.session_state["live_veh_lat"], st.session_state["live_veh_lon"])
        elif all_ll_s and len(all_ll_s) >= 2:
            total_pts = len(all_ll_s)
            cur_i = min(int(sim_prog * (total_pts - 1)), total_pts - 1)
            vehicle_cur_ll = all_ll_s[cur_i]
            next_i = min(cur_i + 1, total_pts - 1)
            if cur_i < next_i:
                lat1, lon1 = all_ll_s[cur_i]
                lat2, lon2 = all_ll_s[next_i]
                x1, y1 = ll_to_utm(lat1, lon1)
                x2, y2 = ll_to_utm(lat2, lon2)
                veh_bearing = bearing_deg(x1, y1, x2, y2)
                card_dir    = cardinal_direction(veh_bearing)
                
        exact_dest = (inc_geom.x, inc_geom.y) if inc_geom else None

        # ── REROUTE TRIGGER (Priority 4) ──
        if st.session_state.get("trigger_reroute") and vehicle_cur_ll:
            st.session_state["trigger_reroute"] = False
            orig_inc = st.session_state.get("orig_inc_node")
            if orig_inc:
                lx, ly = ll_to_utm(vehicle_cur_ll[0], vehicle_cur_ll[1])
                cur_node, _ = snap_point_to_node(nodes_gdf, Point(lx, ly))
                repath, retime = shortest_path(G, cur_node, orig_inc, st.session_state.get("blocked_edges", []))
                if retime < float("inf"):
                    new_result = dict(st.session_state["result"])
                    new_result["leg1_path"] = repath
                    new_result["leg1_time_min"] = retime
                    new_result["total_time_min"] = retime + new_result.get("leg2_time_min", 0)
                    new_result["leg1_station_name"] = "Live Re-route"
                    new_result["vehicle_origin_ll"] = vehicle_cur_ll
                    st.session_state["result"] = new_result
                    st.session_state["leg1_ll"] = path_to_ll_via_geometry(G, repath, exact_dest_coord=exact_dest)
                    st.session_state["sim_progress"] = 0.0
                    st.session_state["route_version"] = st.session_state.get("route_version", 0) + 1
                    sim_prog = 0.0
                    st.rerun()

        # Find current street for navigation banner
        if is_navigating and st.session_state.get("result"):
            res = st.session_state["result"]
            path_len = len(res["leg1_path"])
            if path_len >= 2:
                eidx = min(int(sim_prog * (path_len - 1)), path_len - 2)
                u, v = res["leg1_path"][eidx], res["leg1_path"][eidx+1]
                ed = G.get_edge_data(u, v)
                if ed and 0 in ed and 'name' in ed[0]:
                    cur_street = ed[0]['name']

        show_inc_ll = (inc_lat, inc_lon) if (inc_lat and inc_lon and inc_mode != "Preset list") else None
        show_veh_ll = (vehicle_cur_ll if vehicle_cur_ll
                       else (veh_origin_ll if veh_mode not in ("At dispatch station",) else None))

        recenter_ll = None
        if st.session_state.get("recenter_trigger"):
            recenter_ll = show_veh_ll if show_veh_ll else [6.565, 3.375]
            st.session_state["recenter_trigger"] = False



        # Clean control panel above tabs/map
        if st.session_state.get("result"):
            prog_val = st.session_state.get("sim_progress", 0.0)
            cur_street_nav = ""
            res_val = st.session_state["result"]
            path_len_nav = len(res_val["leg1_path"])
            if path_len_nav >= 2:
                eidx_nav = min(int(sim_prog * (path_len_nav - 1)), path_len_nav - 2)
                u_nav, v_nav = res_val["leg1_path"][eidx_nav], res_val["leg1_path"][eidx_nav+1]
                ed_nav = G.get_edge_data(u_nav, v_nav)
                if ed_nav and 0 in ed_nav and 'name' in ed_nav[0]:
                    cur_street_nav = ed_nav[0]['name']

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(13,31,56,0.95), rgba(6,20,38,0.95)); padding:16px 24px; border-radius:12px; margin-bottom:16px; border:1px solid rgba(52,211,153,0.3); border-left:5px solid #10b981; box-shadow: 0 8px 24px rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:space-between; backdrop-filter: blur(10px);">
              <div>
                <div style="display:flex; align-items:center; gap:8px;">
                  <div style="background:rgba(16,185,129,0.2); padding:4px 8px; border-radius:6px; color:#34d399; font-size:0.7rem; font-weight:800; text-transform:uppercase; letter-spacing:0.1em;">Live Nav</div>
                  <span style="color:#94a3b8; font-size:0.8rem; font-weight:500;">Current Route</span>
                </div>
                <div style="color:#f8fafc; font-size:1.4rem; font-weight:800; margin-top:6px; text-shadow: 0 2px 4px rgba(0,0,0,0.3); display:flex; align-items:center; gap:8px;">
                  <span style="font-size:1.6rem; filter: drop-shadow(0 2px 2px rgba(0,0,0,0.5));">📍</span> {cur_street_nav if cur_street_nav else 'Proceed on Route'}
                </div>
              </div>
              <div style="text-align:right; background:rgba(0,0,0,0.4); padding:10px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);">
                <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:2px;">Heading</div>
                <div style="font-size:1.6rem; font-weight:800; color:#38bdf8; line-height:1; display:flex; align-items:center; gap:6px;">
                  <div style="transform:rotate({veh_bearing:.1f}deg); font-size:1.4rem; transition: transform 0.3s ease;">🧭</div>
                  {card_dir}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            is_trk = st.session_state.get("sim_tracking", False)
            trk_label = "⏸ Pause Tracking" if is_trk else "▶ Start Tracking"

            # prominent control card
            st.markdown("""
            <style>
            .control-box {
                background: rgba(13, 25, 48, 0.95);
                border: 1px solid rgba(99, 179, 237, 0.35);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            }
            </style>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("<div class='control-box'>", unsafe_allow_html=True)
                c_btn1, c_btn2, c_btn3, c_btn4, c_btn5 = st.columns([2, 1, 1, 1.5, 1.5])
                with c_btn1:
                    if st.button(trk_label, use_container_width=True, type="primary" if not is_trk else "secondary", key="map_ctrl_trk"):
                        st.session_state["sim_tracking"] = not is_trk
                        if not is_trk:
                            import time
                            st.session_state["sim_last_update"] = time.time()
                            st.session_state["recenter_trigger"] = True
                        st.rerun()
                with c_btn2:
                    if st.button("↺ Reset", use_container_width=True, key="map_ctrl_rst"):
                        st.session_state["sim_progress"] = 0.0
                        st.session_state["sim_tracking"] = False
                        st.session_state["last_spoken_step"] = None
                        st.session_state["custom_dest_node"] = None
                        st.session_state["clicked_dest_lat"] = None
                        st.session_state["clicked_dest_lon"] = None
                        st.session_state["map_clicked_lat"] = None
                        st.session_state["map_clicked_lon"] = None
                        if "sim_last_update" in st.session_state:
                            del st.session_state["sim_last_update"]
                        st.rerun()
                with c_btn3:
                    if st.button("🎯 Recenter", use_container_width=True, key="map_ctrl_rcnt"):
                        st.session_state["recenter_trigger"] = True
                        st.rerun()
                with c_btn4:
                    voice_on = st.toggle("🔊 Voice guidance", value=st.session_state.get("voice_enabled", False), key="map_ctrl_voice_toggle")
                    st.session_state["voice_enabled"] = voice_on
                    
                    camera_follow = st.toggle("🔒 Follow vehicle", value=st.session_state.get("camera_follow", True), key="map_ctrl_camera_toggle")
                    st.session_state["camera_follow"] = camera_follow
                with c_btn5:
                    app_fullscreen = st.toggle("⛶ Fullscreen App", value=st.session_state.get("app_fullscreen", False), key="map_ctrl_fs_toggle")
                    st.session_state["app_fullscreen"] = app_fullscreen
                    if app_fullscreen:
                        st.markdown("""
                        <style>
                        .main .block-container { max-width: 100% !important; padding-top: 0 !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
                        [data-testid="stSidebar"] { display: none !important; }
                        header[data-testid="stHeader"] { display: none !important; }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    sim_speed = st.selectbox(
                        "⚡ Speed",
                        options=[1, 2, 5, 10, 20, 30],
                        index=2, # Default is 5x
                        format_func=lambda x: f"{x}x",
                        key="sim_speed_sel"
                    )

                if is_trk:
                    st.markdown("---")
                    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#cbd5e1; margin-bottom:8px'>🚧 Report Blockage Ahead</div>", unsafe_allow_html=True)
                    bc_col1, bc_col2 = st.columns([3, 1])
                    with bc_col1:
                        upcoming = []
                        if st.session_state.get("result") and all_dirs:
                            d_idx = min(int(prog_val * (len(all_dirs) - 1)), len(all_dirs) - 1)
                            for d in all_dirs[d_idx:d_idx+6]:
                                sname = d.get("street_name")
                                if sname and sname not in upcoming and sname != "unnamed road":
                                    upcoming.append(sname)
                        block_target = st.selectbox("Select street to block", upcoming if upcoming else ["No named streets ahead"], label_visibility="collapsed")
                    with bc_col2:
                        if st.button("Block & Reroute", use_container_width=True, type="primary"):
                            if upcoming and block_target != "No named streets ahead":
                                res = st.session_state.get("result")
                                if res and "leg1_path" in res:
                                    path = res["leg1_path"]
                                    edges_to_block = []
                                    for i in range(len(path)-1):
                                        u, v = path[i], path[i+1]
                                        ed = G.get_edge_data(u, v)
                                        if ed and 0 in ed and ed[0].get('name') == block_target:
                                            edges_to_block.append((u, v))
                                            edges_to_block.append((v, u))
                                    if edges_to_block:
                                        if "blocked_edges" not in st.session_state:
                                            st.session_state["blocked_edges"] = []
                                        st.session_state["blocked_edges"].extend(edges_to_block)
                                        st.session_state["trigger_reroute"] = True
                                        st.rerun()

                # Slider (Part B Fix: Accurate percentage sync mapping)
                # We do NOT use 'key' here so that we can programmatically update the slider's value via the background task.
                current_prog = st.session_state.get("sim_progress", 0.0)
                prog_val_ui = st.slider("Vehicle Route Progress", 0, 100,
                                        value=int(current_prog * 100),
                                        step=1, format="%d%%")
                
                # Only update session state if the user manually dragged the slider (i.e. slider value differs from current percent)
                if prog_val_ui != int(current_prog * 100):
                    st.session_state["sim_progress"] = prog_val_ui / 100.0
                prog_val = st.session_state["sim_progress"]
                # Speak direction if voice enabled
                if voice_on and all_dirs:
                    d_idx = min(int(prog_val * (len(all_dirs) - 1)), len(all_dirs) - 1)
                    cur_dir_step = all_dirs[d_idx]
                    if cur_dir_step["step"] != st.session_state.get("last_spoken_step"):
                        st.session_state["last_spoken_step"] = cur_dir_step["step"]
                        spk_text = cur_dir_step["instruction"].replace("'", "\\'")
                        st.components.v1.html(
                            f"""<script>
                            if ('speechSynthesis' in window) {{
                                window.speechSynthesis.cancel();
                                var msg = new SpeechSynthesisUtterance('{spk_text}');
                                msg.rate = 0.95;
                                window.speechSynthesis.speak(msg);
                            }}
                            </script>""", height=0, width=0)
                st.markdown("</div>", unsafe_allow_html=True)

        # 🔍 Search & Pin Address on Map (Moved to map column top)
        search_query = st_keyup("🔍 Search Address or Place in Lagos (e.g. Yaba, Ikeja, Lekki):", key="loc_search_query", debounce=500)
        
        if search_query and len(search_query.strip()) >= 2:
            sq_lower = search_query.strip().lower()
            # Fast local dictionary lookup
            local_matches = []
            for k, v in LAGOS_AREAS.items():
                if sq_lower in k.lower():
                    local_matches.append({
                        "place_id": f"local_{k.replace(' ', '')}",
                        "display_name": v["display_name"],
                        "lat": v["lat"],
                        "lon": v["lon"]
                    })
            
            # If not enough local matches, query Nominatim (cached via session state)
            nominatim_results = []
            if len(sq_lower) >= 4:
                cache_key = f"nominatim_{sq_lower}"
                if cache_key not in st.session_state:
                    with st.spinner("Searching OSM map..."):
                        st.session_state[cache_key] = geocode_lagos(search_query)
                nominatim_results = st.session_state[cache_key] or []
            
            # Combine
            all_matches = local_matches + [
                {
                    "place_id": f"osm_{r['place_id']}",
                    "display_name": r["display_name"],
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"])
                }
                for r in nominatim_results
            ]
            
            # Unique by display_name
            seen = set()
            unique_matches = []
            for m in all_matches:
                if m["display_name"] not in seen:
                    seen.add(m["display_name"])
                    unique_matches.append(m)
            
            # Render Matches
            if unique_matches:
                st.markdown("<div style='margin-bottom:8px; color:#94a3b8; font-size:0.8rem;'>Suggestions (Click to pin on map):</div>", unsafe_allow_html=True)
                cols = st.columns(min(len(unique_matches), 4))
                for i, r in enumerate(unique_matches[:4]):
                    dname = r['display_name'].split(",")[0]
                    cx, cy = ll_to_utm(float(r["lat"]), float(r["lon"]))
                    _, snap_dist = snap_point_to_node(nodes_gdf, Point(cx, cy))
                    in_study = snap_dist < 6000
                    
                    if in_study:
                        if cols[i].button(f"📍 {dname}", key=f"search_set_{r['place_id']}", use_container_width=True):
                            st.session_state["map_clicked_lat"] = float(r["lat"])
                            st.session_state["map_clicked_lon"] = float(r["lon"])
                            st.session_state["loc_search_query"] = "" # Clear search
                            st.rerun()
                    else:
                        cols[i].button(f"❌ {dname} (Out of Bounds)", key=f"search_set_{r['place_id']}", use_container_width=True, disabled=True, help="This location is outside our study area in Lagos.")
            else:
                st.info("No matching locations found in Lagos.")
        
        # 📍 Map Click / Search Pinning Prompt
        if st.session_state.get("map_clicked_lat") and st.session_state.get("map_clicked_lon"):
            mlat = st.session_state["map_clicked_lat"]
            mlon = st.session_state["map_clicked_lon"]
            st.markdown(f"""
            <div style="background: rgba(30, 58, 138, 0.95); padding: 12px 18px; border-radius: 10px; border: 1.5px solid #3b82f6; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <div style="color: #93c5fd; font-weight: 700; font-size: 0.85rem; margin-bottom: 2px;">
                    📍 Selected Location: <b>{mlat:.5f}°N, {mlon:.5f}°E</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            cb1, cb2, cb3, cb4 = st.columns(4)
            if cb1.button("🚨 Set as Incident", key="btn_set_inc", use_container_width=True, type="primary"):
                st.session_state["clicked_inc_lat"] = mlat
                st.session_state["clicked_inc_lon"] = mlon
                st.session_state["inc_mode_sel"] = "Click on map"
                st.session_state["map_clicked_lat"] = None
                st.session_state["map_clicked_lon"] = None
                st.rerun()
            if cb2.button("🚑 Set as Vehicle Start", key="btn_set_veh", use_container_width=True, type="primary"):
                st.session_state["clicked_veh_lat"] = mlat
                st.session_state["clicked_veh_lon"] = mlon
                st.session_state["veh_mode_sel"] = "Click on map"
                st.session_state["map_clicked_lat"] = None
                st.session_state["map_clicked_lon"] = None
                st.rerun()
            if cb3.button("🏥 Set as Custom Destination", key="btn_set_dest", use_container_width=True, type="primary"):
                st.session_state["clicked_dest_lat"] = mlat
                st.session_state["clicked_dest_lon"] = mlon
                dx, dy = ll_to_utm(mlat, mlon)
                dest_node, _ = snap_point_to_node(nodes_gdf, Point(dx, dy))
                st.session_state["custom_dest_node"] = dest_node
                st.session_state["map_clicked_lat"] = None
                st.session_state["map_clicked_lon"] = None
                st.rerun()
            if cb4.button("❌ Dismiss / Clear Pins", key="btn_dismiss_click", use_container_width=True):
                st.session_state["map_clicked_lat"] = None
                st.session_state["map_clicked_lon"] = None
                st.rerun()

        map_tab1, map_tab2 = st.tabs(["🗺️ 2D Interactive Map (Folium)", "🌐 3D Perspective (Pydeck)"])
        
        with map_tab1:
            is_tracking = st.session_state.get("sim_tracking", False)
            if is_tracking:
                st.caption(f"🗺️ Live 2D Navigation (Smooth Tracking) · {tile_choice}")
                build_3d_pydeck_chart(
                    result=st.session_state.get("result"),
                    G=G, stations_gdf=stations_gdf, incidents_gdf=incidents_gdf,
                    sa_polys=sa_polys if show_cov else {},
                    sim_progress=sim_prog, vehicle_ll=show_veh_ll, bearing=veh_bearing,
                    is_navigating=is_navigating, tile_choice=tile_choice,
                    force_2d=True,
                    exact_dest_coord=exact_dest
                )
            else:
                st.caption(f"🗺️ Interactive Folium Map · {tile_choice}")
                # ── PART A1 & A2 FIX: Remove Folium Map Caching ──
                # Rebuild the map freshly every time to guarantee old routes/markers are entirely wiped out.
                base_m = build_navigable_map(
                    stations_gdf, incidents_gdf, sa_polys,
                    show_cov=show_cov, cov_t=cov_t,
                    tile_name=tile_choice,
                    incident_ll=None,
                    vehicle_ll=None,
                    click_mode=click_mode,
                    bearing=0.0,
                    card_dir="N",
                    recenter_ll=None,
                )
            
                if st.session_state.get("result"):
                    hg = st.session_state["result"].get("leg2_hospital_geom")
                    exact_h_coord = (hg.x, hg.y) if hg else None
                    base_m, leg1_ll_new, leg2_ll_new = overlay_route(
                        base_m, st.session_state["result"], G,
                        sim_progress=sim_prog if sim_prog > 0 else None,
                        exact_dest_coord=exact_dest,
                        exact_h_coord=exact_h_coord,
                    )
                
                show_dest_ll = None
                if st.session_state.get("clicked_dest_lat"):
                    show_dest_ll = (st.session_state["clicked_dest_lat"], st.session_state["clicked_dest_lon"])
                add_dynamic_markers(base_m, show_inc_ll, show_veh_ll, veh_bearing, dest_ll=show_dest_ll)
    
                _rv = st.session_state.get("route_version", 0)
                map_key = f"folium_2d_rv{_rv}_inc{st.session_state.get('clicked_inc_lat')}_{st.session_state.get('clicked_inc_lon')}_veh{st.session_state.get('clicked_veh_lat')}_{st.session_state.get('clicked_veh_lon')}"
                camera_follow = st.session_state.get("camera_follow", True)
                center_args = {}
                if recenter_ll:
                    center_args = {"center": recenter_ll, "zoom": 13}
                
                map_data = st_folium(
                    base_m,
                    height=800,
                    width="stretch",
                    returned_objects=["last_clicked"],
                    key=map_key,
                    **center_args
                )
                st.session_state["last_rendered_map_key"] = map_key

        with map_tab2:
            st.caption(f"🌐 Native 3D Perspective — Tilted Pitch (45°) · {tile_choice}")
            build_3d_pydeck_chart(
                result=st.session_state.get("result"),
                G=G, stations_gdf=stations_gdf, incidents_gdf=incidents_gdf,
                sa_polys=sa_polys if show_cov else {},
                sim_progress=sim_prog, vehicle_ll=show_veh_ll, bearing=veh_bearing,
                is_navigating=is_navigating, tile_choice=tile_choice,
                force_2d=False,
                exact_dest_coord=exact_dest
            )

    # ── CHARTS ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 System Analytics & Routing Batch Results")
    st.markdown("""
    <div style="background: rgba(13, 25, 48, 0.6); padding: 16px; border-radius: 12px; border: 1px solid rgba(99, 179, 237, 0.2); margin-bottom: 16px;">
        <span style="color:#94a3b8; font-size:0.9rem;">Review comprehensive network performance metrics and batch routing results below.</span>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Response Time Comparison",
        "🗺️ Service Area Coverage",
        "📈 Time Distribution",
    ])
    with tab1:
        if os.path.exists("outputs.zip"):
            with open("outputs.zip", "rb") as f:
                st.download_button("📥 Download Analysis Reports (.zip)", f, file_name="routing_analysis_reports.zip", mime="application/zip", use_container_width=True, type="primary")
        
        if os.path.exists("outputs/comparison_chart.png"):
            st.image("outputs/comparison_chart.png",
                     caption="Network-Optimised (Dijkstra) vs Straight-Line Baseline — All 20 Incidents",
                     use_container_width=True)
        else:
            st.info("Run `python run_all.py` to generate analysis charts.")
    with tab2:
        if os.path.exists("outputs/service_area_map.png"):
            st.image("outputs/service_area_map.png",
                     caption="Service Area Analysis — SA_t(s) = {v ∈ V | d(s,v) ≤ t}, t = 5, 10, 15 min",
                     use_container_width=True)
    with tab3:
        if os.path.exists("outputs/response_time_hist.png"):
            st.image("outputs/response_time_hist.png",
                     caption="Distribution of Ambulance Response Times — All Incidents",
                     use_container_width=True)

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;color:#475569;font-size:0.75rem;
                border-top:1px solid rgba(255,255,255,0.05); margin-top:2rem;">
      <b style="color:#64748b;">GIS-Based Optimal Ambulance Routing &amp; Emergency Response Time Analysis using Network Analysis</b><br>
      Makanjuola, Thomas Oluwadamilare (125/21/1/0180)<br>
      Abiola Ajimobi Technical University, Ibadan · 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
