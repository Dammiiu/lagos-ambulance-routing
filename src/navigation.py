"""
src/navigation.py

Navigation helpers for the ambulance routing system:
  - Turn-by-turn direction generation from a Dijkstra path
  - Journey simulation state management
  - Route coordinate interpolation for smooth animation

These utilities support the live navigation simulation feature in app.py.
NOTE: This is a SIMULATION of the pre-computed Dijkstra optimal route —
not real GPS tracking. Real GPS would require device hardware and
a persistent backend, which is outside this project's scope.
"""
import math
import warnings
warnings.filterwarnings("ignore")

from pyproj import Transformer

_to_wgs84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)


# --- Bearing & Turn Computation ---

def bearing_deg(x1, y1, x2, y2) -> float:
    """
    Compass bearing (0°=North, 90°=East) from point (x1,y1) to (x2,y2)
    in a projected CRS where x=Easting, y=Northing.
    """
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))  # N=0, E=90
    return angle % 360


def turn_direction(bear_in: float, bear_out: float) -> str:
    """
    Classify a turn given incoming and outgoing bearings (degrees).
    Returns one of: 'Continue straight', 'Turn left', 'Turn right',
                    'Sharp left', 'Sharp right', 'U-turn'
    """
    diff = (bear_out - bear_in + 360) % 360
    if diff > 180:
        diff -= 360  # now in [-180, 180]
    if abs(diff) <= 20:
        return "Continue straight"
    elif diff < -120:
        return "Sharp left"
    elif diff < -20:
        return "Turn left"
    elif diff > 120:
        return "Sharp right"
    elif diff > 20:
        return "Turn right"
    else:
        return "U-turn"


def cardinal_direction(bear: float) -> str:
    """Return a compass cardinal/intercardinal label for a bearing."""
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    ix = int((bear + 22.5) / 45) % 8
    return dirs[ix]


# --- Turn-by-Turn Direction Generation ---

def generate_directions(G, path: list) -> list:
    """
    Walk the Dijkstra path and generate turn-by-turn instructions.

    Algorithm:
    1. Group consecutive edges that share the same street name into one
       segment ("Head [dir] on [street] for [dist]m")
    2. When the street name changes, compute the turn direction using
       bearing change at the junction node
    3. Emit a numbered instruction list

    Parameters
    ----------
    G    : nx.DiGraph with edge attrs: name, length_m, geometry
    path : list of node IDs from Dijkstra

    Returns
    -------
    list of dicts, each with:
        step        : int    — step number (1-based)
        instruction : str    — human-readable instruction
        distance_m  : float  — distance for this segment
        street_name : str    — current street name
        node_idx    : int    — index in path where this instruction starts
        coord       : tuple  — (lat, lon) WGS84 of turn point
    """
    if len(path) < 2:
        return []

    directions = []
    step = 1

    # Gather per-edge info: (name, length_m, bearing_out, geometry_coords)
    edge_info = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ed_dict = G.get_edge_data(u, v) or {}
        ed = ed_dict[0] if 0 in ed_dict else ed_dict
        
        name    = ed.get("name", "") or ""
        length  = ed.get("length_m", 0.0) or 0.0
        geom    = ed.get("geometry", [])

        # Bearing: use first two geometry points if available, else node coords
        if len(geom) >= 2:
            bx1, by1 = geom[0]
            bx2, by2 = geom[1]
        elif "x" in G.nodes.get(u, {}) and "x" in G.nodes.get(v, {}):
            bx1, by1 = G.nodes[u]["x"], G.nodes[u]["y"]
            bx2, by2 = G.nodes[v]["x"], G.nodes[v]["y"]
        else:
            bx1, by1, bx2, by2 = 0, 0, 1, 0

        bear = bearing_deg(bx1, by1, bx2, by2)
        edge_info.append({
            "name": name,
            "length_m": float(length),
            "bearing": bear,
            "node_u": path[i],
            "node_v": path[i + 1],
        })

    # Group consecutive same-name edges
    segments = []  # list of {name, total_length, bearing_start, node_start_idx, bear_end}
    cur_name    = edge_info[0]["name"]
    cur_length  = edge_info[0]["length_m"]
    cur_bear    = edge_info[0]["bearing"]
    cur_node_idx = 0

    for i in range(1, len(edge_info)):
        ei = edge_info[i]
        if ei["name"] == cur_name:
            cur_length += ei["length_m"]
        else:
            segments.append({
                "name":       cur_name,
                "length_m":   cur_length,
                "bearing":    cur_bear,
                "node_idx":   cur_node_idx,
                "bear_out":   ei["bearing"],
            })
            cur_name     = ei["name"]
            cur_length   = ei["length_m"]
            cur_bear     = ei["bearing"]
            cur_node_idx = i

    # Last segment
    segments.append({
        "name":       cur_name,
        "length_m":   cur_length,
        "bearing":    cur_bear,
        "node_idx":   cur_node_idx,
        "bear_out":   cur_bear,
    })

    # Build instruction list
    for s_idx, seg in enumerate(segments):
        node_path_idx = seg["node_idx"]
        node_id = path[node_path_idx]

        # Get WGS84 coords of this node for the map marker
        nd = G.nodes.get(node_id, {})
        if "x" in nd:
            lon, lat = _to_wgs84.transform(nd["x"], nd["y"])
            coord = (lat, lon)
        else:
            coord = None

        street = seg["name"] if seg["name"] else "unnamed road"
        dist   = seg["length_m"]

        if s_idx == 0:
            # First instruction: heading direction
            card = cardinal_direction(seg["bearing"])
            instr = f"Head {card} on {street} for {dist:.0f} m"
        else:
            # Turn instruction based on bearing change from previous segment
            prev_bear = segments[s_idx - 1]["bear_out"]
            turn = turn_direction(prev_bear, seg["bearing"])
            if dist > 10:
                instr = f"{turn} onto {street} and continue {dist:.0f} m"
            else:
                instr = f"{turn} onto {street}"

        directions.append({
            "step":        step,
            "instruction": instr,
            "distance_m":  dist,
            "street_name": street,
            "node_idx":    node_path_idx,
            "coord":       coord,
        })
        step += 1

    # Final: arrive
    last_node = path[-1]
    nd = G.nodes.get(last_node, {})
    if "x" in nd:
        lon, lat = _to_wgs84.transform(nd["x"], nd["y"])
        arr_coord = (lat, lon)
    else:
        arr_coord = None
    directions.append({
        "step":        step,
        "instruction": "Arrive at destination",
        "distance_m":  0,
        "street_name": "",
        "node_idx":    len(path) - 1,
        "coord":       arr_coord,
    })

    return directions


def format_directions_html(directions: list, leg_label: str = "Route") -> str:
    """Render direction list as styled scrollable card list with turn icons."""
    if not directions:
        return "<p style='color:#64748b;font-size:0.78rem;'>No route directions available.</p>"

    icons = {
        "Turn left": "↰",
        "Sharp left": "↰",
        "Turn right": "↱",
        "Sharp right": "↱",
        "Continue straight": "↑",
        "Head": "🧭",
        "U-turn": "↻",
        "Arrive": "📍",
    }

    html_parts = [
        f"<div style='font-size:0.7rem;font-weight:700;color:#90cdf4;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:0.5rem;padding-bottom:0.3rem;"
        f"border-bottom:1px solid rgba(255,255,255,0.08)'>{leg_label}</div>"
    ]
    for d in directions:
        instr = d["instruction"]
        dist  = d["distance_m"]

        icon = "📍" if "Arrive" in instr else "🧭"
        for k, v in icons.items():
            if instr.startswith(k):
                icon = v
                break

        dist_str = f"<span style='color:#94a3b8;font-weight:600;font-size:0.68rem;background:rgba(255,255,255,0.06);padding:0.12rem 0.4rem;border-radius:10px;'>{dist:.0f} m</span>" if dist > 0 else ""
        step_num = d["step"]

        html_parts.append(
            f"<div style='display:flex;align-items:center;padding:0.45rem 0.55rem;margin-bottom:0.3rem;"
            f"border-radius:8px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);gap:0.6rem'>"
            f"<span style='min-width:18px;color:#64748b;font-size:0.7rem;font-weight:700;'>#{step_num}</span>"
            f"<span style='font-size:1.1rem;color:#38bdf8;line-height:1;'>{icon}</span>"
            f"<span style='flex:1;font-size:0.78rem;color:#e2e8f0;font-weight:500;line-height:1.35'>{instr}</span>"
            f"{dist_str}"
            f"</div>"
        )

    return "".join(html_parts)


# --- Route Coordinate Interpolation (for smooth animation) ---

def interpolate_route(utm_coords: list, step_m: float = 20.0) -> list:
    """
    Densify a route by interpolating points every step_m metres.
    Returns a list of (x, y) UTM tuples.
    """
    if len(utm_coords) < 2:
        return utm_coords

    result = [utm_coords[0]]
    accumulated = 0.0

    for i in range(1, len(utm_coords)):
        x0, y0 = utm_coords[i - 1]
        x1, y1 = utm_coords[i]
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if seg_len < 1e-6:
            continue

        dx = (x1 - x0) / seg_len
        dy = (y1 - y0) / seg_len

        remaining = seg_len
        pos = 0.0

        while accumulated + remaining >= step_m:
            advance = step_m - accumulated
            pos += advance
            remaining -= advance
            accumulated = 0.0
            px = x0 + dx * pos
            py = y0 + dy * pos
            result.append((px, py))

        accumulated += remaining

    if result[-1] != utm_coords[-1]:
        result.append(utm_coords[-1])

    return result


def utm_to_ll_list(utm_coords: list) -> list:
    """Convert list of (x,y) UTM to list of (lat,lon) WGS84."""
    out = []
    for x, y in utm_coords:
        lon, lat = _to_wgs84.transform(x, y)
        out.append((lat, lon))
    return out


def route_total_distance(utm_coords: list) -> float:
    """Total distance in metres along a UTM coordinate list."""
    total = 0.0
    for i in range(1, len(utm_coords)):
        x0, y0 = utm_coords[i - 1]
        x1, y1 = utm_coords[i]
        total += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return total


def cumulative_distances(utm_coords: list) -> list:
    """Return list of cumulative distances (metres) at each coord point."""
    dists = [0.0]
    for i in range(1, len(utm_coords)):
        x0, y0 = utm_coords[i - 1]
        x1, y1 = utm_coords[i]
        dists.append(dists[-1] + math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    return dists
