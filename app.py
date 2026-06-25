"""
Running Leaderboard App - Strava API + Streamlit
Python 3.12.13 | pandas | numpy 2.0.2 | streamlit (latest)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Strava Running Leaderboard",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# STRAVA AUTH HELPERS
# ─────────────────────────────────────────
CLIENT_ID     = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI  = os.getenv("STRAVA_REDIRECT_URI", "http://localhost:8501")
BASE_URL      = "https://www.strava.com/api/v3"


def get_auth_url() -> str:
    return (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=activity:read_all"
    )


def exchange_code(code: str) -> dict:
    """Exchange auth code for tokens."""
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code":          code,
            "grant_type":    "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()


def refresh_token(refresh_tok: str) -> dict:
    """Refresh expired access token (Strava tokens expire in 6 hours)."""
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type":    "refresh_token",
            "refresh_token": refresh_tok,
        },
    )
    resp.raise_for_status()
    return resp.json()


def get_valid_token() -> str | None:
    """Return a valid access token, refreshing if expired."""
    if "access_token" not in st.session_state:
        return None
    if datetime.now().timestamp() >= st.session_state.get("expires_at", 0):
        new = refresh_token(st.session_state["refresh_token"])
        st.session_state.update(
            access_token=new["access_token"],
            refresh_token=new["refresh_token"],
            expires_at=new["expires_at"],
        )
    return st.session_state["access_token"]


# ─────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────

def fetch_activities(token: str, pages: int = 5) -> list[dict]:
    """Fetch all Run activities, paginated."""
    activities = []
    headers = {"Authorization": f"Bearer {token}"}
    for page in range(1, pages + 1):
        resp = requests.get(
            f"{BASE_URL}/athlete/activities",
            headers=headers,
            params={"per_page": 100, "page": page},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # Filter to runs only
        activities.extend(a for a in batch if a.get("sport_type") == "Run")
    return activities


# ─────────────────────────────────────────
# DATA PARSING  (Step 4 & 5)
# ─────────────────────────────────────────

def parse_activities(raw: list[dict]) -> pd.DataFrame:
    """
    Extract relevant fields and derive 'location' from city/state or
    start_latlng grid cell (5 km buckets) as a fallback.
    """
    records = []
    for a in raw:
        lat_lng = a.get("start_latlng") or [None, None]
        records.append(
            {
                "id":            a["id"],
                "name":          a["name"],
                "date":          pd.to_datetime(a["start_date_local"]),
                "distance_km":   round(a["distance"] / 1000, 2),
                "moving_time_s": a["moving_time"],
                "elevation_m":   a.get("total_elevation_gain", 0),
                "city":          a.get("location_city") or "",
                "state":         a.get("location_state") or "",
                "country":       a.get("location_country") or "",
                "start_lat":     lat_lng[0],
                "start_lng":     lat_lng[1],
                "athlete_name":  a.get("athlete", {}).get("firstname", "You"),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Build human-readable location label
    df["location"] = df.apply(_derive_location, axis=1)
    df["pace_min_km"] = (df["moving_time_s"] / 60) / df["distance_km"]
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def _derive_location(row) -> str:
    """City → State → Grid cell fallback."""
    if row["city"]:
        return f"{row['city']}, {row['state']}".strip(", ")
    if row["state"]:
        return row["state"]
    if row["start_lat"] is not None:
        # 0.045° ≈ 5 km bucket
        lat_b = round(row["start_lat"] / 0.045) * 0.045
        lng_b = round(row["start_lng"] / 0.045) * 0.045
        return f"Area ({lat_b:.2f}, {lng_b:.2f})"
    return "Unknown"


# ─────────────────────────────────────────
# LEADERBOARD LOGIC  (Step 6)
# ─────────────────────────────────────────

def build_leaderboard(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """Rank athletes by total km at the selected location."""
    subset = df if location == "🌍 All Locations" else df[df["location"] == location]
    lb = (
        subset.groupby("athlete_name")
        .agg(
            total_km=("distance_km", "sum"),
            total_runs=("id", "count"),
            avg_pace=("pace_min_km", "mean"),
            best_distance=("distance_km", "max"),
        )
        .reset_index()
        .sort_values("total_km", ascending=False)
        .reset_index(drop=True)
    )
    lb.index += 1                               # Rank starts at 1
    lb["total_km"]       = lb["total_km"].round(1)
    lb["avg_pace"]       = lb["avg_pace"].round(2)
    lb["best_distance"]  = lb["best_distance"].round(1)
    lb.rename(columns={
        "athlete_name":  "Athlete",
        "total_km":      "Total KM",
        "total_runs":    "Runs",
        "avg_pace":      "Avg Pace (min/km)",
        "best_distance": "Longest Run (km)",
    }, inplace=True)
    return lb


# ─────────────────────────────────────────
# DEMO DATA  (when not authenticated)
# ─────────────────────────────────────────

def load_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    athletes = ["Ali Hassan", "Sara Khan", "Bilal Ahmed", "Nadia Iqbal", "Omar Farooq"]
    locations = ["Lahore, PB", "Karachi, SD", "Islamabad, IS", "Area (31.55, 74.35)"]
    rows = []
    for athlete in athletes:
        for _ in range(rng.integers(10, 30)):
            dist = float(rng.uniform(3, 21))
            pace = float(rng.uniform(4.5, 7.5))
            rows.append({
                "id":            rng.integers(1_000_000),
                "name":          "Morning Run",
                "date":          pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
                "distance_km":   round(dist, 2),
                "moving_time_s": int(dist * pace * 60),
                "elevation_m":   float(rng.uniform(0, 120)),
                "location":      rng.choice(locations),
                "athlete_name":  athlete,
                "pace_min_km":   round(pace, 2),
                "month":         "",
            })
    df = pd.DataFrame(rows)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df.sort_values("date", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────

def render_kpi(df: pd.DataFrame):
    total_km   = df["distance_km"].sum()
    total_runs = len(df)
    avg_pace   = df["pace_min_km"].mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("🏃 Total Runs",    f"{total_runs}")
    col2.metric("📏 Total KM",      f"{total_km:,.1f} km")
    col3.metric("⚡ Avg Pace",      f"{avg_pace:.2f} min/km")


def render_leaderboard(df: pd.DataFrame, location: str):
    lb = build_leaderboard(df, location)
    if lb.empty:
        st.info("No runs found for this location.")
        return

    # Medal for top 3
    def medal(rank):
        return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

    lb_display = lb.copy()
    lb_display.index = lb_display.index.map(medal)
    st.dataframe(lb_display, use_container_width=True, height=320)


def render_charts(df: pd.DataFrame, location: str):
    subset = df if location == "🌍 All Locations" else df[df["location"] == location]
    if subset.empty:
        return

    tab1, tab2, tab3 = st.tabs(["📊 KM by Month", "🏅 KM per Athlete", "🗺️ Distance Distribution"])

    with tab1:
        monthly = (
            subset.groupby(["month", "athlete_name"])["distance_km"]
            .sum()
            .reset_index()
        )
        fig = px.bar(
            monthly, x="month", y="distance_km",
            color="athlete_name", barmode="stack",
            labels={"distance_km": "KM", "month": "Month", "athlete_name": "Athlete"},
            title="Monthly KM (stacked by athlete)",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        per_athlete = (
            subset.groupby("athlete_name")["distance_km"]
            .sum()
            .reset_index()
            .sort_values("distance_km", ascending=True)
        )
        fig2 = px.bar(
            per_athlete, y="athlete_name", x="distance_km",
            orientation="h",
            labels={"distance_km": "Total KM", "athlete_name": "Athlete"},
            title="Total KM per Athlete",
            color="distance_km",
            color_continuous_scale="Teal",
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = px.histogram(
            subset, x="distance_km", nbins=20,
            labels={"distance_km": "Run Distance (km)"},
            title="Run Distance Distribution",
            color_discrete_sequence=["#00c9a7"],
        )
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    st.title("🏃 Running Leaderboard")
    st.caption("Powered by Strava API v3 · Python 3.12 · Streamlit")

    # ── Sidebar ──────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        demo_mode = st.toggle("Use Demo Data", value=True)
        st.markdown("---")

        if not demo_mode:
            # OAuth flow
            token = get_valid_token()
            if not token:
                # Check for callback code in URL
                params = st.query_params
                if "code" in params:
                    with st.spinner("Authenticating with Strava…"):
                        tok_data = exchange_code(params["code"])
                        st.session_state.update(
                            access_token=tok_data["access_token"],
                            refresh_token=tok_data["refresh_token"],
                            expires_at=tok_data["expires_at"],
                        )
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.markdown(f"[🔗 Connect Strava]({get_auth_url()})")
                    st.info("Connect your Strava account to load real data.")
                    return
            else:
                st.success("✅ Strava Connected")
                if st.button("Disconnect"):
                    for k in ["access_token", "refresh_token", "expires_at"]:
                        st.session_state.pop(k, None)
                    st.rerun()

        st.markdown("---")
        pages = st.slider("Activity pages to fetch", 1, 10, 3, help="Each page = 100 activities")
        st.caption("Rate limit: 200 req / 15 min · 2,000 / day")

    # ── Load Data ─────────────────────────
    if demo_mode:
        df = load_demo_data()
        st.info("🔵 **Demo Mode** — showing synthetic data. Toggle off to connect Strava.", icon="ℹ️")
    else:
        token = get_valid_token()
        with st.spinner("Fetching your Strava activities…"):
            raw = fetch_activities(token, pages=pages)
        df = parse_activities(raw)
        if df.empty:
            st.warning("No running activities found. Go log a run! 🏃")
            return

    # ── Location Selector (Step 7) ────────
    locations = ["🌍 All Locations"] + sorted(df["location"].unique().tolist())
    selected  = st.selectbox("📍 Filter by Location", locations)

    # ── KPIs ──────────────────────────────
    render_kpi(df)
    st.divider()

    # ── Leaderboard (Step 8) ─────────────
    st.subheader(f"🏆 Leaderboard — {selected}")
    render_leaderboard(df, selected)
    st.divider()

    # ── Charts (Step 9) ───────────────────
    st.subheader("📈 Progress & Analytics")
    render_charts(df, selected)

    # ── Raw data expander ─────────────────
    with st.expander("🗂️ Raw Activity Data"):
        cols_show = ["name", "date", "distance_km", "pace_min_km", "elevation_m", "location", "athlete_name"]
        st.dataframe(df[[c for c in cols_show if c in df.columns]], use_container_width=True)


if __name__ == "__main__":
    main()
