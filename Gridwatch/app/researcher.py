"""
researcher.py — GridWatch Experience 2
Researcher / City Official Dashboard
"""

import json
import subprocess
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import requests
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip",
        "install", "requests",
    ])
    import requests


def _install_if_missing(package, import_name=None):
    name = import_name or package.replace("-", "_")
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


_install_if_missing("folium")
_install_if_missing("streamlit-folium", "streamlit_folium")
_install_if_missing("plotly")

try:
    import folium
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except ImportError:
    _FOLIUM_OK = False

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


# ── City geographic coordinates ───────────────────────────────────────────────
CITY_COORDS = {
    "Adelanto":           (34.5827, -117.4325),
    "Agoura Hills":       (34.1530, -118.7617),
    "Apple Valley":       (34.5008, -117.1858),
    "Arcata":             (40.8665, -124.0827),
    "Arvin":              (35.2088, -118.8285),
    "Alturas":            (41.4871, -120.5425),
    "Avenal":             (36.0038, -120.1297),
    "Baldwin Park":       (34.0853, -117.9609),
    "Barstow":            (34.8958, -117.0228),
    "Bell Gardens":       (33.9645, -118.1512),
    "Blythe":             (33.6175, -114.5961),
    "Carson":             (33.8317, -118.2819),
    "Cathedral City":     (33.7797, -116.4653),
    "Chino":              (34.0122, -117.6889),
    "Chino Hills":        (33.9939, -117.7328),
    "Coachella":          (33.6803, -116.1738),
    "Compton":            (33.8958, -118.2201),
    "Corcoran":           (36.0977, -119.5596),
    "Crescent City":      (41.7558, -124.2026),
    "Cudahy":             (33.9622, -118.1862),
    "Delano":             (35.7688, -119.2471),
    "Desert Hot Springs": (33.9611, -116.5014),
    "Firebaugh":          (36.8591, -120.4566),
    "Hawthorne":          (33.9164, -118.3525),
    "Hemet":              (33.7475, -116.9719),
    "Huron":              (36.2013, -120.1016),
    "Huntington Park":    (33.9814, -118.2248),
    "Indio":              (33.7206, -116.2156),
    "Inglewood":          (33.9617, -118.3531),
    "Kettleman City":     (36.0077, -119.9616),
    "King City":          (36.2127, -121.1260),
    "La Puente":          (34.0206, -117.9495),
    "Lompoc":             (34.6391, -120.4579),
    "Lynwood":            (33.9303, -118.2114),
    "Maywood":            (33.9872, -118.1873),
    "Moreno Valley":      (33.9425, -117.2297),
    "Ontario":            (34.0633, -117.6509),
    "Palm Springs":       (33.8303, -116.5453),
    "Parlier":            (36.6099, -119.5257),
    "Perris":             (33.7825, -117.2286),
    "Pomona":             (34.0553, -117.7500),
    "Rialto":             (34.1064, -117.3703),
    "San Joaquin":        (36.6063, -120.1883),
    "Santa Maria":        (34.9530, -120.4357),
    "Shafter":            (35.5007, -119.2719),
    "South Gate":         (33.9547, -118.2120),
    "Susanville":         (40.4160, -120.6524),
    "Victorville":        (34.5362, -117.2928),
    "Wasco":              (35.5938, -119.3410),
    "Yreka":              (41.7352, -122.6344),
}
_CA_CENTER = (36.7783, -119.4179)

CITY_REGIONS = {
    "Inland Empire":     ["Adelanto", "Apple Valley", "Barstow", "Blythe", "Cathedral City",
                          "Coachella", "Desert Hot Springs", "Hemet", "Indio", "Palm Springs",
                          "Perris", "Victorville"],
    "Central Valley":    ["Arvin", "Avenal", "Corcoran", "Delano", "Firebaugh", "Huron",
                          "Kettleman City", "Parlier", "San Joaquin", "Shafter", "Wasco"],
    "Northern California": ["Alturas", "Arcata", "Crescent City", "Susanville", "Yreka"],
    "Central Coast":     ["King City", "Lompoc", "Santa Maria"],
    "LA Metro":          ["Compton", "Cudahy", "Huntington Park", "Lynwood", "Maywood",
                          "Hawthorne", "Inglewood", "South Gate", "Bell Gardens", "Carson",
                          "La Puente", "Baldwin Park"],
    "Inland SoCal":      ["Chino", "Chino Hills", "Ontario", "Pomona", "Rialto", "Moreno Valley"],
    "Other":             ["Agoura Hills"],
}
_REGION_ORDER = [
    "Inland Empire", "Central Valley", "LA Metro",
    "Inland SoCal", "Central Coast", "Northern California", "Other",
]


# ── Pure helpers ──────────────────────────────────────────────────────────────
def _coords(city):
    return CITY_COORDS.get(city, _CA_CENTER)


def _region(city):
    for r, cities in CITY_REGIONS.items():
        if city in cities:
            return r
    return "Other"


def _eligible_programs(income):
    p = []
    if income < 66_527: p.append("LIHEAP")
    if income < 64_300: p.append("CARE")
    if 64_300 <= income <= 80_375: p.append("FERA")
    if income < 64_300: p.append("WAP")
    if income < 64_300: p.append("ESA")
    return p


def _eligible_label(city_income, limit):
    ratio = city_income / limit
    if ratio < 0.70:  return "~Most residents"
    if ratio < 1.00:  return "~Many residents"
    if ratio < 1.15:  return "~Some residents"
    return "~Few residents"


# ── AI helpers (all cached per city in session state) ─────────────────────────
def _ai_risk_explanation(client, model_name, city_data, feature_impacts, weather_context=""):
    key = f"r2_risk_{city_data['city']}"
    if key in st.session_state:
        return st.session_state[key]

    top3 = feature_impacts.head(3)
    drivers = ", ".join(f"{k.replace('_',' ')} ({v:+.4f})" for k, v in top3.items())
    min_temp = city_data.get("min_temp_forecast", "N/A")

    weather_section = (
        f"Live 2-week weather forecast:\n{weather_context}\n" if weather_context else ""
    )
    prompt = (
        f"City: {city_data['city']}\n"
        f"Energy burden: {city_data['energy_burden']:.2f}%\n"
        f"Household income: ${city_data['household_income']:,.0f}\n"
        f"Avg annual energy cost: ${city_data['avg_annual_energy_cost']:,.0f}\n"
        f"Risk level: {'HIGH' if city_data['risk_label'] == 1 else 'LOW'}\n"
        f"Top SHAP drivers: {drivers}\n"
        f"Min temp forecast: {min_temp}°F\n"
        f"{weather_section}\n"
        "Explain in plain English why this city is at energy risk and what the main factors are."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are an energy policy analyst. Explain why a city is at energy risk in clear, "
                    "non-technical language suitable for a city official. Be specific, use the actual "
                    "numbers provided, and keep it to 3 sentences maximum."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        result = resp.choices[0].message.content.strip()
    except Exception:
        result = "AI analysis temporarily unavailable."

    st.session_state[key] = result
    return result


def _ai_forecast_summary(client, model_name, city_name, burden, weekly):
    key = f"r2_forecast_{city_name}"
    if key in st.session_state:
        return st.session_state[key]

    weeks_str = "\n".join(
        f"Week {w['week']}: max {w['max_temp']:.0f}°F, min {w['min_temp']:.0f}°F, stress={w['stress']}"
        for w in weekly[:2]
    )
    prompt = (
        f"City: {city_name}\nCurrent energy burden: {burden:.1f}%\n"
        f"Next 2 weeks:\n{weeks_str}\n\n"
        "In 2 sentences, summarize how the next 2 weeks look for energy burden risk in this city. "
        "Be specific about which week looks more concerning and why."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        result = resp.choices[0].message.content.strip()
    except Exception:
        result = "Forecast summary temporarily unavailable."

    st.session_state[key] = result
    return result


def _ai_policy_recs(client, model_name, city_data, feature_impacts, eligible, forecast_stress, weather_context=""):
    key = f"r2_policy_{city_data['city']}"
    if key in st.session_state:
        return st.session_state[key]

    top3 = feature_impacts.head(3)
    drivers = ", ".join(k.replace("_", " ") for k in top3.index[:3])
    programs_str = ", ".join(eligible) if eligible else "None (income above program thresholds)"

    weather_section = (
        f"Current live weather (next 2 weeks):\n{weather_context}\n"
        "Factor this into your immediate action recommendations. "
        "If HIGH stress is forecast, prioritize cooling assistance. "
        "If LOW/MODERATE, focus on longer term efficiency programs.\n"
        if weather_context else ""
    )
    user_prompt = (
        f"City: {city_data['city']}\n"
        f"Energy burden: {city_data['energy_burden']:.2f}% (threshold is 3%)\n"
        f"Household income: ${city_data['household_income']:,.0f}\n"
        f"Avg annual energy cost: ${city_data['avg_annual_energy_cost']:,.0f}\n"
        f"Risk level: {'HIGH' if city_data['risk_label'] == 1 else 'LOW'}\n"
        f"Top risk drivers: {drivers}\n"
        f"Programs currently available: {programs_str}\n"
        f"Next 2 weeks forecast stress: {forecast_stress}\n"
        f"{weather_section}\n"
        "Return valid JSON only — no markdown fences — with exactly these three string keys:\n"
        '{"immediate": "...", "medium_term": "...", "policy_gaps": "..."}\n'
        "Each value must be 2–3 bullet points starting with • and separated by newlines.\n"
        "immediate: actions in next 30 days.\n"
        "medium_term: policy changes in 6 months.\n"
        "policy_gaps: 2 missing programs that would help this city."
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": (
                    "You are a senior energy policy advisor to California city governments. "
                    "Give specific, actionable recommendations with program names, dollar amounts, "
                    "and timeframes. Return valid JSON only — no markdown, no explanations outside the JSON."
                )},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        result = {
            "immediate":    "• AI recommendations temporarily unavailable.",
            "medium_term":  "• AI recommendations temporarily unavailable.",
            "policy_gaps":  "• AI recommendations temporarily unavailable.",
        }

    st.session_state[key] = result
    return result


def get_live_weather_forecast(lat, lon):
    try:
        import requests

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "daily": "temperature_2m_max,temperature_2m_min",
            "forecast_days": 16,
            "timezone": "America/Los_Angeles",
        }
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if "daily" in data:
                max_c = data["daily"]["temperature_2m_max"]
                min_c = data["daily"]["temperature_2m_min"]
                data["daily"]["temperature_2m_max"] = [
                    round((t * 9 / 5) + 32, 1)
                    for t in max_c
                ]
                data["daily"]["temperature_2m_min"] = [
                    round((t * 9 / 5) + 32, 1)
                    for t in min_c
                ]
                return data
        return None
    except Exception:
        return None


def _weekly_summaries(forecast_data):
    """Accept a Fahrenheit forecast dict and return 2 weekly summaries."""
    max_temps = forecast_data["daily"]["temperature_2m_max"]
    min_temps = forecast_data["daily"]["temperature_2m_min"]
    weeks = []
    for w in range(2):
        s, e   = w * 7, min(w * 7 + 7, len(max_temps))
        if s >= len(max_temps):
            break
        wmax   = max(max_temps[s:e])
        wmin   = min(min_temps[s:e])
        avg_mx = sum(max_temps[s:e]) / (e - s)
        if wmax > 95 or wmin < 35:
            stress = "HIGH"
        elif wmax > 80:
            stress = "MODERATE"
        else:
            stress = "LOW"
        weeks.append({"week": w + 1, "label": f"Wk {w+1}",
                      "max_temp": wmax, "min_temp": wmin,
                      "avg_max": avg_mx, "stress": stress})
    return weeks


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Coverage Overview
# ══════════════════════════════════════════════════════════════════════════════
def _show_coverage_overview(data_df):
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#0F172A;'>California Energy Risk Overview</h2>"
        "<p style='color:#64748B;margin-top:0;'>Exploring energy affordability risk across 50 California cities</p>",
        unsafe_allow_html=True,
    )

    n_high = int((data_df["risk_label"] == 1).sum())
    n_low  = int((data_df["risk_label"] == 0).sum())
    st.markdown(
        f"<div style='margin-bottom:16px;font-size:1.05rem;color:#1E293B;'>"
        f"🔴 <strong style='color:#DC2626;'>{n_high} High Risk</strong> &nbsp;&nbsp; "
        f"🟢 <strong style='color:#16A34A;'>{n_low} Low Risk</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_list, col_map = st.columns([4, 6])

    # ── City list ─────────────────────────────────────────────────────────────
    with col_list:
        st.markdown("#### Cities We Cover")

        region_map = {}
        for city in sorted(data_df["city"].unique()):
            region_map.setdefault(_region(city), []).append(city)

        for reg in _REGION_ORDER:
            if reg not in region_map:
                continue
            cities_in_reg = region_map[reg]
            st.markdown(f"**{reg}** ({len(cities_in_reg)})")
            for city in sorted(cities_in_reg):
                row = data_df[data_df["city"] == city].iloc[0]
                icon = "🔴" if row["risk_label"] == 1 else "🟢"
                if st.button(f"{icon} {city}", key=f"city_btn_{city}",
                             use_container_width=True):
                    _select_city(city)
            st.markdown("---")

    # ── Folium map ────────────────────────────────────────────────────────────
    with col_map:
        st.markdown("#### California Energy Risk Map")

        if not _FOLIUM_OK:
            st.warning("Install `folium` and `streamlit-folium` to enable the interactive map.")
        else:
            m = folium.Map(
                location=[36.7783, -119.4179],
                zoom_start=6,
                tiles="CartoDB positron",
            )

            for _, row in data_df.iterrows():
                city = row["city"]
                lat, lon = _coords(city)
                color     = "#FF4444" if row["risk_label"] == 1 else "#44BB44"
                risk_lbl  = "HIGH" if row["risk_label"] == 1 else "LOW"

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    tooltip=folium.Tooltip(
                        f"<b>{city}</b><br>"
                        f"Risk: {risk_lbl}<br>"
                        f"Energy Burden: {row['energy_burden']:.1f}%<br>"
                        f"Avg Income: ${row['household_income']:,.0f}"
                    ),
                    popup=folium.Popup(city, max_width=200),
                ).add_to(m)

            legend_html = (
                '<div style="position:fixed;bottom:20px;right:20px;z-index:9999;'
                'background:rgba(255,255,255,0.95);padding:8px 12px;border-radius:6px;'
                'color:#1E293B;font-size:12px;border:1px solid #E2E8F0;'
                'box-shadow:0 1px 3px rgba(0,0,0,0.1);">'
                '<span style="color:#DC2626;">●</span> High Risk &nbsp;'
                '<span style="color:#16A34A;">●</span> Low Risk</div>'
            )
            m.get_root().html.add_child(folium.Element(legend_html))

            map_result = st_folium(
                m, height=520, use_container_width=True, key="overview_map",
                returned_objects=["last_object_clicked_popup", "last_object_clicked_tooltip"],
            )

            if map_result:
                # Try popup first, then tooltip (varies by streamlit-folium version)
                clicked_city = (
                    map_result.get("last_object_clicked_popup")
                    or map_result.get("last_object_clicked_tooltip")
                )
                if isinstance(clicked_city, str) and clicked_city in data_df["city"].values:
                    if st.session_state.get("researcher_city") != clicked_city:
                        _select_city(clicked_city)

    st.markdown(
        f"<p style='color:#94A3B8;font-size:0.8rem;margin-top:1rem;'>"
        f"Data last updated: {datetime.now().strftime('%B %d, %Y')}</p>",
        unsafe_allow_html=True,
    )


def _select_city(city):
    st.session_state["researcher_city"]   = city
    st.session_state["researcher_screen"] = 2
    st.session_state["r2_chat_messages"]  = []
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — City Research Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def _show_city_dashboard(data_df, shap_df, client, model_name):
    city_name = st.session_state.get("researcher_city")
    if not city_name or city_name not in data_df["city"].values:
        st.session_state["researcher_screen"] = 1
        st.rerun()
        return

    city_data       = data_df[data_df["city"] == city_name].iloc[0]
    city_shap       = shap_df[shap_df["city"] == city_name].iloc[0]
    feature_impacts = city_shap.drop("city").sort_values(key=abs, ascending=False)

    is_high     = city_data["risk_label"] == 1
    risk_txt    = "HIGH RISK" if is_high else "LOW RISK"
    burden      = float(city_data["energy_burden"])
    income      = float(city_data["household_income"])
    energy_cost = float(city_data["avg_annual_energy_cost"])
    min_temp    = city_data.get("min_temp_forecast", None)
    eligible    = _eligible_programs(income)

    # Fetch live weather early so it's available for all AI prompts
    lat, lon = _coords(city_name)
    weather_key = f"r2_weather_{city_name}"
    if weather_key not in st.session_state or st.session_state[weather_key] is None:
        if lat and lon:
            st.session_state[weather_key] = get_live_weather_forecast(lat, lon)
        else:
            st.session_state[weather_key] = None
    weather_data = st.session_state[weather_key]

    weather_context = ""
    overall_stress = "LOW"
    if weather_data and "daily" in weather_data:
        max_temps = weather_data["daily"]["temperature_2m_max"]
        min_temps = weather_data["daily"]["temperature_2m_min"]
        weeks = []
        for i in range(2):
            s = i * 7
            e = min(s + 7, len(max_temps))
            if s >= len(max_temps):
                break
            w_max = max(max_temps[s:e])
            w_min = min(min_temps[s:e])
            stress = "HIGH" if w_max > 95 or w_min < 35 else "MODERATE" if w_max > 80 else "LOW"
            weeks.append({"week": i + 1, "max": w_max, "min": w_min, "stress": stress})
        for w in weeks:
            weather_context += f"Week {w['week']}: Max {w['max']}°F, Min {w['min']}°F — {w['stress']} stress\n"
        if weeks:
            overall_stress = (
                "HIGH"     if any(w["stress"] == "HIGH"     for w in weeks) else
                "MODERATE" if any(w["stress"] == "MODERATE" for w in weeks) else
                "LOW"
            )

    if st.button("← Back to Coverage Overview", key="back_overview"):
        st.session_state["researcher_screen"] = 1
        st.rerun()

    st.markdown(
        f"<h2 style='margin-bottom:4px;color:#0F172A;'>{city_name} — Energy Risk Dashboard</h2>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Section 1: City Snapshot ──────────────────────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>City Snapshot</h2>"
        f"<p style='margin:0;font-size:0.85rem;color:#64748B;'>Key metrics for {city_name}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    burden_color = "#4ADE80" if burden <= 3 else "#F87171"
    risk_bg = "#14532D" if "LOW" in risk_txt else "#1A0000"
    risk_border = "#22C55E" if "LOW" in risk_txt else "#DC2626"
    risk_color = "#4ADE80" if "LOW" in risk_txt else "#F87171"
    temp_display = f"{min_temp:.0f}" if min_temp is not None else "N/A"

    st.markdown(f"""
<div style="display: flex; gap: 12px; margin: 1rem 0;">

  <div style="flex:1; background:#1E293B;
              border:1px solid #334155;
              border-radius:12px; padding:1.2rem;
              text-align:center;">
    <div style="font-size:0.7rem; font-weight:500;
                color:#94A3B8; text-transform:uppercase;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;">
      Energy Burden
    </div>
    <div style="font-size:2rem; font-weight:700;
                color:{burden_color};">
      {burden:.1f}%
    </div>
    <div style="font-size:0.75rem; color:#64748B;
                margin-top:0.3rem;">
      % of income on energy
    </div>
  </div>

  <div style="flex:1; background:#1E293B;
              border:1px solid #334155;
              border-radius:12px; padding:1.2rem;
              text-align:center;">
    <div style="font-size:0.7rem; font-weight:500;
                color:#94A3B8; text-transform:uppercase;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;">
      Avg Household Income
    </div>
    <div style="font-size:2rem; font-weight:700;
                color:#38BDF8;">
      ${income:,.0f}
    </div>
    <div style="font-size:0.75rem; color:#64748B;
                margin-top:0.3rem;">
      Median annual income
    </div>
  </div>

  <div style="flex:1; background:#1E293B;
              border:1px solid #334155;
              border-radius:12px; padding:1.2rem;
              text-align:center;">
    <div style="font-size:0.7rem; font-weight:500;
                color:#94A3B8; text-transform:uppercase;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;">
      Avg Energy Cost
    </div>
    <div style="font-size:2rem; font-weight:700;
                color:#FBBF24;">
      ${energy_cost:,.0f}
    </div>
    <div style="font-size:0.75rem; color:#64748B;
                margin-top:0.3rem;">
      Yearly expenditure
    </div>
  </div>

  <div style="flex:1; background:{risk_bg};
              border:2px solid {risk_border};
              border-radius:12px; padding:1.2rem;
              text-align:center;">
    <div style="font-size:0.7rem; font-weight:500;
                color:#94A3B8; text-transform:uppercase;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;">
      Risk Level
    </div>
    <div style="font-size:1.6rem; font-weight:700;
                color:{risk_color};">
      {risk_txt}
    </div>
    <div style="font-size:0.75rem; color:#64748B;
                margin-top:0.3rem;">
      ML model prediction
    </div>
  </div>

  <div style="flex:1; background:#1E293B;
              border:1px solid #334155;
              border-radius:12px; padding:1.2rem;
              text-align:center;">
    <div style="font-size:0.7rem; font-weight:500;
                color:#94A3B8; text-transform:uppercase;
                letter-spacing:0.08em;
                margin-bottom:0.5rem;">
      Min Temp Forecast
    </div>
    <div style="font-size:2rem; font-weight:700;
                color:#C084FC;">
      {temp_display}°F
    </div>
    <div style="font-size:0.75rem; color:#64748B;
                margin-top:0.3rem;">
      Projected minimum
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Section 2: Risk Analysis ──────────────────────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>Risk Analysis</h2>"
        "<p style='margin:0;font-size:0.85rem;color:#64748B;'>AI-generated explanation of energy risk factors</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Generating risk explanation..."):
        explanation = _ai_risk_explanation(client, model_name, city_data, feature_impacts, weather_context)

    st.info(explanation)

    with st.expander("View technical SHAP breakdown (for researchers)"):
        shap_display = (
            feature_impacts.reset_index()
            .rename(columns={"index": "Feature", 0: "SHAP Impact"})
        )
        st.dataframe(shap_display, use_container_width=True)

    st.divider()

    # ── Section 3: Active Assistance Programs ─────────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>Active Assistance Programs</h2>"
        "<p style='margin:0;font-size:0.85rem;color:#64748B;'>Programs available to residents based on city income data</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    best = eligible[0] if eligible else None
    prog_rows = [
        {"Program": "LIHEAP", "Income Limit (fam. of 4)": "$66,527",
         "Residents Eligible": _eligible_label(income, 66_527),  "Status": "Active (Seasonal)"},
        {"Program": "CARE",   "Income Limit (fam. of 4)": "$64,300",
         "Residents Eligible": _eligible_label(income, 64_300),  "Status": "Active (Year-round)"},
        {"Program": "FERA",   "Income Limit (fam. of 4)": "$64,301–$80,375",
         "Residents Eligible": "~Some residents" if 64_300 <= income <= 80_375 else "~Few residents",
         "Status": "Active (Year-round)"},
        {"Program": "WAP",    "Income Limit (fam. of 4)": "$64,300",
         "Residents Eligible": _eligible_label(income, 64_300),  "Status": "Active (Limited funding)"},
        {"Program": "ESA",    "Income Limit (fam. of 4)": "$64,300",
         "Residents Eligible": _eligible_label(income, 64_300),  "Status": "Active (Year-round)"},
    ]
    prog_df = pd.DataFrame(prog_rows)

    def _highlight_best(row):
        if best and row["Program"] == best:
            return ["background-color:#F0FDF4;color:#166534"] * len(row)
        return [""] * len(row)

    st.dataframe(
        prog_df.style.apply(_highlight_best, axis=1),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Section 4: Seasonal Risk Pattern + Weather ────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>Seasonal Risk Pattern</h2>"
        "<p style='margin:0;font-size:0.85rem;color:#64748B;'>Historical energy burden estimates — months of highest energy stress</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    months        = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season_mult   = [1.20, 1.20, 0.85, 0.85, 0.85, 1.35,
                     1.35, 1.35, 1.35, 0.85, 0.85, 1.20]
    monthly_b     = [burden * m for m in season_mult]

    if _PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=monthly_b,
            mode="lines+markers",
            name="Energy Burden %",
            line=dict(color="#FF6B6B", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.15)",
        ))
        fig.add_hline(
            y=3.0, line_dash="dash", line_color="red",
            annotation_text="3% Affordability Threshold",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"Monthly Energy Burden Pattern — {city_name}",
            xaxis_title="Month",
            yaxis_title="Energy Burden (%)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1E293B", family="Inter"),
            showlegend=False,
            height=350,
            margin=dict(t=40, b=20),
        )
        fig.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Chart library not available. Run: pip install plotly")

    st.caption(
        "Pattern based on NOAA historical temperature data. "
        "Peaks indicate months of highest energy vulnerability for this city."
    )

    # Projected weather (live Open-Meteo API)
    st.markdown("#### 2-Week Live Temperature Forecast")
    st.caption(
        "Live weather forecast — next 16 days from Open-Meteo. Updates daily."
    )

    # weather_data, overall_stress already fetched at top of function
    if weather_data:
        st.caption(
            "📡 Live forecast from Open-Meteo — updated daily. "
            "Data reflects actual weather conditions for this location."
        )
        weekly = _weekly_summaries(weather_data)

        with st.spinner("Generating forecast summary..."):
            forecast_summary = _ai_forecast_summary(client, model_name, city_name, burden, weekly)

        st.markdown(
            f"<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:6px;"
            f"padding:12px 16px;margin-bottom:12px;color:#1E293B;'>{forecast_summary}</div>",
            unsafe_allow_html=True,
        )

        if _PLOTLY_OK:
            stress_colors = {"HIGH": "#FF4444", "MODERATE": "#FFA500", "LOW": "#44BB44"}
            fig_w = go.Figure(go.Bar(
                x=[w["label"] for w in weekly],
                y=[w["avg_max"] for w in weekly],
                marker_color=[stress_colors.get(w["stress"], "#888") for w in weekly],
                text=[w["stress"] for w in weekly],
                textposition="outside",
            ))
            fig_w.update_layout(
                title="2-Week Live Temperature Forecast",
                xaxis_title="Week",
                yaxis_title="Max Temperature (°F)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#1E293B", family="Inter"),
                showlegend=False,
                height=300,
                margin=dict(t=40, b=20),
            )
            fig_w.update_xaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
            fig_w.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
            st.plotly_chart(fig_w, use_container_width=True)

            for w in weekly:
                emoji = "🔴" if w["stress"] == "HIGH" else "🟡" if w["stress"] == "MODERATE" else "🟢"
                st.markdown(
                    f"{emoji} **Week {w['week']}**: Max {w['max_temp']:.0f}°F, "
                    f"Min {w['min_temp']:.0f}°F — **{w['stress']}** stress"
                )

    else:
        st.warning("Live forecast temporarily unavailable.")

    st.divider()

    # ── Section 5: Policy Recommendations ────────────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>Policy Recommendations</h2>"
        "<p style='margin:0;font-size:0.85rem;color:#64748B;'>AI-generated suggestions for city officials and policymakers</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Generating policy recommendations..."):
        recs = _ai_policy_recs(client, model_name, city_data, feature_impacts,
                               eligible, overall_stress, weather_context)

    col_imm, col_med = st.columns(2)
    with col_imm:
        st.markdown(
            f"<div style='background:#FFF7ED;border-left:4px solid #F59E0B;"
            f"border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;min-height:140px;'>"
            f"<p style='color:#92400E;font-weight:700;margin:0 0 8px;'>Immediate Actions (Next 30 Days)</p>"
            f"<p style='color:#1E293B;font-size:0.9em;line-height:1.7;white-space:pre-line;'>"
            f"{recs.get('immediate','')}</p></div>",
            unsafe_allow_html=True,
        )
    with col_med:
        st.markdown(
            f"<div style='background:#EFF6FF;border-left:4px solid #3B82F6;"
            f"border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;min-height:140px;'>"
            f"<p style='color:#1D4ED8;font-weight:700;margin:0 0 8px;'>Medium Term (Next 6 Months)</p>"
            f"<p style='color:#1E293B;font-size:0.9em;line-height:1.7;white-space:pre-line;'>"
            f"{recs.get('medium_term','')}</p></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='background:#F5F3FF;border-left:4px solid #8B5CF6;"
        f"border-radius:0 12px 12px 0;padding:1.2rem 1.5rem;margin-top:12px;'>"
        f"<p style='color:#6D28D9;font-weight:700;margin:0 0 8px;'>Policy Gaps Identified</p>"
        f"<p style='color:#1E293B;font-size:0.9em;line-height:1.7;white-space:pre-line;'>"
        f"{recs.get('policy_gaps','')}</p></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Section 6: Research Assistant Chat ───────────────────────────────────
    st.markdown(
        "<div style='border-left:3px solid #1E40AF;padding-left:12px;margin-bottom:1rem;'>"
        "<h2 style='margin:0;font-size:1.25rem;font-weight:600;color:#1E293B;'>Research Assistant</h2>"
        "<p style='margin:0;font-size:0.85rem;color:#64748B;'>Ask technical questions about this city's data, compare programs, or explore policy options</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if "r2_chat_messages" not in st.session_state:
        st.session_state["r2_chat_messages"] = []

    # Starter question buttons (only when chat is empty)
    starters = [
        "Why is energy burden above the 3% threshold for this city?",
        "Which program would help the most residents here?",
        "What would happen if household income dropped by 10%?",
    ]
    if not st.session_state["r2_chat_messages"]:
        sq1, sq2, sq3 = st.columns(3)
        for col, q, idx in zip([sq1, sq2, sq3], starters, range(3)):
            with col:
                if st.button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state["r2_pending_q"] = q
                    st.rerun()

    # Render chat history
    for msg in st.session_state["r2_chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", r"\$"))

    # Build researcher system prompt
    top3_detail = "\n".join(
        f"  {k.replace('_', ' ')}: {v:+.4f}"
        for k, v in feature_impacts.head(3).items()
    )
    researcher_system = (
        f"You are GridWatch's research assistant for city officials and policy researchers.\n\n"
        f"City data — {city_name}:\n"
        f"  Energy burden: {burden:.2f}%\n"
        f"  Household income: ${income:,.0f}\n"
        f"  Avg annual energy cost: ${energy_cost:,.0f}\n"
        f"  Risk: {'HIGH' if is_high else 'LOW'}\n"
        f"  Top SHAP drivers:\n{top3_detail}\n"
        f"  Available programs: {', '.join(eligible) if eligible else 'None'}\n\n"
        "Answer technical questions precisely. Cite specific numbers from city data. "
        "If asked to compare with another city, explain you can only speak to the currently selected city."
    )

    pending = st.session_state.pop("r2_pending_q", None)
    user_input = st.chat_input("Ask a research question...") or pending

    if user_input:
        st.session_state["r2_chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        api_msgs = [{"role": "system", "content": researcher_system}]
        api_msgs += [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state["r2_chat_messages"]]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = client.chat.completions.create(
                        model=model_name, messages=api_msgs, temperature=0.3
                    )
                    ans = resp.choices[0].message.content.strip()
                except Exception as e:
                    ans = f"Research assistant temporarily unavailable: {e}"
            st.markdown(ans.replace("$", r"\$"))

        st.session_state["r2_chat_messages"].append({"role": "assistant", "content": ans})

    st.markdown(
        f"<p style='color:#94A3B8;font-size:0.8rem;margin-top:2rem;'>"
        f"Data last updated: {datetime.now().strftime('%B %d, %Y')}</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — called from app.py
# ══════════════════════════════════════════════════════════════════════════════
def show_researcher_dashboard(data_df, shap_df, client, model_name):
    # Init session state
    if "researcher_screen" not in st.session_state:
        st.session_state["researcher_screen"] = 1
    if "researcher_city" not in st.session_state:
        st.session_state["researcher_city"] = None

    screen     = st.session_state.get("researcher_screen", 1)
    city_label = st.session_state.get("researcher_city", "")

    # Persistent top bar
    viewing_part = (
        f'<span style="color:#CBD5E1;margin:0 4px;">|</span>'
        f'<span style="color:#64748B;">Currently viewing: <strong>{city_label}</strong></span>'
        if city_label and screen == 2 else ""
    )
    st.markdown(
        f"<div style='background:white;border:1px solid #E2E8F0;padding:0.75rem 2rem;"
        f"margin-bottom:16px;font-size:0.9rem;border-radius:8px;'>"
        f"<strong style='color:#1E40AF;font-size:1rem;'>GridWatch</strong>"
        f"<span style='color:#CBD5E1;margin:0 8px;'>|</span>"
        f"<span style='color:#64748B;'>Research Dashboard</span>"
        f"{viewing_part}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if screen == 1:
        _show_coverage_overview(data_df)
    else:
        _show_city_dashboard(data_df, shap_df, client, model_name)
