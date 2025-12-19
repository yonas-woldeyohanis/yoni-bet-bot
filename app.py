import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="QUANT-X TERMINAL", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton > button { width: 100%; height: 50px; border-radius: 10px; }
    .report-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #3e445e; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    models = ['result', 'goals', 'btts', 'corners']
    brains = {m: joblib.load(f'model_{m}.pkl') for m in models}
    with open('current_elo.json', 'r') as f:
        elo = json.load(f)
    with open('team_stats_snapshot.json', 'r') as f:
        stats = json.load(f)
    try:
        with open('upcoming_matches.json', 'r') as f:
            fixtures = json.load(f)
    except:
        fixtures = []
    return brains, elo, stats, fixtures

brains, elo_ratings, team_stats, upcoming_matches = load_assets()

# --- HELPER: Naming Safety ---
def get_team_data(name):
    if name in team_stats: return team_stats[name]
    # Try simple search if name varies (e.g. "Port Vale FC" vs "Port Vale")
    for key in team_stats.keys():
        if name in key or key in name: return team_stats[key]
    return None

# --- MAIN UI ---
st.title("🛡️ QUANT-X PRO TERMINAL")

# 1. FIXTURE GRID
st.subheader("🏟️ Daily Schedule")
if not upcoming_matches:
    st.warning("No fixtures found. Please run the update script.")

cols = st.columns(3)
for i, match in enumerate(upcoming_matches):
    with cols[i % 3]:
        if st.button(f"{match['home']} vs {match['away']}", key=f"m_{i}"):
            st.session_state.selected_match = match

# 2. ANALYSIS
if 'selected_match' in st.session_state:
    m = st.session_state.selected_match
    h_team, a_team = m['home'], m['away']
    
    h_snap = get_team_data(h_team)
    a_snap = get_team_data(a_team)

    if not h_snap or not a_snap:
        st.error(f"Data missing for {h_team} or {a_team}. Check naming in CSV.")
    else:
        # DATA PREP
        feats = np.array([[
            elo_ratings.get(h_team, 1500), elo_ratings.get(a_team, 1500),
            h_snap['goals'], a_snap['goals'], h_snap['corners'], a_snap['corners'],
            h_snap['eff'], a_snap['eff'], h_snap['btts'], a_snap['btts']
        ]])

        p_res = brains['result'].predict_proba(feats)[0] # [A, D, H]
        p_goals = brains['goals'].predict_proba(feats)[0][1]
        p_btts = brains['btts'].predict_proba(feats)[0][1]
        p_corn = brains['corners'].predict_proba(feats)[0][1]

        st.divider()
        st.header(f"🔍 Analysis: {h_team} vs {a_team}")

        # METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Home Win", f"{p_res[2]:.1%}")
        c2.metric("Draw", f"{p_res[1]:.1%}")
        c3.metric("Away Win", f"{p_res[0]:.1%}")
        c4.metric("BTTS Yes", f"{p_btts:.1%}")

        # CHARTS
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.pie(values=[p_res[2], p_res[1], p_res[0]], names=['Home', 'Draw', 'Away'], 
                         hole=0.4, title="Win Probability", color_discrete_sequence=['#00ffcc', '#3e445e', '#ff4b4b'])
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            st.write("### 📊 Pro Markets")
            st.write(f"**Draw No Bet (DNB) {h_team}:** { (p_res[2]/(p_res[2]+p_res[0])):.1%}")
            st.write(f"**Over 2.5 Goals:** {p_goals:.1%}")
            st.write(f"**Over 9.5 Corners:** {p_corn:.1%}")

        # VALUE CALC
        st.sidebar.subheader("💰 Value Checker")
        odds = st.sidebar.number_input("Bookie Odds", value=2.0)
        edge = (max(p_res) * odds) - 1
        if edge > 0.02:
            st.sidebar.success(f"VALUE FOUND: {edge:.1%}")
            # Kelly
            st.sidebar.write(f"Suggested Bet: ${( ( (odds-1)*max(p_res) - (1-max(p_res)) ) / (odds-1) / 8) * 100:.2f}")
        else:
            st.sidebar.error("No Value Detected")