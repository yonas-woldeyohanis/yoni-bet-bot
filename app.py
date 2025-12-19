import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="QUANT-X TERMINAL", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR "BETTING CARD" LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #1e2130;
        color: white;
        border: 1px solid #3e445e;
        border-radius: 10px;
        height: 60px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        border-color: #00ffcc;
        color: #00ffcc;
        background-color: #262a3d;
    }
    .metric-container {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #3e445e;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & BRAIN LOADING ---
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

# --- 2. SIDEBAR (PORTFOLIO) ---
with st.sidebar:
    st.title("🏦 WALLET")
    wallet = st.selectbox("Wallet", ["Bybit (USDT)", "Telebirr (ETB)"])
    balance = st.number_input("Balance", value=100.0 if wallet == "Bybit" else 5000.0)
    st.divider()
    st.info("Select a match from the main dashboard to begin analysis.")

# --- 3. MAIN DASHBOARD ---
st.title("🛡️ QUANT-X PRO TERMINAL")
st.write(f"📅 Schedule for {datetime.now().strftime('%A, Dec %d')}")

# Initialize Session State for selected match
if 'home_team' not in st.session_state:
    st.session_state.home_team = None
    st.session_state.away_team = None

# --- MATCH GRID (THE CARDS) ---
st.subheader("🏟️ Select Match to Analyze")
cols = st.columns(2) # Create 2 columns for match cards

for i, match in enumerate(upcoming_matches):
    col_idx = i % 2
    with cols[col_idx]:
        # Professional Match Card Button
        if st.button(f"⚽ {match['home']}  vs  {match['away']}", key=f"match_{i}"):
            st.session_state.home_team = match['home']
            st.session_state.away_team = match['away']

# --- DYNAMICAL ANALYSIS SECTION ---
if st.session_state.home_team:
    h_team = st.session_state.home_team
    a_team = st.session_state.away_team
    
    st.divider()
    st.header(f"🔍 Deep Analysis: {h_team} vs {a_team}")
    
    # 1. RUN INFERENCE
    h_snap = team_stats[h_team]
    a_snap = team_stats[a_team]
    
    # 10-Feature Vector
    feats = np.array([[
        elo_ratings[h_team], elo_ratings[a_team],
        h_snap['goals'], a_snap['goals'],
        h_snap['corners'], a_snap['corners'],
        h_snap['eff'], a_snap['eff'],
        h_snap['btts'], a_snap['btts']
    ]])

    p_res = brains['result'].predict_proba(feats)[0] # [A, D, H]
    p_goals = brains['goals'].predict_proba(feats)[0][1]
    p_btts = brains['btts'].predict_proba(feats)[0][1]
    p_corn = brains['corners'].predict_proba(feats)[0][1]

    # 2. PROBABILITY TILES
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: st.metric(f"🏠 {h_team}", f"{p_res[2]:.1%}")
    with col_b: st.metric("⚖️ DRAW", f"{p_res[1]:.1%}")
    with col_c: st.metric(f"🚀 {a_team}", f"{p_res[0]:.1%}")
    with col_d: st.metric("⚽ BTTS", f"{p_btts:.1%}")

    # 3. ADVANCED MARKETS (REAL USE)
    st.subheader("📊 Professional Markets")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.write("**Draw No Bet (DNB)**")
        dnb_h = p_res[2] / (p_res[2] + p_res[0])
        dnb_a = p_res[0] / (p_res[2] + p_res[0])
        st.write(f"{h_team}: {dnb_h:.1%}")
        st.write(f"{a_team}: {dnb_a:.1%}")

    with m2:
        st.write("**Goal / Corner Pressure**")
        st.write(f"Over 2.5 Goals: {p_goals:.1%}")
        st.write(f"Over 9.5 Corners: {p_corn:.1%}")

    with m3:
        st.write("**Suggested Stake**")
        user_odds = st.number_input("Bookie Odds:", value=2.0, step=0.01)
        # Simple Value Check
        edge = (max(p_res) * user_odds) - 1
        if edge > 0:
            b = user_odds - 1
            kelly = ((b * max(p_res)) - (1 - max(p_res))) / b
            bet = (kelly / 8) * balance
            st.success(f"Bet: {'$' if wallet == 'Bybit' else 'ETB'} {max(0.20, bet):.2f}")
        else:
            st.error("No Value detected.")

    # 4. VOLATILITY CHART
    st.write("### 📈 Match Momentum Chart")
    fig = px.bar(x=['Win', 'Draw', 'Loss', 'BTTS', 'O2.5', 'Corners'], 
                 y=[p_res[2], p_res[1], p_res[0], p_btts, p_goals, p_corn],
                 labels={'x': 'Market', 'y': 'Probability'},
                 color_discrete_sequence=['#00ffcc'])
    st.plotly_chart(fig, use_container_width=True)

    if st.button("❌ Close Analysis"):
        st.session_state.home_team = None
        st.rerun()

else:
    st.info("Welcome back, Yoni. Select a match from the cards above to see the AI scouting report.")