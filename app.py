import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="QUANT-X TERMINAL", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stButton > button { background-color: #1e2130; color: #00ffcc; border: 1px solid #3e445e; border-radius: 10px; height: 50px; font-weight: bold; }
    .report-card { background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #3e445e; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        models = ['result', 'goals', 'btts', 'corners']
        brains = {m: joblib.load(f'model_{m}.pkl') for m in models}
        with open('current_elo.json', 'r') as f:
            elo = json.load(f)
        with open('team_stats_snapshot.json', 'r') as f:
            stats = json.load(f)
        with open('upcoming_matches.json', 'r') as f:
            fixtures = json.load(f)
        return brains, elo, stats, fixtures
    except Exception as e:
        st.error(f"Critical System Error: {e}")
        return None, None, None, None

brains, elo_ratings, team_stats, upcoming_matches = load_assets()

# --- HELPER: Professional Naming Safety ---
def get_safe_stats(name):
    # If team exists, return stats. If not (like Cardiff), return League Averages
    if name in team_stats:
        return team_stats[name]
    return {'goals': 1.3, 'corners': 4.8, 'eff': 0.10, 'btts': 0.52}

# --- MAIN UI ---
st.title("🛡️ QUANT-X PRO TERMINAL")
st.sidebar.title("🏦 WALLET MANAGER")
wallet = st.sidebar.selectbox("Wallet", ["Bybit (USDT)", "Telebirr (ETB)"])
balance = st.sidebar.number_input("Capital", value=100.0 if wallet == "Bybit" else 5000.0)

# --- DAILY SCHEDULE ---
st.subheader("🏟️ Select Match for Deep Analysis")
if upcoming_matches:
    cols = st.columns(3)
    for i, match in enumerate(upcoming_matches):
        with cols[i % 3]:
            if st.button(f"{match['home']} vs {match['away']}", key=f"btn_{i}"):
                st.session_state.selected_match = match

# --- DYNAMICAL ANALYSIS ---
if 'selected_match' in st.session_state:
    m = st.session_state.selected_match
    h_team, a_team = m['home'], m['away']
    
    h_snap = get_safe_stats(h_team)
    a_snap = get_safe_stats(a_team)

    # Prepare Data
    feats = np.array([[
        elo_ratings.get(h_team, 1500), elo_ratings.get(a_team, 1500),
        h_snap['goals'], a_snap['goals'], h_snap['corners'], a_snap['corners'],
        h_snap['eff'], a_snap['eff'], h_snap['btts'], a_snap['btts']
    ]])

    # Inference
    p_res = brains['result'].predict_proba(feats)[0] # [A, D, H]
    p_goals = brains['goals'].predict_proba(feats)[0][1]
    p_btts = brains['btts'].predict_proba(feats)[0][1]
    p_corn = brains['corners'].predict_proba(feats)[0][1]

    st.divider()
    st.header(f"🔍 Analysis Report: {h_team} vs {a_team}")

    # PROBABILITY TILES
    t1, t2, t3, t4 = st.columns(4)
    t1.metric(f"🏠 {h_team}", f"{p_res[2]:.1%}")
    t2.metric("⚖️ DRAW", f"{p_res[1]:.1%}")
    t3.metric(f"🚀 {a_team}", f"{p_res[0]:.1%}")
    t4.metric("⚽ BTTS", f"{p_btts:.1%}")

    # THE TWO GRAPHS (PIE AND BAR)
    st.divider()
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.write("### 📈 Winning Probability")
        fig_pie = px.pie(
            values=[p_res[2], p_res[1], p_res[0]], 
            names=[h_team, 'Draw', a_team], 
            hole=0.4, 
            color_discrete_sequence=['#00ffcc', '#3e445e', '#ff4b4b']
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_right:
        st.write("### 📊 Market Volatility (Momentum)")
        m_labels = ['Goals (>2.5)', 'Corners (>9.5)', 'BTTS (Yes)']
        m_values = [p_goals, p_corn, p_btts]
        fig_bar = go.Figure(data=[go.Bar(
            x=m_labels, y=m_values, 
            marker_color='#00ffcc',
            text=[f"{v:.1%}" for v in m_values],
            textposition='auto'
        )])
        fig_bar.update_layout(yaxis_range=[0,1], plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ADVANCED MARKETS
    st.subheader("📋 Advanced Market Intelligence")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.write("**Draw No Bet (DNB)**")
        st.write(f"{h_team}: { (p_res[2]/(p_res[2]+p_res[0])):.1%}")
        st.write(f"{a_team}: { (p_res[0]/(p_res[2]+p_res[0])):.1%}")
    with m2:
        st.write("**Expected Goals (xG) Threat**")
        st.write(f"{h_team}: {h_snap['goals']:.2f}")
        st.write(f"{a_team}: {a_snap['goals']:.2f}")
    with m3:
        odds = st.number_input("Bookie Odds", value=2.0)
        edge = (max(p_res) * odds) - 1
        if edge > 0:
            st.success(f"VALUE: {edge:.1%}")
            st.write(f"Stake: {max(0.20, (( (odds-1)*max(p_res) - (1-max(p_res)) )/(odds-1)/8)*balance):.2f}")
        else:
            st.error("No Value Detected")