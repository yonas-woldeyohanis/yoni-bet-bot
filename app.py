import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- ELITE UI CONFIG ---
st.set_page_config(page_title="QUANT-X TERMINAL", layout="wide", initial_sidebar_state="expanded")

# --- 1. THE BRAIN LOADING ---
@st.cache_resource
def load_all_brains():
    # These must match the filenames exactly as they appear on your GitHub
    models = ['result', 'goals', 'btts', 'home_o15', 'away_o15', 'corners']
    loaded_models = {}
    for m in models:
        try:
            loaded_models[m] = joblib.load(f'model_{m}.pkl')
        except:
            st.error(f"Brain missing: model_{m}.pkl not found on GitHub!")
    return loaded_models

# Initialize brains and data
brains = load_all_brains()

# Load current Elo Ratings
try:
    with open('current_elo.json', 'r') as f:
        elo_ratings = json.load(f)
except FileNotFoundError:
    st.error("Error: current_elo.json is missing. Please push it to GitHub.")

# Load the Lightweight Team Stats Snapshot
try:
    with open('team_stats_snapshot.json', 'r') as f:
        team_stats = json.load(f)
except FileNotFoundError:
    st.warning("Snapshot missing. Run '11_create_snapshot.py' and push 'team_stats_snapshot.json' to GitHub.")

# --- 2. SIDEBAR (WALLET & TRACKER) ---
with st.sidebar:
    st.title("🏦 PORTFOLIO MANAGER")
    wallet = st.selectbox("Betting Wallet", ["Bybit (USDT)", "Telebirr (ETB)", "Bank (CBE)"])
    balance = st.number_input("Total Capital", value=100.0 if wallet == "Bybit" else 10000.0)
    risk_level = st.slider("Risk Tolerance (1=Safe, 5=Aggressive)", 1, 5, 2)
    currency = "$" if wallet == "Bybit" else "ETB"

    st.divider()
    st.subheader("🎯 VALUE CALCULATOR")
    user_odds = st.number_input("Bookie Odds (1xBet/Telebirr):", value=2.0, step=0.01)

# --- 3. MAIN DASHBOARD ---
st.title("🛡️ QUANT-X: PROFESSIONAL MATCH INTELLIGENCE")

col_match1, col_match2 = st.columns(2)
with col_match1:
    h_team = st.selectbox("🏠 HOME SIDE", sorted(list(elo_ratings.keys())))
with col_match2:
    a_team = st.selectbox("🚀 AWAY SIDE", sorted(list(elo_ratings.keys())))

if st.button("⚡ EXECUTE DYNAMICAL ANALYSIS"):
    try:
        # Get Stats from Snapshot
        h_snap = team_stats[h_team]
        a_snap = team_stats[a_team]
        
        # Build Feature Vector for AI
        # Feature order: Elo_H, Elo_A, Conv_H, Conv_A, BTTS_H, BTTS_A
        feats = np.array([[
            elo_ratings[h_team], 
            elo_ratings[a_team], 
            h_snap['conv_rate'], 
            a_snap['conv_rate'], 
            h_snap['btts_rate'], 
            a_snap['btts_rate']
        ]])

        # Multi-Model Inference
        p_res = brains['result'].predict_proba(feats)[0]   # [Away, Draw, Home]
        p_goals = brains['goals'].predict_proba(feats)[0][1]
        p_btts = brains['btts'].predict_proba(feats)[0][1]
        p_corn = brains['corners'].predict_proba(feats)[0][1]

        # --- ROW 1: CORE PROBABILITIES ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{h_team} Win", f"{p_res[2]:.1%}")
        m2.metric("Draw (X)", f"{p_res[1]:.1%}")
        m3.metric(f"{a_team} Win", f"{p_res[0]:.1%}")
        m4.metric("BTTS: YES", f"{p_btts:.1%}")

        # --- ROW 2: ADVANCED MARKETS ---
        st.subheader("📊 ADVANCED PROFESSIONAL MARKETS")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("**Draw No Bet (DNB)**")
            dnb_h = p_res[2] / (p_res[2] + p_res[0])
            dnb_a = p_res[0] / (p_res[2] + p_res[0])
            st.write(f"{h_team}: {dnb_h:.1%}")
            st.write(f"{a_team}: {dnb_a:.1%}")

        with c2:
            st.write("**Double Chance**")
            st.write(f"1X (Home/Draw): {(p_res[2] + p_res[1]):.1%}")
            st.write(f"X2 (Away/Draw): {(p_res[0] + p_res[1]):.1%}")

        with c3:
            st.write("**Asian Handicap (AH 0.0)**")
            ah_val = (elo_ratings[h_team] - elo_ratings[a_team]) / 100
            st.write(f"Projected AH Line: {ah_val:+.2f}")

        # --- ROW 3: VISUALIZATIONS ---
        st.divider()
        viz1, viz2 = st.columns(2)
        
        with viz1:
            st.write("### 📈 Winning Probability")
            fig = px.pie(values=[p_res[2], p_res[1], p_res[0]], 
                         names=['Home', 'Draw', 'Away'], 
                         hole=.4,
                         color_discrete_sequence=['#00cc96', '#636efa', '#ef553b'])
            st.plotly_chart(fig, use_container_width=True)

        with viz2:
            st.write("### 🚩 Momentum & Volatility")
            categories = ['Over 2.5 Goals', 'Over 9.5 Corners', 'BTTS']
            values = [p_goals, p_corn, p_btts]
            fig2 = go.Figure(data=[go.Bar(x=categories, y=values, marker_color='#00ffcc')])
            st.plotly_chart(fig2, use_container_width=True)

        # --- VALUE CHECK ---
        edge = (max(p_res) * user_odds) - 1
        if edge > 0.05:
            st.sidebar.success(f"PROFIT EDGE: {edge:.1%}")
            b = user_odds - 1
            kelly = ((b * max(p_res)) - (1 - max(p_res))) / b
            bet_amt = (kelly / (10/risk_level)) * balance
            st.sidebar.warning(f"💡 SUGGESTED BET: {currency}{max(0.20, bet_amt):.2f}")
        else:
            st.sidebar.info(f"Edge: {edge:.1%}. Wait for better odds.")

    except Exception as e:
        st.error(f"Dynamical Error: {e}")