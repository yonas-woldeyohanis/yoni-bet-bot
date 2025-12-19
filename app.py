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


# --- LIVE FIXTURE LOADER ---
try:
    with open('upcoming_matches.json', 'r') as f:
        upcoming_data = json.load(f)
except:
    upcoming_data = []

st.subheader("📅 Upcoming Live Fixtures")
if upcoming_data:
    selected_match_label = st.selectbox("Select a Match to Analyze", 
        [f"{m['home']} vs {m['away']}" for m in upcoming_data])
    
    # Extract team names from selection
    h_team, a_team = selected_match_label.split(" vs ")
else:
    st.warning("No live fixtures found. Run 12_live_fetcher.py")
    h_team = st.selectbox("🏠 Manual Home", sorted(list(elo_ratings.keys())))
    a_team = st.selectbox("🚀 Manual Away", sorted(list(elo_ratings.keys())))

# --- 3. MAIN DASHBOARD ---
st.title("🛡️ QUANT-X: PROFESSIONAL MATCH INTELLIGENCE")

col_match1, col_match2 = st.columns(2)
with col_match1:
    h_team = st.selectbox("🏠 HOME SIDE", sorted(list(elo_ratings.keys())))
with col_match2:
    a_team = st.selectbox("🚀 AWAY SIDE", sorted(list(elo_ratings.keys())))

if st.button("⚡ EXECUTE DYNAMICAL ANALYSIS"):
    try:
        h_snap = team_stats[h_team]
        a_snap = team_stats[a_team]
        
        # THE 10 FEATURE VECTOR
        feats = np.array([[
            elo_ratings[h_team], elo_ratings[a_team], # 1, 2
            h_snap['goals'], a_snap['goals'],         # 3, 4
            h_snap['corners'], a_snap['corners'],     # 5, 6
            h_snap['eff'], a_snap['eff'],             # 7, 8
            h_snap['btts'], a_snap['btts']            # 9, 10
        ]])

        # Inference
        p_res = brains['result'].predict_proba(feats)[0]
        p_goals = brains['goals'].predict_proba(feats)[0][1]
        p_btts = brains['btts'].predict_proba(feats)[0][1]
        p_corn = brains['corners'].predict_proba(feats)[0][1]

        # --- ROW 1: CORE PROBABILITIES ---
        st.divider()
        
        # FIX: Check if teams are the same
        if h_team == a_team:
            st.warning("⚠️ Warning: You have selected the same team for both sides. Results are for mathematical testing only.")

        m1, m2, m3, m4 = st.columns(4)
        # FIX: Better labels so you don't get confused
        m1.metric(f"🏠 {h_team} Win", f"{p_res[2]:.1%}")
        m2.metric("⚖️ Draw (X)", f"{p_res[1]:.1%}")
        m3.metric(f"🚀 {a_team} Win", f"{p_res[0]:.1%}")
        m4.metric("⚽ BTTS: YES", f"{p_btts:.1%}")

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

        st.subheader("🔍 Elite Market Analysis")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.write("**Goal Prediction**")
            # Pro Tip: Over 1.5 is lower risk than Over 2.5
            p_o15 = p_goals + 0.15 # Approximate O1.5 based on O2.5
            st.metric("Over 1.5 Goals", f"{min(p_o15, 0.99):.1%}")
            st.metric("BTTS (Yes)", f"{p_btts:.1%}")
            
        with col_m2:
            st.write("**Pressure & Corners**")
            st.metric("High Corner Pressure (>9.5)", f"{p_corn:.1%}")
            # Individual team goal threat
            h_threat = h_snap['goals'] * h_snap['eff']
            st.write(f"🏠 {h_team} Threat Score: {h_threat:.2f}")

        # --- THE PRO SUMMARY ---
        st.divider()
        if p_corn > 0.70 and p_goals > 0.60:
            st.success("🔥 DYNAMICAL SIGNAL: High pressure match. Recommended Market: Over 8.5 Corners + Over 1.5 Goals.")
        elif p_res[1] > 0.35:
            st.info("⚖️ TIGHT MATCH: High Draw probability. Recommended: Draw No Bet (DNB) or Handicap (+0.5).")

        

        # --- ROW 4: DYNAMICAL SCOUTING REPORT ---
        st.subheader("📋 Professional Scouting Report")
        s1, s2 = st.columns(2)
        
        with s1:
            st.write(f"**{h_team} Offensive Efficiency**")
            # Higher efficiency means they need FEWER shots to score.
            h_eff_score = "Elite" if h_snap['eff'] > 0.15 else "Average"
            st.info(f"Rating: {h_eff_score} ({h_snap['eff']:.1%})")
            
        with s2:
            st.write(f"**{a_team} BTTS Trend**")
            # High BTTS rate means they have a "leaky" defense but good attack.
            btts_trend = "Very High" if a_snap['btts'] > 0.60 else "Stable"
            st.info(f"Trend: {btts_trend} ({a_snap['btts']:.1%})")

        # --- THE FINAL VERDICT ---
        st.divider()
        if p_res[2] > 0.55 and h_snap['eff'] > 0.12:
            st.success(f"💎 HIGH CONFIDENCE: {h_team} is statistically dominant at home. Recommended: Straight Win (W1).")
        elif p_btts > 0.65:
            st.success("🔥 GOAL SIGNAL: Both teams show aggressive attacking patterns. Recommended: BTTS (Yes).")


        

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