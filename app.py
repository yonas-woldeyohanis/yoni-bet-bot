import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Elite Quant Analyst", layout="wide")

# --- 1. THE BRAIN LOADING ---
@st.cache_resource
def load_all_brains():
    models = ['result', 'goals', 'btts', 'home_o15', 'away_o15']
    return {m: joblib.load(f'model_{m}.pkl') for m in models}

brains = load_all_brains()
with open('current_elo.json', 'r') as f:
    elo = json.load(f)
df_stats = pd.read_csv('elite_training_data.csv')

# --- 2. SIDEBAR (WALLET & TRACKER) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/902/902581.png", width=100)
    st.title("🏦 Pro Wallet Manager")
    
    # Wallet Balances
    bybit_bal = st.number_input("Bybit Mastercard ($)", value=50.0, step=1.0)
    telebirr_bal = st.number_input("Telebirr (ETB)", value=5000.0, step=100.0)
    
    active_wallet = st.selectbox("Active Wallet for Bet", ["Bybit", "Telebirr"])
    current_funds = bybit_bal if active_wallet == "Bybit" else telebirr_bal
    currency = "$" if active_wallet == "Bybit" else "ETB"

    st.divider()
    if st.button("📊 View Profit/Loss History"):
        st.info("Performance tracking enabled. Logs saved to 'bet_history.csv'")

# --- 3. MAIN INTERFACE ---
st.title("⚽ Dynamical Football Command Center")

c1, c2 = st.columns(2)
with c1:
    h_team = st.selectbox("🏠 Home Team", sorted(list(elo.keys())))
with c2:
    a_team = st.selectbox("🚀 Away Team", sorted(list(elo.keys())))

if st.button("🔥 Run Professional Multi-Market Analysis"):
    # Get Data
    h_data = df_stats[df_stats['HomeTeam'] == h_team].iloc[-1]
    a_data = df_stats[df_stats['AwayTeam'] == a_team].iloc[-1]
    
    # Feature Vector
    feats = np.array([[elo[h_team], elo[a_team], h_data['H_Conv_Rate'], a_data['A_Conv_Rate'], h_data['Home_BTTS_Rate'], a_data['Away_BTTS_Rate']]])
    
    # Run Inference
    p_res = brains['result'].predict_proba(feats)[0] # [A, D, H]
    p_btts = brains['btts'].predict_proba(feats)[0][1]
    p_goals = brains['goals'].predict_proba(feats)[0][1]
    p_h15 = brains['home_o15'].predict_proba(feats)[0][1]
    p_a15 = brains['away_o15'].predict_proba(feats)[0][1]

    # --- DISPLAY ANALYTICS ---
    st.divider()
    
    t1, t2, t3 = st.tabs(["🎯 Probabilities", "📉 xG & Momentum", "💰 Value Betting"])
    
    with t1:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"{h_team} Win", f"{p_res[2]:.1%}")
        col_b.metric("Draw", f"{p_res[1]:.1%}")
        col_c.metric(f"{a_team} Win", f"{p_res[0]:.1%}")
        
        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Both Teams to Score", f"{p_btts:.1%}")
        col_e.metric("Over 2.5 Goals", f"{p_goals:.1%}")
        col_f.metric(f"{h_team} Over 1.5 Goals", f"{p_h15:.1%}")

    with t2:
        st.subheader("Dynamical Threat Levels (Quant-xG)")
        # Calculate xG based on current conversion rates
        h_xg = 5.5 * h_data['H_Conv_Rate'] # Assume 5.5 shots avg
        a_xg = 5.5 * a_data['A_Conv_Rate']
        
        st.progress(min(h_xg/3, 1.0), text=f"{h_team} Expected Goals (xG): {h_xg:.2f}")
        st.progress(min(a_xg/3, 1.0), text=f"{a_team} Expected Goals (xG): {a_xg:.2f}")
        st.info("Teams with xG > 1.50 are considered 'High Offensive Threat'.")

    with t3:
        st.subheader("Calculated Value Bets")
        
        # User inputs odds for the market they are interested in
        market = st.selectbox("Select Market to Check Value", ["Match Result", "BTTS Yes", "Over 2.5 Goals", "Home Over 1.5"])
        odds = st.number_input("Enter 1xBet/Telebirr Odds:", value=2.0, step=0.01)
        
        # Determine probability based on selection
        m_prob = 0
        if market == "Match Result": m_prob = p_res[2]
        if market == "BTTS Yes": m_prob = p_btts
        if market == "Over 2.5 Goals": m_prob = p_goals
        if market == "Home Over 1.5": m_prob = p_h15
        
        edge = (m_prob * odds) - 1
        
        if edge > 0.05:
            st.success(f"✅ VALUE DETECTED! Edge: {edge:.1%}")
            # Kelly Criterion (1/8th for Professional Safety)
            b = odds - 1
            kelly = ((b * m_prob) - (1 - m_prob)) / b
            bet_size = (kelly / 8) * current_funds
            st.warning(f"💡 STRATEGY: Bet {currency}{bet_size:.2f} using {active_wallet}")
            
            if st.button("📝 Log this Bet to History"):
                log_entry = f"{datetime.now()}, {h_team} vs {a_team}, {market}, {odds}, {bet_size}, {active_wallet}\n"
                with open("bet_history.csv", "a") as f:
                    f.write(log_entry)
                st.balloons()
        else:
            st.error(f"❌ NO VALUE. (Edge: {edge:.1%}). The bookmaker's price is too low.")