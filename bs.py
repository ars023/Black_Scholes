import numpy as np
import streamlit as st
from scipy.stats import norm
import plotly.express as px

st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide") # not working for some reason
st.markdown("# Black-Scholes Pricing Model")

def black_Scholes(s, k, sigma, T, r, delta):
    d1 = (np.log(s/k) + (r - delta + (0.5 * (sigma ** 2)) * T)) / (sigma * np.sqrt(T))
    d2 = (np.log(s/k) + (r - delta - (0.5 * (sigma ** 2)) * T)) / (sigma * np.sqrt(T))

    call = s * (np.e ** (-delta*T)) * norm.cdf(d1) - k * (np.e ** (-r*T)) * norm.cdf(d2)
    put = k * (np.e ** (-r*T)) * norm.cdf(-d2) - s * (np.e ** (-delta*T)) * norm.cdf(-d1)

    return call, put


#print(black_Scholes(100, 100, 0.2, 1, 0.05, 0.03))


st.sidebar.title("Black-Scholes Model")
S = st.sidebar.number_input("Current Asset Price", min_value=0.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price",        min_value=0.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.0, value=1.0, step=0.05, format="%.2f")
sigma = st.sidebar.number_input("Volatility (σ)",   min_value=0.001, value=0.20, step=0.01, format="%.2f")
r = st.sidebar.number_input("Risk-Free Interest Rate", min_value=-0.05, value=0.05, step=0.005, format="%.3f")
delta = st.sidebar.number_input("Dividend Yield", min_value=0.0, value=0.00, step=0.005, format="%.3f")


c_val, p_val = black_Scholes(S, K, sigma, T, r, delta)
col1, col2 = st.columns(2)
with col1:
    st.metric("CALL Value", f"${c_val:,.2f}")
with col2:
    st.metric("PUT Value", f"${p_val:,.2f}")

st.markdown("## Options Price — Interactive Heatmap")
st.caption(
    "Explore how option prices fluctuate with varying Spot Price and Volatility at a fixed Strike."
)

S_grid = np.linspace(max(1e-6, 0.2*S), 2.0*S if S > 0 else 200, 25)
sig_grid = np.linspace(0.05, 0.60, 25)

c_mat = np.zeros((25,25))
p_mat = np.zeros((25,25))

for i, Sg in enumerate(S_grid):
    for j, sg in enumerate(sig_grid):
        c_mat[i,j], p_mat[i,j] = black_Scholes(Sg, K, sg, T, r, delta)


cfig = px.imshow(c_mat,
                color_continuous_scale='RdBu_r',
                x=[f"{s:.2f}" for s in sig_grid],
                y=[f"{s:.2f}" for s in S_grid],
                labels=dict(x="Volatility", y="Spot Price", color="Call Price"),
                aspect="auto")

pfig = px.imshow(p_mat,
                color_continuous_scale='RdBu_r',
                x=[f"{s:.2f}" for s in sig_grid],
                y=[f"{s:.2f}" for s in S_grid],
                labels=dict(x="Volatility", y="Spot Price", color="Put Price"),
                aspect="auto")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Call Price Heatmap")
    st.plotly_chart(cfig, use_container_width=True)

with c2:
    st.subheader("Put Price Heatmap")
    st.plotly_chart(pfig, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit • Numpy • SciPy • Plotly")
