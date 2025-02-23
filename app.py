# ----------------------------------------------------------------
# File: app.py
# ----------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objs as go
import datetime

# Statsmodels for Fourier Regression & confidence intervals
import statsmodels.api as sm

from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------------------
# Constants (hard-coded)
SMOOTHING_WINDOW = 3
SEASONAL_PERIOD = 100
NUM_HARMONICS = 3

# ----------------------------------------------------------------
# Auto-refresh every 5 seconds (for live DB updates, if any)
st_autorefresh(interval=1000, key="datarefresh")

# ----------------------------------------------------------------
# Title and brief description
st.title("How Machine Learning Can Help in an Industrial Setting: Time Series Demo")
st.markdown("""
This demo shows how a **Fourier-based regression** can be applied to **reservoir pressure data** 
with seasonal patterns. We simulate spikes and apply a model that captures sinusoidal behavior 
via Fourier features.
""")

# ----------------------------------------------------------------
# Session state for spike simulation
if "cumulative_adjustment" not in st.session_state:
    st.session_state.cumulative_adjustment = 0
if "spike_start_reading" not in st.session_state:
    st.session_state.spike_start_reading = None

# ----------------------------------------------------------------
# Database connection (cached)
@st.cache_resource(show_spinner=False)
def get_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="pressure_db",
            user="user",
            password="password"
        )
        return conn
    except Exception as e:
        st.error("Error connecting to database:")
        st.error(e)
        return None

conn = get_connection()

# ----------------------------------------------------------------
# Sidebar: Only Forecast Horizon
st.sidebar.title("Options & Controls")
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (# future readings)", 
    min_value=1, 
    max_value=2000, 
    value=10
)

# Buttons: Simulate spike or normalize
if st.sidebar.button("Simulate Pressure Spike"):
    st.session_state.cumulative_adjustment += 20
    query = "SELECT COALESCE(MAX(id), 0) FROM pressure"
    cur = conn.cursor()
    cur.execute(query)
    current_max = cur.fetchone()[0]
    cur.close()
    st.session_state.spike_start_reading = current_max + 1
    st.sidebar.success(f"Pressure spike simulated! (+{st.session_state.cumulative_adjustment} total)")

if st.sidebar.button("Normalize Pressure"):
    st.session_state.cumulative_adjustment = 0
    st.session_state.spike_start_reading = None
    st.sidebar.success("Pressure normalization activated!")

# Show DB connection status
if conn:
    st.sidebar.success("Connected to DB")
else:
    st.sidebar.error("Failed to connect to DB")

# ----------------------------------------------------------------
# Load data from the database (cached)
@st.cache_data(ttl=5)
def load_data():
    query = "SELECT id, pressure FROM pressure ORDER BY id ASC"
    df = pd.read_sql(query, conn)
    return df

df = load_data()

if df.empty:
    st.write("No data available yet...")
else:
    df = df.sort_values("id")

    # Apply spike adjustments
    if st.session_state.spike_start_reading is not None:
        df.loc[df['id'] >= st.session_state.spike_start_reading, 'pressure'] += st.session_state.cumulative_adjustment

    # Smoothing
    df['smoothed_pressure'] = df['pressure'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # ------------------------------------------------------------
    # Prepare training data
    X_raw = df[['id']].values
    y = df['pressure'].values

    # Future horizon
    last_id = df['id'].max()
    future_readings = np.arange(last_id + 1, last_id + forecast_horizon + 1).reshape(-1, 1)

    # ------------------------------------------------------------
    # Fourier feature creation
    def create_fourier_features(X, period, num_harmonics):
        """
        Return columns: [1, X, sin(1x), cos(1x), sin(2x), cos(2x), ..., sin(kx), cos(kx)]
        We'll add a constant later via statsmodels, so we won't add it here.
        """
        X = X.ravel()
        features = [X]  # keep raw ID as a feature
        for k in range(1, num_harmonics + 1):
            features.append(np.sin(2.0 * np.pi * k * X / period))
            features.append(np.cos(2.0 * np.pi * k * X / period))
        return np.column_stack(features)

    X_fourier = create_fourier_features(X_raw, SEASONAL_PERIOD, NUM_HARMONICS)
    future_fourier = create_fourier_features(future_readings, SEASONAL_PERIOD, NUM_HARMONICS)

    # ------------------------------------------------------------
    # Single Model: Fourier Regression with statsmodels

    # 1) Add constant column for intercept
    X_fourier_sm = sm.add_constant(X_fourier)
    future_fourier_sm = sm.add_constant(future_fourier)

    # 2) Fit model
    model = sm.OLS(y, X_fourier_sm).fit()

    # 3) Predict + confidence intervals
    forecast_res = model.get_prediction(future_fourier_sm)
    y_pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)  # 95% CI
    y_pred_lower = conf_int[:, 0]
    y_pred_upper = conf_int[:, 1]

    # ------------------------------------------------------------
    # Plotly figure
    fig = go.Figure()

    # Raw data
    fig.add_trace(go.Scatter(
        x=df['id'],
        y=df['pressure'],
        mode='lines+markers',
        name='Raw Data',
        line=dict(color='lightgray', width=2),
        marker=dict(size=5)
    ))

    # Smoothed data
    fig.add_trace(go.Scatter(
        x=df['id'],
        y=df['smoothed_pressure'],
        mode='lines',
        name=f"Smoothed (Window={SMOOTHING_WINDOW})",
        line=dict(color='black', width=3)
    ))

    # Fourier Regression Mean
    fig.add_trace(go.Scatter(
        x=future_readings.flatten(),
        y=y_pred_mean,
        mode='lines',
        name="Fourier Regression",
        line=dict(color='orange', width=3),
        line_shape='spline'
    ))

    # Confidence Interval (fill between lower/upper)
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_readings.flatten(), future_readings.flatten()[::-1]]),
        y=np.concatenate([y_pred_upper, y_pred_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',  # orange w/ transparency
        line=dict(color='rgba(255, 165, 0, 0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=f"Industrial-Grade Time Series Prediction (Fourier Regression): Next {forecast_horizon} Readings",
        xaxis_title="Reading Number",
        yaxis_title="Pressure",
        legend_title="Data & Prediction",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
