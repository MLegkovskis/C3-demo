import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objs as go
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from streamlit_autorefresh import st_autorefresh

# ----------------------------------------------------------------
# Auto-refresh every 5 seconds (for live DB updates, if any)
st_autorefresh(interval=5000, key="datarefresh")

# ----------------------------------------------------------------
# Title and brief description
st.title("How Machine Learning Can Help in an Industrial Setting: Time Series Demo")
st.markdown("""
This demo shows how ML models can be applied to **reservoir pressure data** with seasonal patterns.
We simulate spikes and apply models that capture sinusoidal behavior via Fourier features.
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
# Sidebar: Controls
st.sidebar.title("Options & Controls")

smoothing_window = st.sidebar.slider("Smoothing Window (# data points)", 
                                     min_value=1, max_value=10, value=3)
forecast_horizon = st.sidebar.slider("Forecast Horizon (# future readings)", 
                                     min_value=1, max_value=2000, value=10)
seasonal_period = st.sidebar.slider("Seasonal Period (# readings)", 
                                    min_value=1, max_value=1000, value=24)
num_harmonics = st.sidebar.slider("Number of Fourier Harmonics", 
                                  min_value=1, max_value=10, value=3)

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
    df['smoothed_pressure'] = df['pressure'].rolling(window=smoothing_window, min_periods=1).mean()

    # ------------------------------------------------------------
    # Training data
    X_raw = df[['id']].values
    y = df['pressure'].values

    # Future horizon
    last_id = df['id'].max()
    future_readings = np.arange(last_id + 1, last_id + forecast_horizon + 1).reshape(-1, 1)

    # ------------------------------------------------------------
    # Fourier feature creation
    def create_fourier_features(X, period, num_harmonics):
        """
        Return columns: [X, sin(1x), cos(1x), sin(2x), cos(2x), ..., sin(kx), cos(kx)]
        """
        X = X.ravel()
        features = [X]  # keep raw ID as a feature
        for k in range(1, num_harmonics + 1):
            features.append(np.sin(2.0 * np.pi * k * X / period))
            features.append(np.cos(2.0 * np.pi * k * X / period))
        return np.column_stack(features)

    X_fourier = create_fourier_features(X_raw, seasonal_period, num_harmonics)
    future_fourier = create_fourier_features(future_readings, seasonal_period, num_harmonics)

    # ------------------------------------------------------------
    # Define models (no plain Linear Regression)
    models = {
        "SVR (RBF)": SVR(kernel='rbf', C=100, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Fourier Regression": LinearRegression(),  # A simple linear model on Fourier features
    }

    colors = {
        "SVR (RBF)": "green",
        "Random Forest": "purple",
        "Fourier Regression": "orange"
    }

    # ------------------------------------------------------------
    # Train each model & predict
    predictions = {}
    for name, model in models.items():
        try:
            model.fit(X_fourier, y)
            y_pred = model.predict(future_fourier)
            predictions[name] = y_pred
        except Exception as e:
            st.error(f"Error training model {name}: {e}")
            predictions[name] = np.zeros(forecast_horizon)

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
        name=f"Smoothed (Window={smoothing_window})",
        line=dict(color='black', width=3)
    ))

    # Predictions (smooth lines, no dashes)
    for name, y_pred in predictions.items():
        fig.add_trace(go.Scatter(
            x=future_readings.flatten(),
            y=y_pred,
            mode='lines',
            name=name,
            line=dict(color=colors[name], width=3),
            line_shape='spline'  # makes the line look smoothly curved
        ))

    fig.update_layout(
        title=f"Industrial-Grade Time Series Predictions: Next {forecast_horizon} Readings",
        xaxis_title="Reading Number",
        yaxis_title="Pressure",
        legend_title="Data & Predictions",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
