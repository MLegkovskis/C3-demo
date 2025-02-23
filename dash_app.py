import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import psycopg2
import statsmodels.api as sm
import datetime

# ---------------------------
# Constants (hard-coded)
SMOOTHING_WINDOW = 3
SEASONAL_PERIOD = 100
NUM_HARMONICS = 3

# ---------------------------
# Database connection function
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
        print("Error connecting to DB:", e)
        return None

# ---------------------------
# Data loading function
def load_data():
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()  # return empty DataFrame if connection fails
    query = "SELECT id, pressure FROM pressure ORDER BY id ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------------------------
# Fourier feature creation
def create_fourier_features(X, period, num_harmonics):
    X = X.ravel()
    features = [X]  # include the raw id as a feature
    for k in range(1, num_harmonics + 1):
        features.append(np.sin(2.0 * np.pi * k * X / period))
        features.append(np.cos(2.0 * np.pi * k * X / period))
    return np.column_stack(features)

# ---------------------------
# Generate the figure based on current data and spike simulation state
def generate_figure(forecast_horizon, store_data):
    df = load_data()
    if df.empty:
        return go.Figure()

    df.sort_values("id", inplace=True)

    # Apply spike adjustments if a spike has been simulated
    cumulative_adjustment = store_data.get("cumulative_adjustment", 0)
    spike_start_reading = store_data.get("spike_start_reading", None)
    if spike_start_reading is not None:
        df.loc[df['id'] >= spike_start_reading, 'pressure'] += cumulative_adjustment

    # Apply smoothing
    df['smoothed_pressure'] = df['pressure'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Prepare training data
    X_raw = df[['id']].values
    y = df['pressure'].values

    last_id = df['id'].max()
    future_readings = np.arange(last_id + 1, last_id + forecast_horizon + 1).reshape(-1, 1)

    # Create Fourier features
    X_fourier = create_fourier_features(X_raw, SEASONAL_PERIOD, NUM_HARMONICS)
    future_fourier = create_fourier_features(future_readings, SEASONAL_PERIOD, NUM_HARMONICS)

    # Fit Fourier Regression using statsmodels
    X_fourier_sm = sm.add_constant(X_fourier)
    future_fourier_sm = sm.add_constant(future_fourier)
    model = sm.OLS(y, X_fourier_sm).fit()
    forecast_res = model.get_prediction(future_fourier_sm)
    y_pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)
    y_pred_lower = conf_int[:, 0]
    y_pred_upper = conf_int[:, 1]

    # Build the Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['id'],
        y=df['pressure'],
        mode='lines+markers',
        name='Raw Data',
        line=dict(color='lightgray', width=2),
        marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=df['id'],
        y=df['smoothed_pressure'],
        mode='lines',
        name=f"Smoothed (Window={SMOOTHING_WINDOW})",
        line=dict(color='black', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=future_readings.flatten(),
        y=y_pred_mean,
        mode='lines',
        name="Fourier Regression",
        line=dict(color='orange', width=3),
        line_shape='spline'
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_readings.flatten(), future_readings.flatten()[::-1]]),
        y=np.concatenate([y_pred_upper, y_pred_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
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
    return fig

# ---------------------------
# Create the Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("How Machine Learning Can Help in an Industrial Setting: Time Series Demo"),
    dcc.Markdown("""
This demo shows how a **Fourier-based regression** can be applied to **reservoir pressure data** 
with seasonal patterns. We simulate spikes and apply a model that captures sinusoidal behavior 
via Fourier features.
    """),
    html.Div([
        html.Label("Forecast Horizon (# future readings)"),
        dcc.Slider(
            id='forecast-horizon-slider',
            min=1,
            max=2000,
            value=10,
            step=1,
            marks={i: str(i) for i in range(0, 2001, 250)}
        )
    ], style={'margin': '20px'}),
    html.Div([
        html.Button("Simulate Pressure Spike", id="simulate-button", n_clicks=0,
                    style={'marginRight': '10px'}),
        html.Button("Normalize Pressure", id="normalize-button", n_clicks=0)
    ], style={'margin': '20px'}),
    dcc.Graph(id='timeseries-graph'),
    # Update the graph every 5 seconds
    dcc.Interval(
        id='interval-component',
        interval=0.5 * 1000,  # 5 seconds in milliseconds
        n_intervals=0
    ),
    # Store to keep spike simulation state
    dcc.Store(id='store-data', data={"cumulative_adjustment": 0, "spike_start_reading": None})
])

# ---------------------------
# Callback to update the spike simulation state (stored in dcc.Store)
@app.callback(
    Output('store-data', 'data'),
    [Input('simulate-button', 'n_clicks'),
     Input('normalize-button', 'n_clicks')],
    [State('store-data', 'data')]
)
def update_store(simulate_clicks, normalize_clicks, current_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_store
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    cumulative_adjustment = current_store.get("cumulative_adjustment", 0)
    spike_start_reading = current_store.get("spike_start_reading", None)

    if button_id == "simulate-button":
        # When simulating, add 20 to the cumulative adjustment and set spike_start_reading
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM pressure")
            current_max = cur.fetchone()[0]
            cur.close()
            conn.close()
        except Exception as e:
            print("Error fetching current max id:", e)
            current_max = 0
        cumulative_adjustment += 20
        spike_start_reading = current_max + 1
        return {"cumulative_adjustment": cumulative_adjustment, "spike_start_reading": spike_start_reading}
    elif button_id == "normalize-button":
        # Reset the simulation state
        return {"cumulative_adjustment": 0, "spike_start_reading": None}
    else:
        return current_store

# ---------------------------
# Callback to update the graph based on the forecast horizon, interval ticks, and store data
@app.callback(
    Output('timeseries-graph', 'figure'),
    [Input('forecast-horizon-slider', 'value'),
     Input('interval-component', 'n_intervals'),
     Input('store-data', 'data')]
)
def update_graph(forecast_horizon, n_intervals, store_data):
    fig = generate_figure(forecast_horizon, store_data)
    return fig

# ---------------------------
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
