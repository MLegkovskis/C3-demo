# C3 Demo: Industrial Time Series ML App

This repository contains a minimal end-to-end demonstration of **machine learning applied to time series data** in an industrial setting. We simulate **reservoir pressure** readings, store them in a PostgreSQL database, and visualize & predict future pressures using **Streamlit**.

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Setup & Installation](#setup--installation)  
4. [Running the Demo](#running-the-demo)  
5. [How It Works](#how-it-works)  
6. [Notes & Customizations](#notes--customizations)

---

## Overview

1. **Database**: A PostgreSQL container stores all pressure readings.  
2. **Data Generator**: Python scripts (`data_generator.py`, `populate_bulk.py`) insert synthetic data that follows a daily sinusoidal cycle plus noise.  
3. **Streamlit App** (`app.py`): Connects to the DB, retrieves data, and performs:  
   - **Anomaly Simulations**: You can simulate spikes or reset to normal pressure.  
   - **Smoothing & Forecasting**: Several ML models (SVR, Random Forest, and a Fourier-based Linear Regression) predict future pressures and display interactive plots.  

The result is a user-friendly dashboard that **continuously updates** and showcases a real-time streaming approach.

---

## Project Structure

```
C3-demo/
│
├── app.py               # Main Streamlit application
├── data_generator.py    # Inserts one row every loop (simulates ongoing real-time data)
├── docker-compose.yml   # Defines the PostgreSQL service
├── init_db.sql          # Automatically creates the "pressure" table on DB startup
├── populate_bulk.py     # Inserts bulk synthetic data (e.g. 3 days' worth) at once
└── README.md            # Documentation (this file)
```

**Key points**:
- `app.py` runs the Streamlit app on your local machine (not containerized).
- `docker-compose.yml` starts a local Postgres container, exposing port `5432`.
- `init_db.sql` runs at container startup to ensure the `pressure` table exists.
- `data_generator.py` simulates data in an accelerated “real-time” fashion (e.g., every 2 seconds it inserts a reading that is 30 minutes ahead in the “simulated” timeline).
- `populate_bulk.py` is a one-time script that backfills the database with a specified number of days of data.

---

## Setup & Installation

1. **Install Docker** (if you don’t already have it):  
   - [Docker Installation Guide](https://docs.docker.com/get-docker/)

2. **Install Python dependencies**:  
   - Ensure you have Python 3.8+  
   - In your terminal, navigate to `C3-demo/` and run:
     ```bash
     pip install -r requirements.txt
     ```
     *(Note: You may need to create a `requirements.txt` listing `streamlit`, `psycopg2`, `plotly`, `scikit-learn`, etc. If it’s missing, you can manually install them:)*  
     ```bash
     pip install streamlit psycopg2-binary plotly scikit-learn streamlit-autorefresh
     ```

3. **Start the database container**:
   ```bash
   docker-compose up -d
   ```
   This launches Postgres in the background. The DB will be accessible at `localhost:5432` with user=`user`, password=`password`, database=`pressure_db`.

---

## Running the Demo

### 1. Start (or Bulk Populate) the Database

- **Option A**: **Populate historical data** immediately (3 days’ worth by default). From the `C3-demo/` folder:
  ```bash
  python populate_bulk.py
  ```
  You should see a message like: *Inserted XXX rows of synthetic data.*  

- **Option B**: **Stream/Accelerate data** in real-time. From `C3-demo/`:
  ```bash
  python data_generator.py
  ```
  This script will insert new rows every couple seconds, each one simulating 30 minutes in the “accelerated” timeline.

*(You can do both: run `populate_bulk.py` once to pre-fill historical data, then `data_generator.py` to keep new data flowing.)*

### 2. Launch the Streamlit App

In a separate terminal window/tab, run:
```bash
streamlit run app.py
```
This starts the Streamlit server, typically on [http://localhost:8501](http://localhost:8501). You’ll see:

- **Sidebar** controls for smoothing window, forecast horizon, seasonal period, etc.  
- **Buttons** to simulate a spike or normalize the pressure.  
- A live-updating plot with:
  - Historical data (raw vs. smoothed).
  - ML model forecasts (SVR, Random Forest, and a Fourier-based Linear Regression).

### 3. Interact with the Dashboard

1. **Observe the Real-Time Updates**: The app auto-refreshes every 5 seconds, so newly inserted data from `data_generator.py` will appear.  
2. **Simulate Pressure Spike**: Press the button in the sidebar to add a jump in values.  
3. **Normalize Pressure**: Reset any artificial spikes.  
4. **Tweak the Sliders**:
   - **Smoothing Window** changes how historical data is smoothed.  
   - **Forecast Horizon** changes how many future readings you predict.  
   - **Seasonal Period** and **# of Fourier Harmonics** influence how the models learn and replicate sinusoidal seasonality.

---

## How It Works

### Database & Table

The `docker-compose.yml` references `init_db.sql`, ensuring we have a table named `pressure`:
```sql
CREATE TABLE IF NOT EXISTS pressure (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pressure FLOAT NOT NULL
);
```
We store each reading’s **timestamp** and **pressure** in this table.

### Data Generators

Two scripts **create synthetic pressure**:
- `populate_bulk.py`: Inserts *multiple days’ worth* of historical data instantly.  
- `data_generator.py`: Inserts *one record every few seconds*, each representing 30 more minutes of real time.  

In both scripts, a **daily sinusoidal** function is used:
```python
minutes = ts.hour * 60 + ts.minute
daily_cycle = 10 * math.sin(2 * math.pi * minutes / 1440)
baseline = 100
noise = random.uniform(-2, 2)
pressure = baseline + daily_cycle + noise
```
This produces a roughly 24-hour wave with slight randomness.

### Streamlit App & ML

1. **Data Fetch**: A simple SQL query loads all rows from `pressure`, sorted by `id`.  
2. **Spike Simulation**: The session state tracks a “cumulative adjustment” if the user clicks “Simulate Pressure Spike.” This is added on the fly to the displayed data for all subsequent records.  
3. **Smoothing**: We apply a rolling window average over the last `N` data points.  
4. **Feature Engineering**: We create multi-harmonic Fourier features to help the models learn sinusoidal patterns:
   ```python
   sin(k * 2πX / period), cos(k * 2πX / period)
   ```
5. **Models**:  
   - **SVR (RBF)**  
   - **Random Forest**  
   - **Fourier Regression**: A standard LinearRegression on the Fourier features.

6. **Forecasting**: For the next `forecast_horizon` readings, we extrapolate *id* values and generate predictions from each model.  
7. **Interactive Plot**: We use **Plotly** to display raw data, smoothed data, and the forecasts for each model. The chart auto‐updates every 5 seconds.

---

## Notes & Customizations

- You can easily **change the “daily” period** to something else by editing the data generator’s function or the `seasonal_period` default slider.  
- The smoothing or ML modeling can be **extended** with additional regressors, e.g., ARIMA, Prophet, etc.  
- Adjust the intervals in `data_generator.py` to speed up or slow down the flow of new data.  
- **Docker**: Currently, only the database is containerized. The Streamlit app runs on the host. You could further dockerize the Streamlit app if desired.
