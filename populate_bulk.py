import psycopg2
import datetime
import math
import random

# Database connection parameters (adjust as needed)
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="pressure_db",
    user="user",
    password="password"
)
cursor = conn.cursor()

def generate_pressure(ts):
    # Create a synthetic signal with some daily cyclic behavior plus noise.
    # For example, a sine curve over a day (1440 minutes) scaled down.
    minutes = ts.hour * 60 + ts.minute
    daily_cycle = 10 * math.sin(2 * math.pi * minutes / 1440)
    baseline = 100
    noise = random.uniform(-2, 2)
    return baseline + daily_cycle + noise

def populate_data(days=3, interval_seconds=1800):
    """
    Populate the DB with synthetic data starting from 'days' days ago until now.
    Data is generated every interval_seconds (default: every 30 minutes = 1800 sec).
    """
    start_time = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    end_time = datetime.datetime.utcnow()
    current_time = start_time
    rows = []
    
    while current_time <= end_time:
        p_val = generate_pressure(current_time)
        rows.append((current_time, p_val))
        current_time += datetime.timedelta(seconds=interval_seconds)
        
    insert_query = "INSERT INTO pressure (timestamp, pressure) VALUES (%s, %s)"
    cursor.executemany(insert_query, rows)
    conn.commit()
    print(f"Inserted {len(rows)} rows of synthetic data.")

if __name__ == "__main__":
    populate_data(days=3, interval_seconds=1800)
    cursor.close()
    conn.close()
