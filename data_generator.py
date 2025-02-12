import time
import random
import math
import datetime
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="pressure_db",
    user="user",
    password="password"
)
cursor = conn.cursor()

def generate_pressure(ts):
    minutes = ts.hour * 60 + ts.minute
    daily_cycle = 10 * math.sin(2 * math.pi * minutes / 1440)
    baseline = 100
    noise = random.uniform(-2, 2)
    return baseline + daily_cycle + noise

# For acceleration, keep track of the simulated time
simulated_time = datetime.datetime.utcnow()

def insert_pressure(simulated_time):
    pressure_value = generate_pressure(simulated_time)
    cursor.execute(
        "INSERT INTO pressure (timestamp, pressure) VALUES (%s, %s)",
        (simulated_time, pressure_value)
    )
    conn.commit()
    print(f"Inserted pressure: {pressure_value:.2f} at {simulated_time}")

if __name__ == "__main__":
    try:
        while True:
            insert_pressure(simulated_time)
            # Instead of waiting 30 minutes, update simulated time by 30 minutes every 5 seconds
            simulated_time += datetime.timedelta(minutes=30)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping accelerated data generator...")
    finally:
        cursor.close()
        conn.close()
