import streamlit as st
import paho.mqtt.client as mqtt
import threading
import queue
import json
from datetime import datetime
import plotly.graph_objects as go
import random
import time
from scipy.stats import ttest_ind
import pandas as pd
import csv
import os

# MQTT Config
BROKER = "b37.mqtt.one"
PORT = 1883
USERNAME = "58citw8880"
PASSWORD = "590degioqu"
TOPIC = "58citw8880/"

CSV_FILE = "pump_real_time_failures.csv"
FIELDNAMES = ["timestamp", "Time", "FaultCode", "AlertMessage", "confidence", "Flow", "Pressure", "flow_std", "pressure_std"]

# Thread-safe queue for MQTT messages
mqtt_queue = queue.Queue()

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        mqtt_queue.put(data)
    except Exception as e:
        print(f"Error decoding MQTT message: {e}")

# MQTT client thread function
def mqtt_loop():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"MQTT connection error: {e}")
        return
    client.loop_forever()

# Initialize CSV file with headers if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

# Initialize Streamlit session state
if "mqtt_thread_started" not in st.session_state:
    thread = threading.Thread(target=mqtt_loop, daemon=True)
    thread.start()
    st.session_state.mqtt_thread_started = True

if "mqtt_data" not in st.session_state:
    st.session_state.mqtt_data = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "flow_values" not in st.session_state:
    st.session_state.flow_values = []

if "pressure_values" not in st.session_state:
    st.session_state.pressure_values = []

if "collection_mode" not in st.session_state:
    st.session_state.collection_mode = "Baseline"

if "baseline_downtime_events" not in st.session_state:
    st.session_state.baseline_downtime_events = []

if "baseline_response_times" not in st.session_state:
    st.session_state.baseline_response_times = []

if "pdm_downtime_events" not in st.session_state:
    st.session_state.pdm_downtime_events = []

if "pdm_response_times" not in st.session_state:
    st.session_state.pdm_response_times = []

if "current_downtime_start" not in st.session_state:
    st.session_state.current_downtime_start = None

if "alert_start_time" not in st.session_state:
    st.session_state.alert_start_time = None

pump_state = {"status": "Running", "action": "None"}

# Move messages from queue to session state and save to CSV
while not mqtt_queue.empty():
    msg = mqtt_queue.get()
    st.session_state.mqtt_data.append(msg)
    # Save to CSV
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({key: msg.get(key, "") for key in FIELDNAMES})
    # Extract data for plotting
    timestamp = msg.get("timestamp") or datetime.now().strftime("%H:%M:%S")
    flow = msg.get("Flow", 0)
    pressure = msg.get("Pressure", 0)
    st.session_state.timestamps.append(timestamp)
    st.session_state.flow_values.append(flow)
    st.session_state.pressure_values.append(pressure)

# Keep only last 20 points
max_points = 20
if len(st.session_state.timestamps) > max_points:
    st.session_state.timestamps = st.session_state.timestamps[-max_points:]
    st.session_state.flow_values = st.session_state.flow_values[-max_points:]
    st.session_state.pressure_values = st.session_state.pressure_values[-max_points:]

# Maintenance Recommendation Logic
def generate_recommendation(data):
    flow = data.get("Flow", 0)
    pressure = data.get("Pressure", 0)
    fault_code = int(str(data.get("FaultCode") or 0))
    alert = data.get("AlertMessage", "")

    recs = []
    fault_binary = format(fault_code, "03b")
    leaking, blocked, friction = map(int, fault_binary)

    if leaking:
        recs.append("🔧 Leaking Cylinder Detected - Inspect seals and replace worn cylinders.")
    if blocked:
        recs.append("🔧 Blocked Inlet Detected - Clean inlet lines and check for clogs.")
    if friction:
        recs.append("🔧 Bearing Friction Detected - Lubricate or replace bearings.")
    if flow < 10:
        recs.append("🛠️ Low flow detected - Could be suction issues or valve wear.")
    if pressure > 80:
        recs.append("⚙️ High pressure detected - Risk of blockage or cavitation.")

    if "Warning" in alert:
        recs.append("⚠️ Warning detected - Immediate inspection recommended.")

    if not recs:
        recs.append("✅ All systems nominal - No faults detected.")

    return recs

# Automation Logic
def automate_response(alert_msg, mode, response_time_range):
    if "Warning" in alert_msg:
        if st.session_state.alert_start_time is None:
            st.session_state.alert_start_time = datetime.now()
        pump_state["status"] = "Slowing Down"
        pump_state["action"] = "Reduce speed to prevent failure"
        if st.session_state.current_downtime_start is None:
            st.session_state.current_downtime_start = datetime.now()
    else:
        if st.session_state.alert_start_time is not None:
            simulated_response = random.uniform(*response_time_range)
            response_time = simulated_response
            if mode == "Baseline":
                st.session_state.baseline_response_times.append(response_time)
            else:
                st.session_state.pdm_response_times.append(response_time)
            st.session_state.alert_start_time = None
        pump_state["status"] = "Running"
        pump_state["action"] = "Normal operation"
        if st.session_state.current_downtime_start is not None:
            downtime_event = {
                "start": st.session_state.current_downtime_start,
                "end": datetime.now()
            }
            if mode == "Baseline":
                st.session_state.baseline_downtime_events.append(downtime_event)
            else:
                st.session_state.pdm_downtime_events.append(downtime_event)
            st.session_state.current_downtime_start = None

# KPI calculation
def calculate_metrics(downtime_events, response_times):
    total_downtime_min = sum([(event['end'] - event['start']).total_seconds() / 60 for event in downtime_events]) if downtime_events else 0
    avg_response_time_min = (sum(response_times) / len(response_times)) if response_times else 0
    num_downtime_events = len(downtime_events)
    return total_downtime_min, avg_response_time_min, num_downtime_events

def calculate_improvement(baseline_value, pdm_value):
    if baseline_value == 0:
        return 100.0 if pdm_value == 0 else 0.0
    return ((baseline_value - pdm_value) / baseline_value) * 100

def perform_t_test(baseline_times, pdm_times):
    if len(baseline_times) < 2 or len(pdm_times) < 2:
        return None, None
    t_stat, p_value = ttest_ind(baseline_times, pdm_times, equal_var=False)
    return t_stat, p_value

def downtime_events_to_df(events):
    data = []
    for i, event in enumerate(events, 1):
        duration = (event['end'] - event['start']).total_seconds() / 60
        data.append({
            "Event Number": i,
            "Start Time": event['start'].strftime("%Y-%m-%d %H:%M:%S"),
            "End Time": event['end'].strftime("%Y-%m-%d %H:%M:%S"),
            "Duration (min)": round(duration, 2)
        })
    return pd.DataFrame(data)

def response_times_to_df(response_times):
    data = []
    for i, rt in enumerate(response_times, 1):
        data.append({
            "Response Number": i,
            "Response Time (min)": round(rt, 2)
        })
    return pd.DataFrame(data)

# Streamlit UI
st.set_page_config(page_title="Triplex Pump Digital Twin", layout="centered")
st.title("🔍⚙️📅 Digital Twin - Triplex Reciprocating Pump")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    collection_mode = st.radio("Data Collection Mode", ["Baseline", "PdM"], index=0 if st.session_state.collection_mode == "Baseline" else 1)
    if collection_mode != st.session_state.collection_mode:
        st.session_state.collection_mode = collection_mode
        st.experimental_rerun()

    if st.button("🔄 Reset Simulation Data"):
        st.session_state.baseline_downtime_events = []
        st.session_state.baseline_response_times = []
        st.session_state.pdm_downtime_events = []
        st.session_state.pdm_response_times = []
        st.session_state.current_downtime_start = None
        st.session_state.alert_start_time = None
        st.success("Simulation data reset.")

# Show MQTT connection status
st.write("### MQTT Connection Status")
if st.session_state.mqtt_thread_started:
    st.success("✅ MQTT client running and subscribed")
else:
    st.warning("❌ MQTT client not running")

# Show latest MQTT data or waiting message
if st.session_state.mqtt_data:
    latest = st.session_state.mqtt_data[-1]
    st.subheader("Latest MQTT Data")
    st.json(latest)
else:
    st.info("Waiting for MQTT data...")

# Plot live Flow and Pressure
st.markdown("### 📊 Live Flow and Pressure")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st.session_state.timestamps,
    y=st.session_state.flow_values,
    mode='lines+markers',
    name='Flow (L/min)',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=st.session_state.timestamps,
    y=st.session_state.pressure_values,
    mode='lines+markers',
    name='Pressure (Bar)',
    line=dict(color='red')
))
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Measurement",
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# Automate response based on alert message
response_range = (10, 120) if st.session_state.collection_mode == "PdM" else (30, 300)
automate_response(latest.get("AlertMessage", "") if st.session_state.mqtt_data else "", st.session_state.collection_mode, response_range)

# Dashboard info
if st.session_state.mqtt_data:
    st.subheader(f"📅 Timestamp: {latest.get('timestamp')}")
    col1, col2 = st.columns(2)
    col1.metric("⏱️ Elapsed Time (s)", latest.get("Time", 0))
    col1.metric("💧 Flow (L/min)", latest.get("Flow", 0))
    col2.metric("🔧 Pressure (Bar)", latest.get("Pressure", 0))
    col2.metric("🧩 Fault Code (binary)", format(int(latest.get("FaultCode", 0)), "03b"))
    st.metric("⏳ Predicted Failure Time", latest.get("PredictedFailureTime", "N/A"))

    if latest.get("AlertMessage"):
        st.markdown("---")
        st.warning(latest.get("AlertMessage"))

    st.markdown("### 🛠️ Maintenance Recommendations")
    for rec in generate_recommendation(latest):
        st.write(rec)

    st.markdown("### 📡 Automated System Response")
    st.info(f"Pump Status: **{pump_state['status']}**")
    st.write(f"Action Taken: {pump_state['action']}")

# Show metrics
baseline_downtime, baseline_response, _ = calculate_metrics(st.session_state.baseline_downtime_events, st.session_state.baseline_response_times)
pdm_downtime, pdm_response, _ = calculate_metrics(st.session_state.pdm_downtime_events, st.session_state.pdm_response_times)

downtime_reduction = calculate_improvement(baseline_downtime, pdm_downtime)
response_improvement = calculate_improvement(baseline_response, pdm_response)
t_stat, p_value = perform_t_test(st.session_state.baseline_response_times, st.session_state.pdm_response_times)

st.markdown("---")
st.markdown("### 📈 Maintenance Metrics")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8 = st.columns(8)
kpi1.metric("Mode", st.session_state.collection_mode)
kpi2.metric("Baseline Downtime", f"{baseline_downtime:.2f} min")
kpi3.metric("Baseline Resp Time", f"{baseline_response:.2f} min")
kpi4.metric("PdM Downtime", f"{pdm_downtime:.2f} min")
kpi5.metric("PdM Resp Time", f"{pdm_response:.2f} min")
kpi6.metric("⏬ Downtime Reduction", f"{downtime_reduction:.2f}%")
kpi7.metric("⏱️ Response Improvement", f"{response_improvement:.2f}%")
if t_stat is not None and p_value is not None:
    kpi8.metric("P-value", f"{p_value:.4f}")

    if p_value < 0.05:
        st.success("Improvement in response time is statistically significant.")
    else:
        st.warning("Improvement in response time is NOT statistically significant.")

if st.session_state.collection_mode == "Baseline":
    st.markdown("---")
    st.markdown("### 💾 Export Baseline Data")

    if st.button("Export Downtime Events CSV"):
        df_downtime = downtime_events_to_df(st.session_state.baseline_downtime_events)
        csv = df_downtime.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Download Downtime CSV", data=csv, file_name="baseline_downtime_events.csv", mime="text/csv")

    if st.button("Export Response Times CSV"):
        df_response = response_times_to_df(st.session_state.baseline_response_times)
        csv = df_response.to_csv(index=False).encode('utf-8-sig')
        st.download_button("Download Response Times CSV", data=csv, file_name="baseline_response_times.csv", mime="text/csv")

# Refresh every 2 seconds to get new MQTT data
time.sleep(2)
st.rerun()
