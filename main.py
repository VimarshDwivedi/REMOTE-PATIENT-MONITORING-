import threading
import time
import random
import queue
import json
import requests
from flask import Flask, jsonify, request, redirect
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

# Queues for real-time data
patient_data_queue = queue.Queue()
alert_queue = queue.Queue()

# Flask app for API
app = Flask(__name__)

# Default Route to Redirect to Dashboard
@app.route("/")
def home():
    return redirect("/dashboard/")

# AI Model (Dummy Prediction Function)
def predict_disease(data):
    if data['heart_rate'] > 120 or data['oxygen_level'] < 90:
        return "High Risk"
    elif data['temperature'] > 101:
        return "Moderate Risk"
    return "Normal"

# Data Collection (Simulated Streaming Data)
def data_collection():
    patient_ids = ["P1", "P2", "P3"]
    while True:
        patient_id = random.choice(patient_ids)
        data = {
            "patient_id": patient_id,
            "heart_rate": random.randint(60, 150),
            "oxygen_level": random.randint(85, 100),
            "temperature": random.uniform(97, 104),
            "timestamp": time.time()
        }
        patient_data_queue.put(data)
        time.sleep(2)  # Simulating real-time data flow

# Data Processing & AI Analysis
def process_data():
    while True:
        if not patient_data_queue.empty():
            data = patient_data_queue.get()
            prediction = predict_disease(data)
            
            if prediction != "Normal":
                alert = {"patient_id": data['patient_id'], "severity": prediction, "data": data}
                alert_queue.put(alert)

            print(f"Processed: {data} -> {prediction}")
        time.sleep(1)

# Flask Routes for API
@app.route("/alerts", methods=["GET"])
def get_alerts():
    alerts = []
    while not alert_queue.empty():
        alerts.append(alert_queue.get())
    return jsonify(alerts)

# Dash UI Dashboard
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dashboard/', external_stylesheets=[dbc.themes.CYBORG])

# Dashboard Layout
dash_app.layout = dbc.Container([
    html.H1("Real-Time Patient Monitoring", className="text-center text-light mt-4"),
    dcc.Interval(id='update-interval', interval=5000, n_intervals=0),
    dbc.Row([
        dbc.Col(dcc.Graph(id='heart-rate-graph'), width=6),
        dbc.Col(dcc.Graph(id='oxygen-level-graph'), width=6)
    ]),
    html.Div(id='alert-output', style={'color': 'red', 'font-weight': 'bold', 'margin-top': '20px'})
], fluid=True)

# Dashboard Callbacks
@dash_app.callback(
    [Output('heart-rate-graph', 'figure'), Output('oxygen-level-graph', 'figure'), Output('alert-output', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_dashboard(n):
    try:
        response = requests.get("http://127.0.0.1:5000/alerts", timeout=2)
        response.raise_for_status()
        alerts_data = response.json()
        alerts = "<br>".join([f"ALERT: {a['patient_id']} - {a['severity']}" for a in alerts_data])
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        alerts = "Error fetching alerts"
    
    # Dummy Graph Data
    heart_rate_fig = go.Figure()
    heart_rate_fig.add_trace(go.Scatter(y=[random.randint(60, 120) for _ in range(10)], mode='lines', name='Heart Rate'))
    heart_rate_fig.update_layout(title='Heart Rate Over Time', template='plotly_dark')
    
    oxygen_level_fig = go.Figure()
    oxygen_level_fig.add_trace(go.Scatter(y=[random.randint(85, 100) for _ in range(10)], mode='lines', name='Oxygen Level'))
    oxygen_level_fig.update_layout(title='Oxygen Level Over Time', template='plotly_dark')
    
    return heart_rate_fig, oxygen_level_fig, alerts

# Start Background Threads
data_thread = threading.Thread(target=data_collection, daemon=True)
data_thread.start()
processing_thread = threading.Thread(target=process_data, daemon=True)
processing_thread.start()

# Run the Application
if __name__ == '__main__':
    app.run(debug=True)





