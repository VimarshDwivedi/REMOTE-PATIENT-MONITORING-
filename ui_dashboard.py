# ui/dashboard.py
import os
import logging
import time
import threading
import queue
from datetime import datetime, timedelta
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatientMonitoringDashboard:
    """Dashboard for the patient monitoring system"""
    
    def __init__(self, data_queue=None, refresh_interval: int = 5):
        """
        Initialize the dashboard
        
        Args:
            data_queue: Queue to get data from
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.data_queue = data_queue
        self.refresh_interval = refresh_interval
        
        # Data storage
        self.patients = {}  # Patient data
        self.alerts = {}    # Alerts by patient
        self.predictions = {}  # Disease predictions by patient
        
        # Create Flask and Dash apps
        self.server = Flask(__name__)
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[dbc.themes.FLATLY],
            title="Patient Monitoring System",
            update_title=None,
            suppress_callback_exceptions=True
        )
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Data processing thread
        self.is_processing = False
        self.processing_thread = None
        
    def start_processing(self):
        """Start the data processing thread"""
        if not self.data_queue:
            logger.warning("No data queue provided, dashboard will use sample data")
            self._load_sample_data()
            return
            
        if self.is_processing:
            logger.warning("Data processing is already running")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Dashboard data processing started")
        
    def stop_processing(self):
        """Stop the data processing thread"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=self.refresh_interval + 1)
        logger.info("Dashboard data processing stopped")
        
    def run_server(self, debug=False, port=8050):
        """
        Run the dashboard server
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        # Start data processing if not already started
        if not self.is_processing and self.data_queue:
            self.start_processing()
            
        # Run the Dash app
        self.app.run_server(debug=debug, port=port)
        
    def _processing_loop(self):
        """Main processing loop that runs at specified intervals"""
        while self.is_processing:
            try:
                # Get all available data from the queue
                data_batch = []
                try:
                    while True:
                        data = self.data_queue.get_nowait()
                        data_batch.append(data)
                        self.data_queue.task_done()
                except queue.Empty:
                    pass
                
                # Process the batch if not empty
                if data_batch:
                    self._process_data(data_batch)
                        
            except Exception as e:
                logger.error(f"Error during dashboard data processing: {str(e)}")
                
            # Sleep for the specified interval
            threading.Event().wait(self.refresh_interval)
    
    def _process_data(self, data_batch):
        """
        Process incoming data batch
        
        Args:
            data_batch: List of data items to process
        """
        for item in data_batch:
            if 'analysis_type' in item:
                self._process_analysis_result(item)
            elif 'heart_rate' in item or 'vital_signs' in item:
                self._process_vital_signs(item)
            elif 'alert_level' in item:
                self._process_alert(item)
    
    def _process_analysis_result(self, item):
        """Process analysis result"""
        patient_id = item.get('patient_id')
        if not patient_id:
            return
            
        analysis_type = item.get('analysis_type')
        
        if analysis_type == 'disease_prediction':
            if patient_id not in self.predictions:
                self.predictions[patient_id] = []
                
            self.predictions[patient_id].append(item)
            # Keep only recent predictions
            if len(self.predictions[patient_id]) > 50:
                self.predictions[patient_id] = self.predictions[patient_id][-50:]
    
    def _process_vital_signs(self, item):
        """Process vital signs data"""
        patient_id = item.get('patient_id')
        if not patient_id:
            return
            
        if patient_id not in self.patients:
            self.patients[patient_id] = []
            
        # If data contains vital_signs key, extract the nested values
        if 'vital_signs' in item:
            vital_data = {
                'patient_id': patient_id,
                'timestamp': item.get('timestamp', datetime.now().isoformat())
            }
            vital_data.update(item['vital_signs'])
            self.patients[patient_id].append(vital_data)
        else:
            self.patients[patient_id].append(item)
            
        # Keep only recent data
        if len(self.patients[patient_id]) > 500:
            self.patients[patient_id] = self.patients[patient_id][-500:]
    
    def _process_alert(self, item):
        """Process alert data"""
        patient_id = item.get('patient_id')
        if not patient_id:
            return
            
        if patient_id not in self.alerts:
            self.alerts[patient_id] = []
            
        self.alerts[patient_id].append(item)
        # Keep only recent alerts
        if len(self.alerts[patient_id]) > 100:
            self.alerts[patient_id] = self.alerts[patient_id][-100:]
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        # Navigation bar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.I(className="fas fa-heartbeat me-2", style={"color": "white", "font-size": "24px"})),
                                dbc.Col(dbc.NavbarBrand("Patient Monitoring System", className="ms-2")),
                            ],
                            align="center",
                        ),
                        href="/",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button("Refresh Data", id="refresh-button", color="light", className="me-2"),
                                    dcc.Interval(id='interval-component', interval=self.refresh_interval * 1000, n_intervals=0),
                                ]
                            ),
                        ],
                        className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
                        align="center",
                    ),
                ]
            ),
            color="primary",
            dark=True,
            className="mb-4"
        )
        
        # Main layout
        self.app.layout = html.Div(
            [
                navbar,
                dbc.Container(
                    [
                        # Patient selection and alerts row
                        dbc.Row(
                            [
                                # Patient selection
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Patient Selection"),
                                                dbc.CardBody(
                                                    [
                                                        html.P("Select a patient to view their data:"),
                                                        dcc.Dropdown(
                                                            id="patient-dropdown",
                                                            options=[],
                                                            value=None,
                                                            placeholder="Select a patient",
                                                            className="mb-3"
                                                        ),
                                                        html.Div(id="patient-info")
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        )
                                    ],
                                    md=4
                                ),
                                
                                # Active alerts
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Active Alerts"),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(id="active-alerts", className="alerts-container")
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        )
                                    ],
                                    md=8
                                )
                            ],
                            className="mb-4"
                        ),
                        
                        # Vital signs and predictions
                        dbc.Row(
                            [
                                # Vital signs
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Vital Signs"),
                                                dbc.CardBody(
                                                    [
                                                        dcc.Graph(id="vital-signs-graph", style={"height": "400px"})
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        )
                                    ],
                                    md=8
                                ),
                                
                                # Predictions and stats
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Current Status"),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(id="current-vitals")
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        ),
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Disease Risk Assessment"),
                                                dbc.CardBody(
                                                    [
                                                        html.Div(id="disease-predictions")
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        )
                                    ],
                                    md=4
                                )
                            ]
                        ),
                        
                        # Historical data
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardHeader("Historical Data"),
                                                dbc.CardBody(
                                                    [
                                                        dbc.Tabs(
                                                            [
                                                                dbc.Tab(
                                                                    [
                                                                        html.Div(id="historical-vitals-content", className="pt-3")
                                                                    ],
                                                                    label="Vital History",
                                                                    tab_id="tab-vitals"
                                                                ),
                                                                dbc.Tab(
                                                                    [
                                                                        html.Div(id="historical-alerts-content", className="pt-3")
                                                                    ],
                                                                    label="Alert History",
                                                                    tab_id="tab-alerts"
                                                                ),
                                                                dbc.Tab(
                                                                    [
                                                                        html.Div(id="disease-trends-content", className="pt-3")
                                                                    ],
                                                                    label="Disease Trends",
                                                                    tab_id="tab-diseases"
                                                                )
                                                            ],
                                                            id="tabs",
                                                            active_tab="tab-vitals"
                                                        )
                                                    ]
                                                )
                                            ],
                                            className="mb-4"
                                        )
                                    ],
                                    md=12
                                )
                            ]
                        )
                    ],
                    fluid=True
                )
            ]
        )
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks"""
        # Update patient dropdown options
        @self.app.callback(
            Output("patient-dropdown", "options"),
            [Input("interval-component", "n_intervals"),
             Input("refresh-button", "n_clicks")]
        )
        def update_patient_options(n_intervals, n_clicks):
            """Update patient dropdown options"""
            patients = list(self.patients.keys())
            return [{"label": f"Patient {p}", "value": p} for p in patients]
        
        # Update active alerts
        @self.app.callback(
            Output("active-alerts", "children"),
            [Input("interval-component", "n_intervals"),
             Input("patient-dropdown", "value"),
             Input("refresh-button", "n_clicks")]
        )
        def update_active_alerts(n_intervals, selected_patient, n_clicks):
            """Update active alerts"""
            alerts_components = []
            
            # Get alerts for the selected patient or all alerts if no patient selected
            all_alerts = []
            
            if selected_patient and selected_patient in self.alerts:
                all_alerts = self.alerts[selected_patient][-10:]  # Get the 10 most recent alerts
            else:
                # Get the 10 most recent alerts across all patients
                for patient_id, patient_alerts in self.alerts.items():
                    all_alerts.extend(patient_alerts[-10:])
                    
                # Sort by timestamp and take the 10 most recent
                all_alerts = sorted(all_alerts, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
            
            if not all_alerts:
                return html.P("No active alerts.", className="text-muted")
                
            for alert in all_alerts:
                # Determine alert color based on level
                alert_level = alert.get('alert_level', 'info')
                if alert_level == 'emergency':
                    color = "danger"
                elif alert_level == 'critical':
                    color = "danger"
                elif alert_level == 'warning':
                    color = "warning"
                else:
                    color = "info"
                
                # Format timestamp
                timestamp = alert.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%H:%M:%S %d/%m/%Y")
                except:
                    formatted_time = timestamp
                    
                # Create alert card
                alert_card = dbc.Alert(
                    [
                        html.H5(f"Patient {alert.get('patient_id')} - {alert_level.upper()}", className="alert-heading"),
                        html.P(alert.get('message', 'No message')),
                        html.Hr(),
                        html.P(f"Time: {formatted_time}", className="mb-0 small")
                    ],
                    color=color,
                    className="mb-3"
                )
                
                alerts_components.append(alert_card)
                
            return alerts_components
        