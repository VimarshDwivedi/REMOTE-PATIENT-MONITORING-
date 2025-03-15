# ai_analysis/analyzer.py
import logging
import numpy as np
import pandas as pd
import pickle
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Base class for AI-based data analysis"""
    
    def __init__(self, input_queue, output_queue=None, analysis_interval: int = 5):
        """
        Initialize the AI analyzer
        
        Args:
            input_queue: Queue to get processed data from
            output_queue: Queue to put analysis results to
            analysis_interval: Time interval in seconds between analysis cycles
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.analysis_interval = analysis_interval
        self.is_analyzing = False
        self.analysis_thread = None
        
    def start_analysis(self):
        """Start the AI analysis in a separate thread"""
        if self.is_analyzing:
            logger.warning("AI analysis is already running")
            return
            
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        logger.info("AI analysis started")
        
    def stop_analysis(self):
        """Stop the AI analysis"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=self.analysis_interval + 1)
        logger.info("AI analysis stopped")
        
    def _analysis_loop(self):
        """Main analysis loop that runs at specified intervals"""
        while self.is_analyzing:
            try:
                # Get all available data from the queue
                data_batch = []
                try:
                    while True:
                        data = self.input_queue.get_nowait()
                        data_batch.append(data)
                        self.input_queue.task_done()
                except queue.Empty:
                    pass
                
                # Process the batch if not empty
                if data_batch:
                    analysis_results = self.analyze_data(data_batch)
                    if analysis_results and self.output_queue:
                        for result in analysis_results:
                            self.output_queue.put(result)
                        
            except Exception as e:
                logger.error(f"Error during AI analysis: {str(e)}")
                
            # Sleep for the specified interval
            threading.Event().wait(self.analysis_interval)
    
    def analyze_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze the data batch
        
        Args:
            data_batch: List of data items to analyze
            
        Returns:
            List of analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze_data method")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a machine learning model from the specified path
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement load_model method")


class AnomalyDetector(AIAnalyzer):
    """Anomaly detection for patient vital signs"""
    
    def __init__(self, input_queue, output_queue=None, analysis_interval: int = 5, 
                 model_path: Optional[str] = None):
        """
        Initialize the anomaly detector
        
        Args:
            input_queue: Queue to get processed data from
            output_queue: Queue to put analysis results to
            analysis_interval: Time interval in seconds between analysis cycles
            model_path: Path to the pre-trained anomaly detection model
        """
        super().__init__(input_queue, output_queue, analysis_interval)
        self.model = None
        self.scaler = None
        
        # In a real system, you would load a pre-trained model
        # For demonstration, we'll create a simple model
        if model_path:
            try:
                self.model, self.scaler = self.load_model(model_path)
                logger.info(f"Loaded anomaly detection model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                self._create_default_model()
        else:
            self._create_default_model()
            
        # Keep track of patient history for contextual anomaly detection
        self.patient_history = {}
        
    def _create_default_model(self):
        """Create a default anomaly detection model"""
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        logger.info("Created default anomaly detection model")
        
    def load_model(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load anomaly detection model and scaler
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (model, scaler)
        """
        model = joblib.load(f"{model_path}/anomaly_model.pkl")
        scaler = joblib.load(f"{model_path}/anomaly_scaler.pkl")
        return model, scaler
        
    def analyze_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in patient vital signs
        
        Args:
            data_batch: List of processed vital sign data
            
        Returns:
            List of anomaly detection results
        """
        results = []
        
        # Group data by patient
        patient_data = {}
        for item in data_batch:
            patient_id = item.get("patient_id")
            if patient_id:
                if patient_id not in patient_data:
                    patient_data[patient_id] = []
                patient_data[patient_id].append(item)
        
        # Process each patient's data
        for patient_id, patient_items in patient_data.items():
            # Update patient history
            if patient_id not in self.patient_history:
                self.patient_history[patient_id] = []
            
            # Add new data to history
            self.patient_history[patient_id].extend(patient_items)
            
            # Keep last 100 readings
            if len(self.patient_history[patient_id]) > 100:
                self.patient_history[patient_id] = self.patient_history[patient_id][-100:]
            
            # Perform anomaly detection
            latest_item = patient_items[-1]  # Get most recent reading
            
            # Extract features for anomaly detection
            features = self._extract_features(latest_item)
            
            # Transform features
            if len(features) > 0:
                features_array = np.array([features])
                
                # Scale features
                scaled_features = self.scaler.fit_transform(features_array)
                
                # Detect anomaly
                anomaly_score = self.model.decision_function(scaled_features)[0]
                is_anomaly = self.model.predict(scaled_features)[0] == -1
                
                # Create result
                result = {
                    "patient_id": patient_id,
                    "timestamp": latest_item.get("timestamp"),
                    "analysis_type": "anomaly_detection",
                    "is_anomaly": is_anomaly,
                    "anomaly_score": float(anomaly_score),
                    "vital_signs": {
                        key: latest_item.get(key) for key in [
                            "heart_rate", "blood_pressure_systolic", "blood_pressure_diastolic",
                            "respiratory_rate", "oxygen_saturation", "temperature"
                        ] if key in latest_item
                    },
                    "contextual_info": self._get_contextual_info(patient_id, latest_item)
                }
                
                # Add severity level based on score
                if is_anomaly:
                    if anomaly_score < -0.5:
                        result["severity"] = "high"
                    else:
                        result["severity"] = "medium"
                else:
                    result["severity"] = "normal"
                
                results.append(result)
        
        return results
    
    def _extract_features(self, data_item: Dict[str, Any]) -> List[float]:
        """
        Extract features for anomaly detection
        
        Args:
            data_item: Data item containing vital signs
            
        Returns:
            List of features
        """
        features = []
        
        # Extract vital sign features
        feature_keys = [
            "heart_rate", "blood_pressure_systolic", "blood_pressure_diastolic",
            "respiratory_rate", "oxygen_saturation", "temperature"
        ]
        
        for key in feature_keys:
            if key in data_item and data_item[key] is not None:
                features.append(float(data_item[key]))
            else:
                features.append(0.0)  # Default value for missing features
        
        # Could add more derived features here
                
        return features
    
    def _get_contextual_info(self, patient_id: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get contextual information for anomaly detection
        
        Args:
            patient_id: ID of the patient
            current_data: Current data item
            
        Returns:
            Dictionary with contextual information
        """
        history = self.patient_history.get(patient_id, [])
        context = {}
        
        if len(history) > 1:
            # Calculate average values from history
            hr_values = [item.get("heart_rate", 0) for item in history if "heart_rate" in item]
            bp_sys_values = [item.get("blood_pressure_systolic", 0) for item in history if "blood_pressure_systolic" in item]
            bp_dia_values = [item.get("blood_pressure_diastolic", 0) for item in history if "blood_pressure_diastolic" in item]
            
            if hr_values:
                context["avg_heart_rate"] = sum(hr_values) / len(hr_values)
                context["hr_change"] = current_data.get("heart_rate", 0) - context["avg_heart_rate"]
                
            if bp_sys_values:
                context["avg_bp_systolic"] = sum(bp_sys_values) / len(bp_sys_values)
                context["systolic_change"] = current_data.get("blood_pressure_systolic", 0) - context["avg_bp_systolic"]
                
            if bp_dia_values:
                context["avg_bp_diastolic"] = sum(bp_dia_values) / len(bp_dia_values)
                context["diastolic_change"] = current_data.get("blood_pressure_diastolic", 0) - context["avg_bp_diastolic"]
        
        return context


class DiseasePredictor(AIAnalyzer):
    """Disease prediction using patient vital signs and history"""
    
    def __init__(self, input_queue, output_queue=None, analysis_interval: int = 10, 
                 model_path: Optional[str] = None):
        """
        Initialize the disease predictor
        
        Args:
            input_queue: Queue to get processed data from
            output_queue: Queue to put analysis results to
            analysis_interval: Time interval in seconds between analysis cycles
            model_path: Path to the pre-trained disease prediction model
        """
        super().__init__(input_queue, output_queue, analysis_interval)
        self.models = {}
        self.scalers = {}
        
        # Disease categories we can predict
        self.disease_categories = [
            "cardiac_issues", "respiratory_problems", "infection", "hypertension"
        ]
        
        # In a real system, you would load pre-trained models
        # For demonstration, we'll create simple models
        if model_path:
            try:
                self._load_models(model_path)
                logger.info(f"Loaded disease prediction models from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load models from {model_path}: {str(e)}")
                self._create_default_models()
        else:
            self._create_default_models()
            
        # Patient history for trend analysis
        self.patient_history = {}
        
    def _create_default_models(self):
        """Create default disease prediction models"""
        for disease in self.disease_categories:
            self.models[disease] = RandomForestClassifier(n_estimators=50, random_state=42)
            self.scalers[disease] = StandardScaler()
        
        logger.info("Created default disease prediction models")
        
    def _load_models(self, model_path: str):
        """
        Load disease prediction models
        
        Args:
            model_path: Path to the models directory
        """
        for disease in self.disease_categories:
            try:
                model_file = f"{model_path}/{disease}_model.pkl"
                scaler_file = f"{model_path}/{disease}_scaler.pkl"
                
                self.models[disease] = joblib.load(model_file)
                self.scalers[disease] = joblib.load(scaler_file)
                
                logger.info(f"Loaded model for {disease}")
            except Exception as e:
                logger.error(f"Failed to load model for {disease}: {str(e)}")
                # Fall back to default model
                self.models[disease] = RandomForestClassifier(n_estimators=50, random_state=42)
                self.scalers[disease] = StandardScaler()
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load all disease prediction models
        
        Args:
            model_path: Path to the models directory
            
        Returns:
            Dictionary of models
        """
        self._load_models(model_path)
        return self.models
        
    def analyze_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict diseases based on patient data
        
        Args:
            data_batch: List of processed patient data
            
        Returns:
            List of disease prediction results
        """
        results = []
        
        # Group data by patient
        patient_data = {}
        for item in data_batch:
            patient_id = item.get("patient_id")
            if patient_id:
                if patient_id not in patient_data:
                    patient_data[patient_id] = []
                patient_data[patient_id].append(item)
        
        # Process each patient's data
        for patient_id, patient_items in patient_data.items():
            # Update patient history
            if patient_id not in self.patient_history:
                self.patient_history[patient_id] = []
            
            # Add new data to history
            self.patient_history[patient_id].extend(patient_items)
            
            # Keep last 24 hours of readings (assuming 5-minute intervals = 288 readings)
            if len(self.patient_history[patient_id]) > 288:
                self.patient_history[patient_id] = self.patient_history[patient_id][-288:]
            
            # Get recent history for analysis
            recent_history = self.patient_history[patient_id][-10:]