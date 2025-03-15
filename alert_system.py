import logging
import threading
import queue
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertManager:
    def __init__(self, input_queue: queue.Queue, alert_interval: int = 1):
        self.input_queue = input_queue
        self.alert_interval = alert_interval
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self.handlers: Dict[str, Callable] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: Dict[str, List[Dict[str, Any]]] = {}
        self.alert_cooldowns: Dict[str, int] = {}
        self.last_alerts: Dict[str, datetime] = {}
    
    def register_handler(self, alert_type: str, handler: Callable):
        self.handlers[alert_type] = handler
        logger.info(f"Registered handler for alert type: {alert_type}")
    
    def start_processing(self):
        if self.is_processing:
            logger.warning("Alert processing is already running")
            return
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Alert processing started")
    
    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=self.alert_interval + 1)
        logger.info("Alert processing stopped")
    
    def _processing_loop(self):
        while self.is_processing:
            try:
                while not self.input_queue.empty():
                    data = self.input_queue.get()
                    self._process_alert(data)
                    self.input_queue.task_done()
            except queue.Empty:
                pass
            time.sleep(self.alert_interval)
    
    def _process_alert(self, data: Dict[str, Any]):
        patient_id = data.get("patient_id")
        if not patient_id:
            logger.warning(f"Missing patient ID in analysis result: {data}")
            return
        analysis_type = data.get("analysis_type")
        if analysis_type == "anomaly_detection":
            self._process_anomaly_alert(patient_id, data)
        elif analysis_type == "disease_prediction":
            self._process_disease_alert(patient_id, data)
        else:
            logger.warning(f"Unknown analysis type: {analysis_type}")
    
    def _process_anomaly_alert(self, patient_id: str, data: Dict[str, Any]):
        if not data.get("is_anomaly", False) or data.get("severity") == "normal":
            return
        self._trigger_alert(self._create_alert(patient_id, "vital_signs_anomaly", AlertLevel.CRITICAL, "Anomaly detected in vital signs"))
    
    def _process_disease_alert(self, patient_id: str, data: Dict[str, Any]):
        predictions = data.get("predictions", {})
        for disease, prediction in predictions.items():
            probability = prediction.get("probability", 0)
            if probability >= 0.7:
                alert_level = AlertLevel.CRITICAL if probability > 0.9 else AlertLevel.WARNING if probability > 0.8 else AlertLevel.INFO
                self._trigger_alert(self._create_alert(patient_id, f"disease_prediction_{disease}", alert_level, f"High probability ({probability:.1%}) of {disease}"))
    
    def _create_alert(self, patient_id: str, alert_type: str, alert_level: AlertLevel, message: str) -> Dict[str, Any]:
        alert = {
            "alert_id": f"{patient_id}_{alert_type}",
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "alert_level": alert_level.value,
            "message": message,
            "status": "active"
        }
        self.active_alerts[alert["alert_id"]] = alert
        self.alert_history.setdefault(patient_id, []).append(alert)
        self.last_alerts[alert["alert_id"]] = datetime.now()
        return alert
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        if alert["alert_type"] in self.handlers:
            try:
                self.handlers[alert["alert_type"]](alert)
            except Exception as e:
                logger.error(f"Error in alert handler for {alert['alert_type']}: {str(e)}")
    
    def resolve_alert(self, alert_id: str):
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["status"] = "resolved"
            self.active_alerts[alert_id]["resolved_at"] = datetime.now().isoformat()
            logger.info(f"Resolved alert {alert_id}")
            del self.active_alerts[alert_id]
            return True
        logger.warning(f"Attempted to resolve non-existent alert: {alert_id}")
        return False
