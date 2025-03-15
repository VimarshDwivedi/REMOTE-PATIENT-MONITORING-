import time
import json
import logging
import threading
import random
import queue
from datetime import datetime
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Base class for data collection from different sources"""

    def __init__(self, collection_interval: int = 5):
        """Initialize the data collector"""
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.collection_thread = None

    def start_collection(self):
        """Start the data collection process in a separate thread"""
        if self.is_collecting:
            logger.warning("Data collection is already running")
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Data collection started")

    def stop_collection(self):
        """Stop the data collection process"""
        if not self.is_collecting:
            logger.warning("Data collection is not running")
            return

        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=self.collection_interval + 1)

        logger.info("Data collection stopped")

    def _collection_loop(self):
        """Main collection loop that runs at specified intervals"""
        while self.is_collecting:
            try:
                data = self.collect_data()
                if data:
                    self.process_data(data)
            except Exception as e:
                logger.error(f"Error during data collection: {str(e)}")

            time.sleep(self.collection_interval)

    def collect_data(self) -> Dict[str, Any]:
        """Collect data from the source"""
        raise NotImplementedError("Subclasses must implement collect_data method")

    def process_data(self, data: Dict[str, Any]):
        """Process the collected data"""
        raise NotImplementedError("Subclasses must implement process_data method")


class PatientVitalCollector(DataCollector):
    """Collector for patient vital signs"""

    def __init__(self, patient_id: str, data_queue: queue.Queue, collection_interval: int = 5):
        """Initialize the patient vital collector"""
        super().__init__(collection_interval)
        self.patient_id = patient_id
        self.data_queue = data_queue

    def collect_data(self) -> Dict[str, Any]:
        """Collect vital signs for the patient"""
        vital_data = {
            "patient_id": self.patient_id,
            "timestamp": datetime.now().isoformat(),
            "heart_rate": random.randint(60, 100),
            "blood_pressure_systolic": random.randint(110, 140),
            "blood_pressure_diastolic": random.randint(70, 90),
            "respiratory_rate": random.randint(12, 20),
            "oxygen_saturation": random.randint(95, 100),
            "temperature": round(random.uniform(36.1, 37.5), 1),
        }

        logger.info(f"Collected vitals for patient {self.patient_id}: {vital_data}")
        return vital_data

    def process_data(self, data: Dict[str, Any]):
        """Process the collected vital sign data"""
        try:
            self.data_queue.put_nowait(data)
            logger.info(f"Added vital data to queue for patient {self.patient_id}")
        except queue.Full:
            logger.error("Data queue is full! Dropping data.")


class DeviceDataCollector(DataCollector):
    """Collector for medical device data"""

    def __init__(
        self,
        device_id: str,
        device_type: str,
        api_endpoint: str,
        data_queue: queue.Queue,
        api_key: Optional[str] = None,
        collection_interval: int = 5,
    ):
        """Initialize the device data collector"""
        super().__init__(collection_interval)
        self.device_id = device_id
        self.device_type = device_type
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.data_queue = data_queue

    def collect_data(self) -> Dict[str, Any]:
        """Collect data from the medical device"""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        try:
            # Simulating API call
            # response = requests.get(f"{self.api_endpoint}/{self.device_id}", headers=headers)
            # device_data = response.json()

            # Simulated data for demonstration
            device_data = {
                "device_id": self.device_id,
                "device_type": self.device_type,
                "timestamp": datetime.now().isoformat(),
                "battery_level": random.randint(30, 100),
                "status": "active",
                "readings": {
                    "value": round(random.uniform(0, 100), 2),
                    "unit": "mmol/L" if self.device_type == "glucose_monitor" else "unit",
                },
            }

            logger.info(f"Collected data from device {self.device_id}: {device_data}")
            return device_data

        except Exception as e:
            logger.error(f"Failed to collect data from device {self.device_id}: {str(e)}")
            return {}

    def process_data(self, data: Dict[str, Any]):
        """Process the collected device data"""
        try:
            self.data_queue.put_nowait(data)
            logger.info(f"Added device data to queue for device {self.device_id}")
        except queue.Full:
            logger.error("Data queue is full! Dropping data.")


if __name__ == "__main__":
    # Create a queue to store collected data
    data_queue = queue.Queue()

    # Create patient and device collectors
    patient_collector = PatientVitalCollector(patient_id="P123", data_queue=data_queue)
    device_collector = DeviceDataCollector(
        device_id="D456",
        device_type="glucose_monitor",
        api_endpoint="http://api.device.com",
        data_queue=data_queue
    )

    # Start data collection
    patient_collector.start_collection()
    device_collector.start_collection()

    try:
        for _ in range(5):  # Collect data for some time
            while not data_queue.empty():
                print("Received Data:", data_queue.get())
            time.sleep(5)

    except KeyboardInterrupt:
        print("Stopping data collection...")

    # Stop collectors
    patient_collector.stop_collection()
    device_collector.stop_collection()

