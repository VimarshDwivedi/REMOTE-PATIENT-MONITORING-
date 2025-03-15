# data_processing/processor.py
import logging
import threading
import queue
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Base class for data processing"""
    
    def __init__(self, input_queue, output_queue=None, processing_interval: int = 1):
        """
        Initialize the data processor
        
        Args:
            input_queue: Queue to get data from
            output_queue: Queue to put processed data to
            processing_interval: Time interval in seconds between processing cycles
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.processing_interval = processing_interval
        self.is_processing = False
        self.processing_thread = None
        
    def start_processing(self):
        """Start the data processing in a separate thread"""
        if self.is_processing:
            logger.warning("Data processing is already running")
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Data processing started")
        
    def stop_processing(self):
        """Stop the data processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=self.processing_interval + 1)
        logger.info("Data processing stopped")
        
    def _processing_loop(self):
        """Main processing loop that runs at specified intervals"""
        while self.is_processing:
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
                    processed_data = self.process_data(data_batch)
                    if processed_data and self.output_queue:
                        self.output_queue.put(processed_data)
                        
            except Exception as e:
                logger.error(f"Error during data processing: {str(e)}")
                
            # Sleep for the specified interval
            threading.Event().wait(self.processing_interval)
    
    def process_data(self, data_batch: List[Dict[str, Any]]) -> Any:
        """
        Process the data batch
        
        Args:
            data_batch: List of data items to process
            
        Returns:
            Processed data
        """
        raise NotImplementedError("Subclasses must implement process_data method")


class VitalSignProcessor(DataProcessor):
    """Processor for patient vital signs data"""
    
    def __init__(self, input_queue, output_queue=None, processing_interval: int = 1):
        """
        Initialize the vital sign processor
        
        Args:
            input_queue: Queue to get data from
            output_queue: Queue to put processed data to
            processing_interval: Time interval in seconds between processing cycles
        """
        super().__init__(input_queue, output_queue, processing_interval)
        self.patient_data = {}  # Store recent data for each patient
        
    def process_data(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process vital sign data
        
        Args:
            data_batch: List of vital sign data items
            
        Returns:
            List of processed vital sign data
        """
        processed_batch = []
        
        for data_item in data_batch:
            # Extract patient ID
            patient_id = data_item.get("patient_id")
            if not patient_id:
                logger.warning(f"Missing patient ID in data item: {data_item}")
                continue
                
            # Apply data cleaning and normalization
            cleaned_data = self._clean_vital_data(data_item)
            
            # Add derived features
            processed_data = self._add_derived_features(cleaned_data)
            
            # Update patient history
            if patient_id not in self.patient_data:
                self.patient_data[patient_id] = []
            
            # Keep the last 20 readings
            self.patient_data[patient_id].append(processed_data)
            if len(self.patient_data[patient_id]) > 20:
                self.patient_data[patient_id] = self.patient_data[patient_id][-20:]
            
            # Add trend analysis
            processed_data = self._add_trend_analysis(processed_data, patient_id)
            
            processed_batch.append(processed_data)
            
        return processed_batch
    
    def _clean_vital_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize vital sign data
        
        Args:
            data: Raw vital sign data
            
        Returns:
            Cleaned data
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        vital_fields = [
            "heart_rate", "blood_pressure_systolic", "blood_pressure_diastolic",
            "respiratory_rate", "oxygen_saturation", "temperature"
        ]
        
        for field in vital_fields:
            if field not in cleaned_data or cleaned_data[field] is None:
                # Impute missing value with a default
                default_values = {
                    "heart_rate": 75,
                    "blood_pressure_systolic": 120,
                    "blood_pressure_diastolic": 80,
                    "respiratory_rate": 16,
                    "oxygen_saturation": 98,
                    "temperature": 36.8
                }
                cleaned_data[field] = default_values.get(field, 0)
                logger.warning(f"Missing {field} in data for patient {cleaned_data.get('patient_id')}, using default")
        
        # Remove outliers (simple range check)
        valid_ranges = {
            "heart_rate": (30, 200),
            "blood_pressure_systolic": (70, 200),
            "blood_pressure_diastolic": (40, 120),
            "respiratory_rate": (8, 40),
            "oxygen_saturation": (70, 100),
            "temperature": (35.0, 40.0)
        }
        
        for field, (min_val, max_val) in valid_ranges.items():
            if field in cleaned_data:
                value = cleaned_data[field]
                if value < min_val or value > max_val:
                    # Replace with closest valid value
                    cleaned_data[field] = min(max(value, min_val), max_val)
