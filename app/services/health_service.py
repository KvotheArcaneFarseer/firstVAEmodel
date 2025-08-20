# app/services/health_service.py
# Description: This module provides the business logic for checking the health
# of the application and its various components.

from datetime import datetime
from typing import Dict, Any, Optional

# We import the DetectionService so we can check its status.
from .detection_service import DetectionService, get_detection_service

class HealthService:
    """
    Performs health checks on the application's critical services.
    """
    def __init__(self, detection_service: DetectionService):
        """
        Initializes the health service.

        Args:
            detection_service (DetectionService): The instance of the detection service to monitor.
        """
        self.detection_service = detection_service

    def check_health(self) -> Dict[str, Any]:
        """
        Runs a series of checks and returns a consolidated health report.

        Returns:
            Dict[str, Any]: A dictionary containing the overall status and details of each component check.
        """
        # We start by assuming the system is healthy.
        is_healthy = True
        
        # --- Check 1: Detection Service Status ---
        # We check the internal state of the detection service that we designed.
        detector_status = "ok"
        detector_message = "Detector is operational."

        if self.detection_service._initialization_error:
            # If an initialization error was saved, the service is unhealthy.
            is_healthy = False
            detector_status = "error"
            detector_message = f"Detector failed to initialize: {self.detection_service._initialization_error}"
        elif self.detection_service._detector is None:
            # If it's not initialized yet, the status is 'pending'. This is not an error.
            detector_status = "pending"
            detector_message = "Detector has not been initialized yet (lazy loading)."

        # --- Consolidate Results ---
        # We build a final report dictionary.
        health_report = {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "detection_service": {
                    "status": detector_status,
                    "message": detector_message
                }
                # In a more advanced app, we would add more checks here,
                # e.g., for database connections, external APIs, etc.
            }
        }
        
        return health_report
    
_health_service_instance: Optional[HealthService] = None

def get_health_service() -> HealthService:
    """
    Acts as a singleton provider for the HealthService.
    """
    global _health_service_instance
    if _health_service_instance is None:
        # Note: We call the other service's provider function here to get the
        # shared instance of the DetectionService.
        _health_service_instance = HealthService(detection_service=get_detection_service())
    return _health_service_instance  # type: ignore

