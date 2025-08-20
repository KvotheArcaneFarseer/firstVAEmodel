# app/dto/response_dto.py
# Description: This file defines the Pydantic models for all outgoing API responses,
# ensuring a consistent, structured, and professional data contract.

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

# --- Enums for Standardization ---
# Using Enums makes our code safer and more readable.
class ResponseStatus(str, Enum):
    """Standardized response statuses."""
    SUCCESS = "success"
    ERROR = "error"

# --- Core Data Models ---
# These models represent the actual data payload of our responses.

class DetectionResult(BaseModel):
    """
    Contains the detailed results of a single anomaly detection analysis.
    This is the core data payload for a successful detection.
    """
    is_anomaly: bool = Field(..., description="The final decision: true if an anomaly is detected, false otherwise.")
    reconstruction_error: float = Field(..., description="The calculated reconstruction error from the VAE model.")
    threshold: float = Field(..., description="The threshold used to make the is_anomaly decision.")
    confidence: float = Field(..., description="A score indicating the confidence of the detection result.")
    feature_errors: Dict[str, float] = Field(..., description="A dictionary of reconstruction errors for each individual feature.")
    processing_time_ms: float = Field(..., description="The time taken to process the request in milliseconds.")

class HealthComponentStatus(BaseModel):
    """Describes the health status of a single application component."""
    status: str = Field(..., description="The status of the component (e.g., 'ok', 'error', 'pending').")
    message: str = Field(..., description="A human-readable message about the component's status.")

# --- Main Response Wrappers ---
# These are the top-level models that our API endpoints will return.

class DetectionResponse(BaseModel):
    """
    The standard wrapper for a successful detection response.
    It includes a status and the detailed result payload.
    """
    status: ResponseStatus = ResponseStatus.SUCCESS
    result: DetectionResult

class HealthCheckResponse(BaseModel):
    """The standard wrapper for a health check response."""
    status: str = Field(..., description="The overall health status of the application ('healthy' or 'unhealthy').")
    timestamp: datetime = Field(..., description="The UTC timestamp of when the health check was performed.")
    components: Dict[str, HealthComponentStatus] = Field(..., description="A dictionary detailing the status of each checked component.")

class ErrorResponse(BaseModel):
    """
    The standard wrapper for any error response. This ensures that clients
    always receive a predictable error format.
    """
    status: ResponseStatus = ResponseStatus.ERROR
    error_message: str = Field(..., description="A clear, human-readable error message.")
    error_details: Optional[str] = Field(None, description="Optional technical details about the error.")
