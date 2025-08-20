# app/api/v1/endpoints/detection.py
# Description: This file defines the API endpoint for performing anomaly detection.
# It includes robust error handling to map service-layer errors to appropriate HTTP responses.

from fastapi import APIRouter, HTTPException, status

# We import the DTOs to define the request and response shapes.
from ....dto.request_dto import DetectionRequest
from ....dto.response_dto import DetectionResponse

# We import our clean dependency alias.
from ...dependencies import DetectionServiceDep
# We also need to import our custom service error to handle it specifically.
from ....services.detection_service import DetectionServiceError

# An APIRouter helps organize endpoints.
router = APIRouter()

@router.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(
    request: DetectionRequest,
    detection_service: DetectionServiceDep
):
    """
    Receives a single sample of 15 feature values and returns an anomaly detection result.
    
    This endpoint uses the dependency injection system to get access to the
    DetectionService and includes robust error handling.
    """
    try:
        # The code here remains clean. We just pass the features from the request
        # directly to our service and return the result.
        return detection_service.detect_anomaly(features=request.player_features.to_list())
    
    except DetectionServiceError as e:
        # This is a critical improvement. If our service layer raises a specific
        # error (like a model initialization failure), we catch it here.
        # We then raise an HTTPException with a 400 Bad Request status,
        # providing a clear error message to the user.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=str(e)
        )
        
    except Exception as e:
        # This is a catch-all for any other unexpected errors.
        # We raise a generic 500 Internal Server Error to avoid leaking
        # sensitive implementation details in the error message.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An unexpected internal error occurred."
        )
