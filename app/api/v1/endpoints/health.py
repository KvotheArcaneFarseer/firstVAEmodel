# app/api/v1/endpoints/health.py
# Description: This file defines the API endpoint for checking the application's health.
# It adopts the best practice of mapping internal status to appropriate HTTP status codes.

from fastapi import APIRouter, Response, status
from typing import Dict, Any

# We import our clean dependency alias for the HealthService.
from ...dependencies import HealthServiceDep

router = APIRouter()

@router.get("/health")
def check_health(
    response: Response,
    health_service: HealthServiceDep
) -> Dict[str, Any]:
    """
    Checks the operational status of the API and its components.

    This endpoint returns a detailed health report and, critically, sets the
    HTTP status code based on the overall health of the system.

    - HTTP 200 OK: The service is healthy or pending initialization.
    - HTTP 503 Service Unavailable: The service is unhealthy (e.g., model failed to load).
    """
    try:
        # Step 1: Get the detailed health report from the health service.
        health_report = health_service.check_health()

        # Step 2: Check the overall status in the report to determine the HTTP status code.
        # This is a key feature of a professional API.
        if health_report.get("status") == "unhealthy":
            # If the service is unhealthy, we set the HTTP status code to 503.
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            # Otherwise (for "healthy" or "pending"), we keep the default 200 OK.
            response.status_code = status.HTTP_200_OK
        
        # Step 3: Return the complete JSON report.
        return health_report

    except Exception as e:
        # If the health check process itself fails unexpectedly,
        # we set the status to 503 and return an error message.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "error": "Failed to perform health check.",
            "detail": str(e)
        }
