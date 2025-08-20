# app/api/dependencies.py
# Description: This module provides the dependency injection system for the API.
# It acts as a central directory for accessing shared service instances.

from fastapi import Depends

# Step 1: Import the singleton provider functions from their respective service files.
# We are no longer creating the instances here; we are just importing the functions
# that are responsible for creating and providing them.
from ..services.detection_service import get_detection_service, DetectionService
from ..services.health_service import get_health_service, HealthService

# Step 2: Define the dependency functions.
# These are simple wrapper functions. Their only job is to call the real provider
# functions. This keeps the API layer cleanly separated from the service layer.

def get_detection_service_dependency() -> DetectionService:
    """Dependency for providing the DetectionService instance."""
    return get_detection_service()

def get_health_service_dependency() -> HealthService:
    """Dependency for providing the HealthService instance."""
    return get_health_service()

# Step 3 (Optional but recommended): Create dependency aliases.
# This is a professional quality-of-life improvement. It creates a clean,
# reusable shortcut that we can use in our endpoint function signatures.
# It makes the endpoint code much more readable.

from typing import Annotated

DetectionServiceDep = Annotated[DetectionService, Depends(get_detection_service_dependency)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service_dependency)]
