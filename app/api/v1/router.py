# app/api/v1/router.py
# Description: This file aggregates all the individual endpoint routers from the v1 API,
# following the best practice of using a main versioned router.

from fastapi import APIRouter

# We import the router objects from our endpoint files using aliases
# to make the code clearer and prevent any potential name conflicts.
from .endpoints.detection import router as detection_router
from .endpoints.health import router as health_router

# This is the main router for the entire v1 of our API.
# By setting the prefix here, all routes included below will automatically
# start with /v1 (e.g., /v1/detect, /v1/health).
api_router = APIRouter(
    prefix="/v1"
)

# Here, we include the routers from our endpoint files into the main v1 router.
# We also assign tags, which will neatly group the endpoints in the API documentation.
api_router.include_router(detection_router, tags=["Detection"])
api_router.include_router(health_router, tags=["Health"])
