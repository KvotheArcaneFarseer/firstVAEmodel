# app/main.py
# Description: The main entry point for the FastAPI application, built with professional
# practices like an app factory, lifespan management, and global error handling.

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# We import the main router and our central settings.
from .api.v1.router import api_router
from .config.settings import settings
# We import our services to initialize them during startup.
from .services.detection_service import get_detection_service
from .services.health_service import get_health_service

# --- Lifespan Management ---
# This function manages what happens when the application starts and stops.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    # --- Startup ---
    print("--- Application Startup ---")
    # By calling the service providers here, we trigger the lazy initialization
    # of our services (and thus the ML model) when the app starts,
    # rather than on the first request. This is a common production pattern.
    print("Initializing services...")
    get_detection_service()
    get_health_service()
    print("Services initialized.")
    
    yield # The application is now running.
    
    # --- Shutdown ---
    print("--- Application Shutdown ---")
    # In a more complex app, you would add cleanup code here,
    # like closing database connections.

# --- App Factory ---
# This function creates and configures our FastAPI application instance.
def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application instance.
    """
    app = FastAPI(
        title=settings.api.title,
        description=settings.api.description,
        version=settings.api.version,
        lifespan=lifespan  # We attach our lifespan manager here.
    )

    # --- Middleware ---
    # Middlewares are functions that process every request before it reaches the endpoint.
    # We will add a simple CORS middleware, which is essential for web frontends.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # --- Routers ---
    # We include our main API router.
    app.include_router(api_router)

    # --- Global Exception Handler ---
    # This is a safety net that catches any unhandled exceptions.
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Handles any exception that was not caught by a more specific handler.
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred.",
                "detail": str(exc)
            },
        )
    
    return app

# --- Global App Instance ---
# We create the final app instance by calling our factory function.
app = create_app()

# We can still have a root endpoint for a simple welcome message.
@app.get("/", tags=["Root"], include_in_schema=False)
async def read_root():
    return {"message": f"Welcome to the {settings.api.title}!"}
