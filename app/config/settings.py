# app/config/settings.py
# Description: This file uses Pydantic to manage the application's settings,
# which can be loaded from environment variables or a .env file.

from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
import torch

class ModelConfig(BaseModel):
    # We define the paths to our trained model and scaler files here.
    model_path: str = Field(
        default="models/final_model.pth",
        description="Path to the trained VAE model weights file (.pth)."
    )
    scaler_path: str = Field(
        default="models/scaler.pkl",
        description="Path to the fitted data scaler file (.pkl)."
    )

    # --- Model Architecture ---
    # These parameters MUST match the architecture of the trained model.
    input_dim: int = Field(
        default=15,
        description="The number of input features for the model." 
    )

    latent_dim: int = Field(
    default=6,
    description="Latent dimension for the VAE."
    )

    hidden_dims: List[int] = Field(
        default_factory=lambda: [20, 12],
        description="A list of the hidden layer dimensions for the VAE."
    )

    device: str = Field(
        default="auto",
        description="Computation device: 'auto', 'cpu', or 'cuda'. "
    )

class DetectionConfig(BaseModel):
    # Configuration for the anomaly detection model.
    threshold: float = Field(
        default=0.1,
        description="Threshold for anomaly detection / recon error threshold to classify a sample as an anomaly.",
        gt=0.0
    )  # ensures threshold is always a positive value

class APIConfig(BaseModel):
    """Configuration for the FastAPI web server itself."""
    title: str = Field(
        default="VAE Anomaly Detection API",
        description="Title of the FastAPI application."
    )
    version: str = Field(
        default="1.0.0",
        description="Version of the FastAPI application."
    )
    description: str = Field(
        default="API for VAE-based anomaly detection.",
        description="Description of the FastAPI application."
    )

# --- Main Settings Class ---

class Settings(BaseModel):
    "Main settings class that aggregates all configurations."
    # BASE_DIR provides reliable reference to the project's root folder.

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Composing final setttings object from smaller classes.
    # makes the access to settingds more organized and modular.

    api: APIConfig = APIConfig()
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()

    # This is a helper funcrtion to resolve the full, absolute paths for the model files.
    def get_model_path(self) -> Path:
        return self.BASE_DIR / self.model.model_path
    
    def get_scaler_path(self) -> Path:
        return self.BASE_DIR / self.model.scaler_path
    
    # This helper resolves the computation device.
    def get_torch_device(self) -> torch.device:
        if self.model.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.model.device)
    
# ---Global Instance---
# Creating an instance of Settings that the rest of the application can use.
settings = Settings()

