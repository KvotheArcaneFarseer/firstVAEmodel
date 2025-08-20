import torch
import numpy as np
import time
from typing import Union, List, Dict, Any
from dataclasses import dataclass
from numpy.typing import NDArray

# We import the classes we just defined in the same 'core' folder.
from .vae_model import VAE
from .preprocessor import DataPreprocessor

# --- Structured Result Class ---
# Using a dataclass is a clean way to define a structured object for our results.
@dataclass
class AnomalyResult:
    """A structured data class to hold the results of an anomaly detection."""
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: float
    feature_errors: Dict[str, float]
    processing_time_ms: float
    # --- Main Anomaly Detector Class ---
class AnomalyDetector:
    """
    Uses a traineed VAE model to compute detailed anomaly detection results.
    """

    def __init__(self, model: VAE, preprocessor: DataPreprocessor, device: torch.device):
        """
        Initializes the Anomaly Detector.
        
        Args:
            model (VAE): The trained VAE model.
            preprocessor (DataPreprocessor): The preprocessor instance.
            device (torch.device): The computation device ('cpu' or 'cuda').

        """

        self.model = model
        self.preprocessor = preprocessor
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def detect(
            self,
            raw_data: List[float],
            threshold: float
    ) -> AnomalyResult:
        """
        Performs full anomaly detection analysis on a single data sample.

        Args:
            raw_data: A single sample as a lsit of 15 floats.
            threshold: The anomaly threshold to use for classification.

        Returns:
            An AnomalyReuslt object containing the detailed detection results.
        """

        start_time = time.perf_counter()

        # Step 1: Preprocess the raw data.
        preprocessed_data = np.asarray(self.preprocessor.preprocess(raw_data), dtype=np.float32).reshape(-1)

        # Step 2: Convert to a PyTorch tensor with batch dimension.
        data_tensor = torch.FloatTensor(preprocessed_data).unsqueeze(0).to(self.device)

        # Step 3: Get the model's reconstruction.
        with torch.no_grad():
            reconstructed_tensor, _, _ = self.model(data_tensor)

        # Step 4: Calculate the overall reconstruction error(MSE).
        overall_error = torch.mean((data_tensor - reconstructed_tensor)**2).item()

        # Step 5: Determine if it's an anomaly.
        is_anomaly = overall_error > threshold

        # Step 6: Calculate per-feature errors for explainability.
        # We get the squared error for each of the 15 features.
        per_feature_errors_tensor = (data_tensor - reconstructed_tensor)**2
        per_feature_errors = per_feature_errors_tensor.squeeze(0).cpu().tolist()

        feature_error_dict = dict(zip(self.preprocessor.config.EXPECTED_FEATURES, per_feature_errors))

        # Step 7: Calculate a simple confidence score.
        # This score is higher the further the error is from the threshold.
        confidence = abs(overall_error - threshold) / (overall_error + threshold + 1e-9)

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        # Step 8: Return the structured result.
        return AnomalyResult(
            is_anomaly=is_anomaly,
            reconstruction_error=overall_error,
            threshold=threshold,
            confidence=confidence,
            feature_errors=feature_error_dict,
            processing_time_ms=processing_time_ms
        )