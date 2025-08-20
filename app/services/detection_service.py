

from typing import Optional, List
import torch

# We import the components this service will use.
from ..core.anomaly_detector import AnomalyDetector, AnomalyResult
from ..core.preprocessor import DataPreprocessor
from ..config.settings import settings
from ..dto.response_dto import DetectionResponse, DetectionResult, ResponseStatus
from ..core.vae_model import VAE

# A custom error type for this service, which is a good practice for error handling.
class DetectionServiceError(Exception):
    """Custom exception for errors specific to the DetectionService."""
    pass

class DetectionService:
    """
    Orchestrates the anomaly detection process, acting as a bridge between
    the API layer and the core ML logic.
    """
    def __init__(self):
        """
        Initializes the detection service.
        Note: We do not load the model here. This is lazy initialization.
        """
        # These are placeholders for our core components. They will be filled
        # only when the first request comes in.
        self._detector: Optional[AnomalyDetector] = None
        self._initialization_error: Optional[str] = None

    def _initialize_detector(self):
        """
        A private helper method that loads all core ML components.
        This is designed to be called only once, on the first request.
        """
        try:
            # Get all necessary paths and settings from our central config.
            model_path = settings.get_model_path()
            scaler_path = settings.get_scaler_path()
            device = settings.get_torch_device()

            # Step 1: Create the preprocessor instance.
            preprocessor = DataPreprocessor(scaler_path=scaler_path)

            # Step 2: Create the VAE model instance and load its trained weights.
            model = VAE(
                input_dim=settings.model.input_dim,
                hidden_dims=settings.model.hidden_dims,
                latent_dim=settings.model.latent_dim
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Step 3: Create the core AnomalyDetector with all its tools.
            self._detector = AnomalyDetector(
                model=model,
                preprocessor=preprocessor,
                device=device
            )
        except Exception as e:
            # If anything goes wrong during this setup, we catch the error,
            # store the message, and re-raise it as our custom service error.
            self._initialization_error = str(e)
            self._detector = None # Ensure detector is None if setup fails.
            raise DetectionServiceError(f"Failed to initialize detector: {e}")

    def detect_anomaly(self, features: list[float]) -> DetectionResponse:
        """
        Handles the business logic for a single anomaly detection request.

        Args:
            features (list[float]): The list of 15 raw feature values from the API request.

        Returns:
            A DetectionResponse DTO ready to be sent back as a JSON response.
        """
        try:
            # Step 1: Ensure the detector has been initialized (lazy loading).
            # This 'if' block will only run on the very first request to this function.
            if self._detector is None:
                # Check if a previous initialization attempt already failed.
                if self._initialization_error:
                    raise DetectionServiceError(f"Detector is unavailable due to a previous initialization error: {self._initialization_error}")
                # If not, attempt to initialize it now.
                self._initialize_detector()
            
            # Step 2: Get the current anomaly threshold from our main settings.
            threshold = settings.detection.threshold

            # Step 3: Use the now-ready anomaly detector to perform the full analysis.
            result: AnomalyResult = self._detector.detect(
                raw_data=features,
                threshold=threshold
            )

            # Step 4: Format the detailed AnomalyResult from the core logic into
            # the DetectionResponse DTO that the API user expects.
            detection_result = DetectionResult(
                is_anomaly=result.is_anomaly,
                reconstruction_error=result.reconstruction_error,
                threshold=result.threshold,
                confidence=result.confidence,
                feature_errors=result.feature_errors,
                processing_time_ms=result.processing_time_ms
            )
            
            return DetectionResponse(
                status=ResponseStatus.SUCCESS,
                result=detection_result
            )
        except Exception as e:
            # Catch any other error during the detection process and wrap it
            # in our custom service error for consistent error handling.
            raise DetectionServiceError(f"An error occurred during detection: {e}")

# --- Singleton Provider ---
# This section ensures that only one instance of DetectionService ever exists.
_detection_service_instance: Optional[DetectionService] = None

def get_detection_service() -> DetectionService:
    """
    Acts as a singleton provider for the DetectionService.
    
    This function ensures that the service is only initialized once,
    and the same instance is returned on every subsequent call.
    """
    global _detection_service_instance
    if _detection_service_instance is None:
        _detection_service_instance = DetectionService()
    return _detection_service_instance

