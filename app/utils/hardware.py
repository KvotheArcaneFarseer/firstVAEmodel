# app/utils/hardware.py
# Description: A professional utility for inspecting hardware and intelligently selecting
# the optimal computation device for PyTorch.

import torch
import logging

# We will use the logging system we set up earlier.
from .logger import get_logger

logger = get_logger(__name__)

class DeviceSelector:
    """
    Intelligently selects the optimal torch.device for computation
    based on hardware availability and simple heuristics.
    """
    def get_optimal_device(self, force_cpu: bool = False) -> torch.device:
        """
        Checks for a CUDA-enabled GPU and returns the best device.

        This is a critical function for any production ML service to ensure
        it leverages available hardware acceleration.

        Args:
            force_cpu (bool): If True, will always return a CPU device,
                              ignoring any available GPUs. Useful for debugging.

        Returns:
            A torch.device object, either 'cuda' or 'cpu'.
        """
        if force_cpu:
            logger.warning("CPU device has been explicitly forced by configuration.")
            return torch.device("cpu")

        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                current_device_idx = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device_idx)
                
                logger.info(f"CUDA is available. Found {gpu_count} GPU(s).")
                logger.info(f"Using primary GPU: {gpu_name} (cuda:{current_device_idx})")
                
                return torch.device("cuda")
            except Exception as e:
                logger.error(f"CUDA was reported as available, but failed to get device properties: {e}")
                logger.warning("Falling back to CPU due to CUDA error.")
                return torch.device("cpu")
        else:
            logger.info("CUDA not available. Using CPU for computation.")
            return torch.device("cpu")

# --- Singleton Instance and Helper Function ---
# This pattern provides a single, shared instance of the DeviceSelector
# and a simple, globally accessible function for the rest of the app to use.
# This ensures that device selection logic is consistent everywhere.

_device_selector_instance = DeviceSelector()

def get_optimal_device(force_cpu: bool = False) -> torch.device:
    """
    A convenient, globally accessible function to get the optimal torch device.
    """
    return _device_selector_instance.get_optimal_device(force_cpu=force_cpu)
