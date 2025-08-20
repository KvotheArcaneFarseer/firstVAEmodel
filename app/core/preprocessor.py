# app/core/preprocessor.py
# Description: This file is responsible for loading the saved RobustScaler
# and providing a function to transform new data.

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from typing import List, cast
import pandas as pd
from typing import Union

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

# --- Configuration Class ---
# A dedicated class to hold all settings related to preprocessing.
class PreprocessorConfig:
    """Configuration for the DataProcessor."""
    # This list is critical. It's the exact 15 features your model was trained on, taken from the feature importance section of the training script.
    EXPECTED_FEATURES: List[str] = [
        'account_stddev', 'delta_profit_p25', 'profit_mean', 'payout_max', 'bet_mean',
        'profit_max', 'profit_median', 'bet_median', 'delta_bet_min', 'delta_bet_max',
        'win_streak_mean', 'delta_profit_min', 'bet_max', 'profit_p75', 'delta_profit_max'
    ]
    NUM_FEATURES: int = 15

# --- Main Preprocessor Class ---
# This class orchestrates the entire preprocessing workflow.
class DataPreprocessor:
    """ Validates and scales raw input data to prepare it for the VAE model."""
    

    def __init__(self, scaler_path: Path, config: PreprocessorConfig = PreprocessorConfig()):
        """Initializes the preprocessor.
        Args:
            scaler_path (Path): Absolute file path to the saved scaler.pkl file.
            config (PreprocessorConfig): Configuration for the preprocessor.
        """    
        self.config = config
        try:
            #load the scaler object fitted on the training data.
            self.scaler = cast(RobustScaler, joblib.load(scaler_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
        except Exception as e:
            raise IOError(f"Failed to load scaler from {scaler_path}: {e}")
        
    def _validate_input(self, data: Union[List[float], List[List[float]]]) -> pd.DataFrame:
        """Validates the structure and shape of the input data."""
        
        #Ensure data id a list.
        if not isinstance(data, list):
            raise DataValidationError("Input data must be a list or a list of lists.")
        
        # Convert to a NumPy array for shape checking.
        data_np = np.array(data)

        # If it's a 1D list (a single sample), reshape it to 2D.
        if data_np.ndim == 1:
            data_np = data_np.reshape(1, -1)

        #Check if the number of features is correct.
        if data_np.shape[1] != self.config.NUM_FEATURES:
            raise DataValidationError(
                f"Incorrect number of features. Expected {self.config.NUM_FEATURES}, got {data_np.shape[1]}."
            )
        
        # Convert the validated data to a pandas DataFrame weith the correct column names.
        return pd.DataFrame(data_np, columns=self.config.EXPECTED_FEATURES)
    
    def preprocess(self, data: Union[List[float], List[List[float]]]) -> np.ndarray:
        """Preprocesses the input data by validating and scaling it.
        
        Args:
            data: the raw input data, as a list or lists of lists.
            
        Returns:
            A NumPy array of the preprocessed data, ready for the model."""
        
        # Step 1: Validate the input data structure.
        validated_df = self._validate_input(data)

        # step 2: Scale the data using the loaded scaler.
        # .values converts the DataFrame to a NumPy array for the scaler.
        scaled_data = self.scaler.transform(validated_df.values)

        return scaled_data

        

