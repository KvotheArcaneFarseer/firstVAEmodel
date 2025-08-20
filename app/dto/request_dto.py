# app/dto/request_dto.py
# Description: This file defines the Pydantic models for all incoming API requests,
# following the best practice of using named fields for clarity and validation.

from pydantic import BaseModel, Field
from typing import List, Optional

class PlayerFeatureData(BaseModel):
    """
    Defines the 15 specific features required for the VAE model.
    Using named fields is much safer and more explicit than a simple list.
    The feature names are taken directly from your training script.
    """
    account_stddev: float
    delta_profit_p25: float
    profit_mean: float
    payout_max: float
    bet_mean: float
    profit_max: float
    profit_median: float
    bet_median: float
    delta_bet_min: float
    delta_bet_max: float
    win_streak_mean: float
    delta_profit_min: float
    bet_max: float
    profit_p75: float
    delta_profit_max: float

    def to_list(self) -> List[float]:
        """
        A helper method to convert this object into the exact ordered list
        that the machine learning model expects.
        """
        return [
            self.account_stddev, self.delta_profit_p25, self.profit_mean, 
            self.payout_max, self.bet_mean, self.profit_max, self.profit_median, 
            self.bet_median, self.delta_bet_min, self.delta_bet_max, 
            self.win_streak_mean, self.delta_profit_min, self.bet_max, 
            self.profit_p75, self.delta_profit_max
        ]

class DetectionRequest(BaseModel):
    """
    Defines the structure for a single anomaly detection request.
    It contains the core feature data plus optional metadata for tracking.
    """
    player_features: PlayerFeatureData = Field(
        ...,
        description="An object containing the 15 named player features."
    )
    request_id: Optional[str] = Field(
        None,
        description="An optional, unique identifier for this request for tracking purposes."
    )
    player_id: Optional[str] = Field(
        None,
        description="An optional identifier for the player."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "player_features": {
                    "account_stddev": 0.5, "delta_profit_p25": 1.2, "profit_mean": -0.3,
                    "payout_max": 0.8, "bet_mean": -0.1, "profit_max": 0.7,
                    "profit_median": 0.2, "bet_median": -0.5, "delta_bet_min": 1.1,
                    "delta_bet_max": 0.4, "win_streak_mean": -0.2, "delta_profit_min": 0.9,
                    "bet_max": 0.1, "profit_p75": -0.4, "delta_profit_max": 0.6
                },
                "request_id": "req-12345-abcde",
                "player_id": "player-9876"
            }
        }
