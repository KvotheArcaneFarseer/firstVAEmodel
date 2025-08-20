# app/services/detection_service.py
# 說明：此模組包含檢測過程的業務邏輯。
# 它採用了延遲初始化和強健的錯誤處理等專業實踐。

from typing import Optional, List
import torch

# 匯入我們需要的核心組件和設定
from ..core.anomaly_detector import AnomalyDetector, AnomalyResult
from ..core.preprocessor import DataPreprocessor
from ..config.settings import settings
from ..dto.response_dto import DetectionResponse

# 為服務層建立一個自訂的錯誤類型
class DetectionServiceError(Exception):
    """檢測服務中發生的特定錯誤。"""
    pass

class DetectionService:
    """
    協調異常檢測過程，作為 API 層和核心 ML 邏輯之間的中介。
    """
    def __init__(self):
        """
        初始化檢測服務。
        注意：我們在這裡不載入模型，以實現延遲初始化。
        """
        # 延遲初始化變數
        self._detector: Optional[AnomalyDetector] = None
        self._initialization_error: Optional[str] = None

    def _initialize_detector(self):
        """
        一個私有的輔助方法，負責載入所有核心 ML 組件。
        這只會在第一次需要時被呼叫。
        """
        try:
            # 從設定中取得模型和 scaler 的路徑
            model_path = settings.get_model_path()
            scaler_path = settings.get_scaler_path()
            device = settings.get_torch_device()

            # 步驟 1: 建立預處理器實例
            preprocessor = DataPreprocessor(scaler_path=scaler_path)

            # 步驟 2: 建立 VAE 模型實例並載入權重
            # (注意：在一個更進階的版本中，這部分也可以被抽離到一個專門的 ModelLoader 類別中)
            from ..core.vae_model import VAE  # 局部匯入以避免循環依賴
            model = VAE(
                input_dim=settings.model.input_dim,
                hidden_dims=settings.model.hidden_dims,
                latent_dim=settings.model.latent_dim
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # 步驟 3: 建立核心的異常檢測器
            self._detector = AnomalyDetector(
                model=model,
                preprocessor=preprocessor,
                device=device
            )
        except Exception as e:
            # 如果在初始化過程中發生任何錯誤，我們捕捉它並儲存錯誤訊息。
            self._initialization_error = str(e)
            # 確保 _detector 保持為 None
            self._detector = None
            # 重新引發一個服務層級的錯誤
            raise DetectionServiceError(f"無法初始化檢測器: {e}")

    def detect_anomaly(self, features: List[float]) -> DetectionResponse:
        """
        處理單個異常檢測請求的業務邏輯。

        Args:
            features (List[float]): 來自 API 請求的 15 個原始特徵值。

        Returns:
            一個準備好作為 JSON 回應傳回的 DetectionResponse DTO。
        """
        try:
            # 步驟 1: 確保檢測器已經被初始化 (延遲初始化)
            # 如果 _detector 尚未被建立，這個 if 區塊將會執行。
            if self._detector is None:
                # 檢查是否有先前儲存的初始化錯誤
                if self._initialization_error:
                    raise DetectionServiceError(f"檢測器先前初始化失敗: {self._initialization_error}")
                # 如果沒有錯誤，則進行初始化
                self._initialize_detector()
            
            # 步驟 2: 從我們的設定中取得當前的異常閾值。
            threshold = settings.detection.threshold

            # 步驟 3: 使用異常檢測器執行完整的分析。
            result: AnomalyResult = self._detector.detect(
                raw_data=features,
                threshold=threshold
            )

            # 步驟 4: 將核心邏輯的詳細 AnomalyResult 格式化為
            # API 使用者期望的、更簡單的 DetectionResponse DTO。
            return DetectionResponse(
                reconstruction_error=result.reconstruction_error,
                is_anomaly=result.is_anomaly,
                threshold=result.threshold
            )
        except Exception as e:
            # 捕捉任何在檢測過程中發生的錯誤，並將其包裝成一個服務層級的錯誤。
            # 在更進階的版本中，我們會在這裡記錄錯誤並返回一個格式化的錯誤回應。
            raise DetectionServiceError(f"執行檢測時發生錯誤: {e}")

