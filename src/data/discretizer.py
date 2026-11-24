"""
discretisation from continuous to binary
main entry is GeneExpressionDiscretizer
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

class DiscretizationMethod(Enum):
    THRESHOLD = "threshold"
    MEDIAN = "median"
    MEAN = "mean"
    QUANTILE = "quantile"

class BaseDiscretizer(ABC):
    def __init__(self) -> None:
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, data: np.ndarray) -> "BaseDiscretizer":
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")

class ThresholdDiscretizer(BaseDiscretizer):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def fit(self, data: np.ndarray) -> "ThresholdDiscretizer":
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (data >= self.threshold).astype(np.int8)

    def save_parameters(self, filepath: str | Path) -> None:
        self._check_fitted()
        np.savez(filepath, threshold=self.threshold)

    def load_parameters(self, filepath: str | Path) -> "ThresholdDiscretizer":
        params = np.load(filepath)
        self.threshold = float(params["threshold"])
        self._fitted = True
        return self

class MedianDiscretizer(BaseDiscretizer):
    def __init__(self) -> None:
        super().__init__()
        self._thresholds: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "MedianDiscretizer":
        self._thresholds = np.median(data, axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (data >= self._thresholds).astype(np.int8)

    def save_parameters(self, filepath: str | Path) -> None:
        self._check_fitted()
        np.savez(filepath, thresholds=self._thresholds)

    def load_parameters(self, filepath: str | Path) -> "MedianDiscretizer":
        params = np.load(filepath)
        self._thresholds = params["thresholds"]
        self._fitted = True
        return self

class MeanDiscretizer(BaseDiscretizer):
    def __init__(self) -> None:
        super().__init__()
        self._thresholds: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "MeanDiscretizer":
        self._thresholds = np.mean(data, axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (data >= self._thresholds).astype(np.int8)

    def save_parameters(self, filepath: str | Path) -> None:
        self._check_fitted()
        np.savez(filepath, thresholds=self._thresholds)

    def load_parameters(self, filepath: str | Path) -> "MeanDiscretizer":
        params = np.load(filepath)
        self._thresholds = params["thresholds"]
        self._fitted = True
        return self

class QuantileDiscretizer(BaseDiscretizer):
    def __init__(self, quantile: float = 0.5) -> None:
        super().__init__()
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile
        self._thresholds: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "QuantileDiscretizer":
        self._thresholds = np.quantile(data, self.quantile, axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (data >= self._thresholds).astype(np.int8)

    def save_parameters(self, filepath: str | Path) -> None:
        self._check_fitted()
        np.savez(filepath, thresholds=self._thresholds, quantile=self.quantile)

    def load_parameters(self, filepath: str | Path) -> "QuantileDiscretizer":
        params = np.load(filepath)
        self._thresholds = params["thresholds"]
        self.quantile = float(params["quantile"])
        self._fitted = True
        return self

class GeneExpressionDiscretizer:
    """wrapper around the specific discretiser classes"""
    def __init__(self, method: DiscretizationMethod = DiscretizationMethod.MEDIAN, **kwargs) -> None:
        self.method = method
        self._discretizer: BaseDiscretizer = self._create_discretizer(**kwargs)

    @property
    def is_fitted(self) -> bool:
        return self._discretizer.is_fitted

    def fit(self, data: np.ndarray) -> "GeneExpressionDiscretizer":
        self._discretizer.fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        discretise values into {0,1}

        Returns:
            X_bin: np.ndarray, dtype int8
        """
        return self._discretizer.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        fit then discretise

        Returns:
            X_bin: np.ndarray, dtype int8
        """
        return self.fit(data).transform(data)

    def save_parameters(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        method_path = filepath.parent / f"{filepath.stem}_method.npz"
        np.savez(method_path, method=self.method.value)
        self._discretizer.save_parameters(filepath)

    def load_parameters(self, filepath: str | Path) -> "GeneExpressionDiscretizer":
        filepath = Path(filepath)
        method_path = filepath.parent / f"{filepath.stem}_method.npz"
        method_params = np.load(method_path, allow_pickle=True)
        self.method = DiscretizationMethod(str(method_params["method"]))
        self._discretizer = self._create_discretizer()
        self._discretizer.load_parameters(filepath)
        return self

    def _create_discretizer(self, **kwargs) -> BaseDiscretizer:
        if self.method == DiscretizationMethod.THRESHOLD:
            threshold = kwargs.get("threshold", 0.5)
            return ThresholdDiscretizer(threshold=threshold)
        if self.method == DiscretizationMethod.MEDIAN:
            return MedianDiscretizer()
        if self.method == DiscretizationMethod.MEAN:
            return MeanDiscretizer()
        if self.method == DiscretizationMethod.QUANTILE:
            quantile = kwargs.get("quantile", 0.5)
            return QuantileDiscretizer(quantile=quantile)
        raise NotImplementedError(f"Method {self.method} not implemented")
