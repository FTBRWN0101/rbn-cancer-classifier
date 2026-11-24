"""
preprocessing and normalisation for gene expression
main entry is DataPreprocessor
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np

class NormalizationStrategy(Enum):
    """normalisation choices"""
    MIN_MAX = "minmax"
    Z_SCORE = "zscore"
    ROBUST = "robust"

class ImputationStrategy(Enum):
    """missing value handling"""
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"

class BaseNormalizer(ABC):
    """base class for normalisers"""
    def __init__(self) -> None:
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, data: np.ndarray) -> "BaseNormalizer":
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")

class MinMaxNormalizer(BaseNormalizer):
    """min max scaler to [0, 1]"""
    def __init__(self, epsilon: float = 1e-10) -> None:
        super().__init__()
        self._min_vals: Optional[np.ndarray] = None
        self._max_vals: Optional[np.ndarray] = None
        self._epsilon = epsilon

    def fit(self, data: np.ndarray) -> "MinMaxNormalizer":
        self._min_vals = np.min(data, axis=0)
        self._max_vals = np.max(data, axis=0)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")
        range_vals = self._max_vals - self._min_vals
        range_vals = np.where(np.abs(range_vals) < self._epsilon, 1.0, range_vals)
        normalized = (data - self._min_vals) / range_vals
        return np.clip(normalized, 0.0, 1.0)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")
        range_vals = self._max_vals - self._min_vals
        return data * range_vals + self._min_vals

    def save_parameters(self, filepath: str | Path) -> None:
        if not self._fitted:
            raise ValueError(f"{self.__class__.__name__} must be fitted before transform")
        np.savez(filepath, min_vals=self._min_vals, max_vals=self._max_vals, epsilon=self._epsilon)

    def load_parameters(self, filepath: str | Path) -> "MinMaxNormalizer":
        params = np.load(filepath)
        self._min_vals = params["min_vals"]
        self._max_vals = params["max_vals"]
        self._epsilon = float(params["epsilon"])
        self._fitted = True
        return self

class DataPreprocessor:
    """impute, drop constant, then normalise"""
    def __init__(
        self,
        imputation_strategy: ImputationStrategy = ImputationStrategy.MEAN,
        normalization_strategy: NormalizationStrategy = NormalizationStrategy.MIN_MAX,
        variance_threshold: float = 1e-10,
    ) -> None:
        self.imputation_strategy = imputation_strategy
        self.normalization_strategy = normalization_strategy
        self.variance_threshold = variance_threshold
        self._normalizer: BaseNormalizer = self._create_normalizer()
        self._kept_feature_indices: Optional[np.ndarray] = None
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def kept_features(self) -> Optional[np.ndarray]:
        return self._kept_feature_indices

    def fit(self, data: np.ndarray) -> "DataPreprocessor":
        """fit preprocessing params from raw data"""
        data_clean = self._impute_missing(data)
        self._kept_feature_indices = self._identify_variable_features(data_clean)
        data_filtered = data_clean[:, self._kept_feature_indices]
        self._normalizer.fit(data_filtered)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        transform using fitted params

        Returns:
            X_out: np.ndarray, filtered then normalised
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        data_clean = self._impute_missing(data)
        data_filtered = data_clean[:, self._kept_feature_indices]
        return self._normalizer.transform(data_filtered)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        fit then transform

        Returns:
            X_out: np.ndarray, filtered then normalised
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """inverse normalisation for kept features only"""
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        return self._normalizer.inverse_transform(data)

    def save_parameters(self, filepath: str | Path) -> None:
        """save kept feature indices and normaliser params"""
        if not self._fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        normalizer_path = filepath.parent / f"{filepath.stem}_normalizer.npz"
        self._normalizer.save_parameters(normalizer_path)
        np.savez(
            filepath,
            kept_feature_indices=self._kept_feature_indices,
            imputation_strategy=self.imputation_strategy.value,
            normalization_strategy=self.normalization_strategy.value,
            variance_threshold=self.variance_threshold,
        )

    def load_parameters(self, filepath: str | Path) -> "DataPreprocessor":
        """load kept feature indices and normaliser params"""
        filepath = Path(filepath)
        params = np.load(filepath, allow_pickle=True)
        self._kept_feature_indices = params["kept_feature_indices"]
        self.imputation_strategy = ImputationStrategy(str(params["imputation_strategy"]))
        self.normalization_strategy = NormalizationStrategy(str(params["normalization_strategy"]))
        self.variance_threshold = float(params["variance_threshold"])
        normalizer_path = filepath.parent / f"{filepath.stem}_normalizer.npz"
        self._normalizer = self._create_normalizer()
        self._normalizer.load_parameters(normalizer_path)
        self._fitted = True
        return self

    def _create_normalizer(self) -> BaseNormalizer:
        if self.normalization_strategy == NormalizationStrategy.MIN_MAX:
            return MinMaxNormalizer()
        raise NotImplementedError(f"Normalization strategy {self.normalization_strategy} not implemented")

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        if not np.isnan(data).any():
            return data
        data_imputed = data.copy()
        if self.imputation_strategy == ImputationStrategy.MEAN:
            col_means = np.nanmean(data, axis=0)
            indices = np.where(np.isnan(data))
            data_imputed[indices] = np.take(col_means, indices[1])
        elif self.imputation_strategy == ImputationStrategy.MEDIAN:
            col_medians = np.nanmedian(data, axis=0)
            indices = np.where(np.isnan(data))
            data_imputed[indices] = np.take(col_medians, indices[1])
        elif self.imputation_strategy == ImputationStrategy.ZERO:
            data_imputed = np.nan_to_num(data, nan=0.0)
        return data_imputed

    def _identify_variable_features(self, data: np.ndarray) -> np.ndarray:
        variances = np.var(data, axis=0)
        kept_indices = np.where(variances > self.variance_threshold)[0]
        n_removed = data.shape[1] - len(kept_indices)
        if n_removed > 0:
            print(f"Removed {n_removed} constant features (kept {len(kept_indices)} features)")
        return kept_indices
