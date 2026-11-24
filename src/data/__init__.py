"""
data module
i load and prep ovarian gene expression data
- load arff
- preprocess normalise
- split train test
- discretise to binary
"""
from .loader import (
    ARFFLoader,
    DatasetMetadata,
    load_ovarian_data,
)
from .preprocessor import (
    DataPreprocessor,
    MinMaxNormalizer,
    BaseNormalizer,
    NormalizationStrategy,
    ImputationStrategy,
)
from .splitter import (
    DataSplitter,
    SplitStatistics,
)
from .discretizer import (
    GeneExpressionDiscretizer,
    ThresholdDiscretizer,
    MedianDiscretizer,
    MeanDiscretizer,
    QuantileDiscretizer,
    DiscretizationMethod,
)
from .runit import (
    main
)

__all__ = [
    "ARFFLoader",
    "DatasetMetadata",
    "load_ovarian_data",
    "DataPreprocessor",
    "MinMaxNormalizer",
    "BaseNormalizer",
    "NormalizationStrategy",
    "ImputationStrategy",
    "DataSplitter",
    "SplitStatistics",
    "GeneExpressionDiscretizer",
    "ThresholdDiscretizer",
    "MedianDiscretizer",
    "MeanDiscretizer",
    "QuantileDiscretizer",
    "DiscretizationMethod",
    "main",
]

__version__ = "1.0.0"
