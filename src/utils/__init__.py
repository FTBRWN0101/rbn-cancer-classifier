"""Utility Module"""
#set matplotlib backend before any imports
import os
os.environ['MPLBACKEND'] = 'Agg'
from .config import (RBNConfig,NKConfig,ExperimentConfig)
from .logging_utils import setup_logger
from .visualisation import plot_curve
from .paths import ProjectPaths
from .validation import (validate_binary_matrix,validate_shapes)
from .runit import main

__all__ = [
    "RBNConfig",
    "NKConfig",
    "ExperimentConfig",
    "setup_logger",
    "plot_curve",
    "ProjectPaths",
    "validate_binary_matrix",
    "validate_shapes",
    "main",]