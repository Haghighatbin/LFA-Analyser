"""
LFA Analyser - Lateral Flow Assay Image Analysis

A Python package for analysing lateral flow assay (LFA) test strips.
Translated from R methodology with enhancements for robustness and usability.

Main Components:
- core: Core analysis algorithms
- preprocessing: Image loading and validation
- utils: Export and visualisation utilities
- cli: Command-line interface
- app: Streamlit web application
"""

__version__ = "1.0.0"
__author__ = "AH"

from .core import analyse_lfa, baseline_als
from .preprocessing import load_image, validate_lfa_image, batch_load_images
from .utils import (
    export_results_to_csv,
    export_results_to_json,
    create_intensity_plot,
    print_results_summary
)

__all__ = [
    'analyse_lfa',
    'baseline_als',
    'load_image',
    'validate_lfa_image',
    'batch_load_images',
    'export_results_to_csv',
    'export_results_to_json',
    'create_intensity_plot',
    'print_results_summary'
]