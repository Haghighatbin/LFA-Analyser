"""
Core LFA Analysis Module

Translated from R script with enhancements for robustness and parametrisation.
Maintains original methodology whilst improving code quality.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings
from lfa_analyser.config import Config


def analyse_lfa(
    image_array: np.ndarray,
    quantile_low: float = Config.QUANTILE_LOW_VAL,
    quantile_high: float = Config.QUANTILE_HIGH_VAL,
    smooth_window: int = Config.SMOOTH_WIN_VAL,
    baseline_region_size: int = Config.BASELINE_REGION_VAL,
    als_lambda: float = Config.ALS_LAMBDA_VAL,
    als_p: float = Config.ALS_P_SYM_VAL,
    als_niter: int = Config.ALS_N_ITER_VAL,
    epsilon = Config.EPSILON,
    auc_window = Config.AUC_WIN_VAL
) -> Dict:
    """
    Analyse lateral flow assay image.
    
    Parameters
    ----------
    image_array : np.ndarray
        Grayscale image as 2D numpy array (height x width)
    quantile_low : float, optional
        Lower quantile for artifact removal (default: 0.05)
    quantile_high : float, optional
        Upper quantile for artifact removal (default: 0.95)
    smooth_window : int, optional
        Moving average window size (default: 10)
    baseline_region_size : int, optional
        Number of pixels for baseline regions (default: 20)
    als_lambda : float, optional
        ALS smoothness parameter (default: 5.0)
    als_p : float, optional
        ALS asymmetry parameter (default: 0.01)
    als_niter : int, optional
        ALS iteration count (default: 20)
    
    Returns
    -------
    dict
        Results dictionary containing:
        - 'TL1_peak': Maximum intensity in first half
        - 'TL2_peak': Maximum intensity in second half
        - 'TL1/TL2 peak ratio': TL1_peak / TL2_peak
        - 'TL1_auc': Area under curve for TL1 region
        - 'TL2_auc': Area under curve for TL2 region
        - 'TL1/TL2 auc_ratio': TL1_auc / TL2_auc
        - 'intensity_profile': Baseline-corrected intensity array
        - 'raw_profile': Pre-correction intensity profile
        - 'metadata': Analysis parameters used
    
    Raises
    ------
    ValueError
        If image dimensions are too small for analysis
    """
    
    # Validate inputs
    if image_array.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {image_array.ndim}D array")
    
    height, width = image_array.shape
    
    if width < baseline_region_size * 3:
        raise ValueError(
            f"Image width ({width}px) too small for baseline regions "
            f"(need at least {baseline_region_size * 3}px)"
        )
    
    # Step 1: Remove artifacts using quantile filtering (per column)
    data_filtered = _remove_artifacts(image_array, quantile_low, quantile_high)
    
    # Step 2: Calculate column means (perpendicular to strip direction)
    col_means = np.nanmean(data_filtered, axis=0)
    
    # Step 3: Extract baseline regions
    baseline_anchors = _extract_baseline_regions(col_means, baseline_region_size)
    
    # Step 4: Apply linear baseline correction
    col_means_corrected = _apply_linear_baseline(
        col_means, 
        baseline_anchors, 
        baseline_region_size
    )
    
    # Step 5: Invert signal (LFA strips are darker at peaks)
    intensity_raw = 1.0 / (col_means_corrected + epsilon)
    
    # Handle infinities and NaNs
    intensity_raw = np.where(np.isfinite(intensity_raw), intensity_raw, np.nan)
    
    # Step 6: Apply moving average smoothing
    intensity_smooth = _moving_average(intensity_raw, smooth_window)
    
    # Step 7: ALS baseline correction
    intensity_corrected = baseline_als(
        intensity_smooth, 
        lam=als_lambda, 
        p=als_p, 
        niter=als_niter
    )
    
    # Step 8: Detect peaks in each half
    n_pixels = len(intensity_corrected)
    half_point = n_pixels // 2
    
    # Valid peak zones (distance from edge artefacts)
    ### Wrong and a temporary approach until we fixed the LFA fabrication design ###
    TL1_VALID_MIN = Config.LEFT_CUT_OFF
    TL2_VALID_MAX = Config.RIGHT_CUT_OFF

    # --- TL1 Peak Detection and Validation ---
    TL1_peak_idx = np.nanargmax(intensity_corrected[:half_point])

    if TL1_VALID_MIN < TL1_peak_idx < half_point:
        TL1_peak = np.nanmax(intensity_corrected[:half_point])
        TL1_start = max(0, TL1_peak_idx - auc_window)
        TL1_end = min(half_point, TL1_peak_idx + auc_window + 1)
        TL1_auc = np.trapz(np.maximum(intensity_corrected[TL1_start:TL1_end], 0))
    else: 
        warnings.warn("TL1 peak is zero or invalid, ratio set to NaN")
        TL1_peak_idx = None
        TL1_peak = np.nan
        TL1_auc = np.nan

    TL2_peak_idx = half_point + np.nanargmax(intensity_corrected[half_point:])
    if half_point < TL2_peak_idx < TL2_VALID_MAX:
        TL2_peak = np.nanmax(intensity_corrected[half_point:])
        TL2_start = max(half_point, TL2_peak_idx - auc_window)
        TL2_end = min(n_pixels, TL2_peak_idx + auc_window + 1)
        TL2_auc = np.trapz(np.maximum(intensity_corrected[TL2_start:TL2_end], 0))
    else: 
        warnings.warn("TL2 peak is zero or invalid, ratio set to NaN")
        TL2_peak_idx = None
        TL2_peak = np.nan
        TL2_auc = np.nan
    
    # --- Calculate Ratios (with NaN and division-by-zero protection) ---
    if np.isnan(TL1_peak) or np.isnan(TL2_peak) or TL2_peak == 0:
        ratio = np.nan
    else:
        ratio = TL1_peak / TL2_peak
    
    if np.isnan(TL1_auc) or np.isnan(TL2_auc) or TL2_auc == 0:
        auc_ratio = np.nan
    else:
        auc_ratio = TL1_auc / TL2_auc

    return {
        'TL1_peak': float(TL1_peak),
        'TL2_peak': float(TL2_peak),
        'ratio': float(ratio),
        'TL1_auc': float(TL1_auc),
        'TL2_auc': float(TL2_auc),
        'auc_ratio': float(auc_ratio),
        'TL1_peak_idx': TL1_peak_idx,
        'TL2_peak_idx': TL2_peak_idx,
        'intensity_profile': intensity_corrected,
        'raw_profile': intensity_smooth,
        'metadata': {
            'image_size': (height, width),
            'quantile_low': quantile_low,
            'quantile_high': quantile_high,
            'smooth_window': smooth_window,
            'baseline_region_size': baseline_region_size,
            'als_lambda': als_lambda,
            'als_p': als_p,
            'als_niter': als_niter,
            'auc_window': auc_window
        }
    }


def _remove_artifacts(
    image: np.ndarray, 
    q_low: float, 
    q_high: float
) -> np.ndarray:
    """
    Remove artifacts by replacing values outside quantile range with NaN.
    
    Applies per-column filtering to handle varying brightness across strip.
    """
    data = image.copy().astype(float)
    
    # Calculate quantiles per column
    q_high_vals = np.percentile(data, q_high * 100, axis=0, keepdims=True)
    q_low_vals = np.percentile(data, q_low * 100, axis=0, keepdims=True)
    
    # Mask values outside quantile range
    data = np.where(data > q_high_vals, np.nan, data)
    data = np.where(data < q_low_vals, np.nan, data)
    
    return data


def _extract_baseline_regions(
    col_means: np.ndarray, 
    region_size: int
) -> Tuple[float, float, float]:
    """
    Extract median values from first, middle, and last regions of strip.
    
    Returns
    -------
    tuple
        (first_region_median, middle_region_median, last_region_median)
    """
    n = len(col_means)
    mid_point = n // 2
    
    # First region
    first_region = col_means[:region_size]
    first_median = np.nanmedian(first_region)
    
    # Middle region (centered)
    mid_start = mid_point - region_size // 2
    mid_end = mid_point + region_size // 2
    middle_region = col_means[mid_start:mid_end]
    middle_median = np.nanmedian(middle_region)
    
    # Last region
    last_region = col_means[-region_size:]
    last_median = np.nanmedian(last_region)
    
    return first_median, middle_median, last_median


def _apply_linear_baseline(
    col_means: np.ndarray,
    baseline_anchors: Tuple[float, float, float],
    region_size: int
) -> np.ndarray:
    """
    Apply piecewise linear baseline correction.
    
    Corrects for uneven illumination using slopes between baseline regions.
    """
    first_val, mid_val, last_val = baseline_anchors
    n = len(col_means)
    half = n // 2
    
    # Calculate slopes
    slope_first = (mid_val - first_val) * 2 / n
    slope_last = (last_val - mid_val) * 2 / n
    
    # Build piecewise linear baseline
    x = np.arange(n)
    baseline = np.zeros(n)
    
    # First half
    baseline[:half] = first_val + slope_first * x[:half]
    
    # Second half
    baseline[half:] = mid_val + slope_last * (x[half:] - half)
    
    return col_means - baseline


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Apply moving average filter with edge padding.
    
    Uses uniform weights and replicates edge values to handle boundaries.
    """
    # Create uniform kernel
    kernel = np.ones(window) / window
    
    # Apply convolution with 'same' mode
    smoothed = np.convolve(signal, kernel, mode='same')
    
    # Fix edges by replicating boundary values
    edge_width = window // 2
    smoothed[:edge_width] = smoothed[edge_width]
    smoothed[-edge_width:] = smoothed[-edge_width-1]
    
    return smoothed


def baseline_als(
    y: np.ndarray, 
    lam: float = 1e5, 
    p: float = 0.01, 
    niter: int = 20
) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline correction.
    
    Based on Eilers & Boelens (2005) "Baseline Correction with Asymmetric Least Squares Smoothing"
    
    Parameters
    ----------
    y : np.ndarray
        Input signal
    lam : float
        Smoothness parameter (larger = smoother baseline)
    p : float
        Asymmetry parameter (0 < p < 1, smaller = more asymmetric)
    niter : int
        Number of iterations
    
    Returns
    -------
    np.ndarray
        Baseline-corrected signal
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    L = len(y)
    
    # Build difference matrix (2nd derivative)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    
    # Initialise weights
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        
        # Update weights asymmetrically
        w = p * (y > z) + (1 - p) * (y < z)
    
    return y - z