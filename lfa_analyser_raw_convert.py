"""
Minimal LFA Analysis - Direct R to Python Translation

Image >> Grayscale >> Artifact removal >> Column means >> 
Baseline anchors >> Linear correction >> Signal inversion >> 
Moving average >> ALS correction >> Peak detection >> Ratio
"""

import numpy as np
from scipy.ndimage import convolve1d
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.stats import trim_mean
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def baseline_als(y, lam=5, p=0.01, niter=20):
    """
    Asymmetric Least Squares baseline correction.
    
    Parameters
    ----------
    y : np.ndarray
        Input signal (1D array)
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
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)
    
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return y - z

def _plot(signal):
        # 11. Plot
    plt.figure(figsize=(10, 4))
    plt.plot(signal, linewidth=2, color='blue')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.title('LFA Intensity Profile')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def read_lfa(image_path):
    """
    Read and analyse LFA image following original R methodology.
    
    This function:
    1. Loads image and converts to grayscale
    2. Removes artifacts via quantile filtering (per column)
    3. Calculates column means
    4. Extracts baseline anchor points (first, middle, last regions)
    5. Applies piecewise linear baseline correction
    6. Inverts signal (1/mean for LFA strips)
    7. Applies moving average smoothing (window=10)
    8. Applies ALS baseline correction
    9. Detects peaks in first/second halves
    10. Calculates TL1/TL2 ratio
    
    Parameters
    ----------
    image_path : str
        Path to LFA image file
    
    Returns
    -------
    tuple
        (TL1_peak, control_peak, ratio, corrected_intensity)
    """
    # 1. Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    data = np.array(img, dtype=float) / 255.0

    # 2. Remove artifacts: exclude <5% and >95% per column
    q_high = np.percentile(data, 95, axis=0, keepdims=True)
    q_low = np.percentile(data, 5, axis=0, keepdims=True)
    data = np.where((data > q_high) | (data < q_low), np.nan, data)
    
    # 3. Column means (mean of each column)
    col_means = np.nanmean(data, axis=0)
    
    # 4. Extract baseline regions (first, middle, last 20 pixels)
    n = len(col_means)
    first_region = np.median(col_means[:20])
    mid_region = np.median(col_means[n//2 - 10 : n//2 + 10])
    last_region = np.median(col_means[-20:])

    # 5. Piecewise linear baseline correction
    slope_first = (mid_region - first_region) * 2 / n
    slope_last = (last_region - mid_region) * 2 / n

    x = np.arange(n)
    baseline = np.zeros(n)
    baseline[:n//2] = first_region + slope_first * x[:n//2]
    baseline[n//2:] = mid_region + slope_last * (x[n//2:] - n//2)
    
    col_means_corrected = col_means - baseline
    _plot(col_means_corrected)

    # 6. Invert signal (LFA strips are darker at peaks)
    epsilon = 0.5
    intensity = 1.0 / (col_means_corrected + epsilon)
    intensity = np.where(np.isfinite(intensity), intensity, np.nan)
    _plot(intensity)

    # 7. Moving average smoothing (window=10)
    kernel = np.ones(6) / 6
    intensity_smooth = convolve1d(intensity, kernel, mode='nearest')
    _plot(intensity_smooth)

    # 8. ALS baseline correction
    intensity_corrected = baseline_als(intensity_smooth, lam=1000, p=0.01, niter=20)
    
    # 9. Detect peaks in each half
    half = n // 2
    TL1_region = intensity_corrected[:half]
    TL1_peak_idx = np.nanargmax(TL1_region)
    TL1_peak_value = TL1_region[TL1_peak_idx]

    TL2_region = intensity_corrected[half:]
    TL2_peak_idx_relative = np.nanargmax(TL2_region)
    TL2_peak_idx_absolute = half + TL2_peak_idx_relative  # ← Add offset!
    TL2_peak_value = TL2_region[TL2_peak_idx_relative]
    
    # 10. Calculate ratio
    ratio = TL1_peak_value / TL2_peak_value if TL2_peak_value > 0 else np.nan
    
    # Print results
    print(f'TL1_peak: {TL1_peak_value:.3f}')
    print(f'TL1_peak_location: {TL1_peak_idx}')
    print(f'TL2_peak: {TL2_peak_value:.3f}')
    # print(f'TL2_peak_location_relative: {TL2_peak_idx_relative}')
    print(f'TL2_peak_location_absolute: {TL2_peak_idx_absolute}')
    print(f'ratio: {ratio:.3f}')
    # print(f'midpoint: {half}')

    _plot(intensity_corrected)
    return TL1_peak_value, TL2_peak_value, ratio, intensity_corrected


# Usage
if __name__ == '__main__':
    TL1_peak, TL2_peak, ratio, profile = read_lfa("Picture3.png")