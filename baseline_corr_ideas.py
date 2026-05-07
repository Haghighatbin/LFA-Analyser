# 1. airPLS (Adaptive Iteratively Reweighted Penalised Least Squares)
# Zhang et al. (2010), Analyst
# Advantages over ALS:

# Automatically adapts to baseline shape
# No need to tune the p parameter
# Better at handling steep baseline changes
# More robust to varying peak widths

# from scipy.sparse import csc_matrix, eye, diags
# from scipy.sparse.linalg import spsolve

# def baseline_airpls(y, lam=100, porder=1, itermax=15):
#     """
#     Adaptive Iteratively Reweighted Penalised Least Squares for baseline correction.
    
#     Better than ALS for signals with:
#     - Steep baseline changes
#     - Varying peak widths
#     - Unknown baseline asymmetry
    
#     Parameters
#     ----------
#     y : np.ndarray
#         Input signal
#     lam : float
#         Smoothness (100-1e9, larger = smoother)
#     porder : int
#         Order of difference penalty (1 or 2)
#     itermax : int
#         Maximum iterations
    
#     Returns
#     -------
#     np.ndarray
#         Baseline-corrected signal
    
#     Reference
#     ---------
#     Zhang et al. (2010) Analyst, 135(5), 1138-1146
#     """
#     L = len(y)
#     D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)) if porder == 2 else \
#         diags([1, -1], [0, -1], shape=(L, L-1))
#     D = lam * D.dot(D.T)
    
#     w = np.ones(L)
#     for i in range(itermax):
#         W = diags(w, 0, shape=(L, L))
#         Z = W + D
#         z = spsolve(csc_matrix(Z), w * y)
        
#         d = y - z
#         dssn = np.abs(d[d < 0].sum())
        
#         if dssn < 0.001 * (abs(y)).sum() or i == itermax - 1:
#             break
            
#         w = (d < 0) + (d >= 0) * np.exp(2 * d / dssn)
    
#     return y - z

# ################################################################################################ #
# ################################################################################################ #

# 2. arPLS (Asymmetrically Reweighted Penalised Least Squares)
# Baek et al. (2015), Analyst
# def baseline_arpls(y, lam=1e4, ratio=0.05, itermax=100):
#     """
#     Asymmetrically Reweighted Penalized Least Squares.
    
#     Improvements over airPLS:
#     - More stable convergence
#     - Better negative peak handling
#     - Faster computation
    
#     Parameters
#     ----------
#     y : np.ndarray
#         Input signal
#     lam : float
#         Smoothness parameter (1e3 - 1e9)
#     ratio : float
#         Convergence threshold (0.001 - 0.1)
#     itermax : int
#         Maximum iterations
    
#     Returns
#     -------
#     np.ndarray
#         Baseline-corrected signal
        
#     Reference
#     ---------
#     Baek et al. (2015) Analyst, 140(1), 250-257
#     """
#     L = len(y)
#     D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
#     D = lam * D.dot(D.T)
    
#     w = np.ones(L)
#     for i in range(itermax):
#         W = diags(w, 0, shape=(L, L))
#         Z = W + D
#         z = spsolve(csc_matrix(Z), w * y)
        
#         d = y - z
#         dn = d[d < 0]
        
#         m = np.mean(dn)
#         s = np.std(dn)
        
#         wt = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        
#         if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
#             break
#         w = wt
    
#     return y - z

# ################################################################################################ #
# ################################################################################################ #

# 3. Morphological Opening (Rolling Ball)
# Sternberg (1983) - Used in ImageJ
# from scipy.ndimage import grey_opening, grey_closing

# def baseline_morphological(y, window=50):
#     """
#     Morphological baseline estimation using rolling ball algorithm.
    
#     Advantages:
#     - Non-parametric (no tuning needed)
#     - Very fast
#     - Works well for smooth baselines
#     - Intuitive window size parameter
    
#     Parameters
#     ----------
#     y : np.ndarray
#         Input signal
#     window : int
#         Rolling ball radius (pixels)
#         Should be larger than peak width
    
#     Returns
#     -------
#     np.ndarray
#         Baseline-corrected signal
#     """
#     # Opening removes peaks, keeping valleys/baseline
#     baseline = grey_opening(y, size=window)
    
#     # Optional: smooth the baseline
#     from scipy.ndimage import gaussian_filter1d
#     baseline = gaussian_filter1d(baseline, sigma=window//10)
    
#     return y - baseline

# ################################################################################################ #
# ################################################################################################ #

# 4. Wavelet-Based Baseline Correction
# using wavelet decomposition

# import pywt

# def baseline_wavelet(y, wavelet='db4', level=3):
#     """
#     Wavelet-based baseline estimation.
    
#     Advantages:
#     - Handles non-stationary baselines
#     - Preserves sharp peaks
#     - Multi-resolution analysis
    
#     Parameters
#     ----------
#     y : np.ndarray
#         Input signal
#     wavelet : str
#         Wavelet type ('db4', 'sym5', 'coif3')
#     level : int
#         Decomposition level
    
#     Returns
#     -------
#     np.ndarray
#         Baseline-corrected signal
#     """
#     # Decompose signal
#     coeffs = pywt.wavedec(y, wavelet, level=level)
    
#     # Zero out detail coefficients (keep only approximation)
#     coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    
#     # Reconstruct baseline
#     baseline = pywt.waverec(coeffs, wavelet)[:len(y)]
    
#     return y - baseline

# ################################################################################################ #
# ################################################################################################ #

# 5. Deep Learning Baseline Correction
# Recent papers (2020-2024)
# # Pseudo-code for DL approach
# import torch
# import torch.nn as nn

# class BaselineNet(nn.Module):
#     """
#     1D CNN for baseline estimation.
    
#     Train on synthetic LFA data with known baselines.
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=15, padding=7),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, kernel_size=15, padding=7),
#             nn.ReLU(),
#             nn.Conv1d(64, 1, kernel_size=15, padding=7)
#         )
    
#     def forward(self, x):
#         # Input: (batch, 1, signal_length)
#         baseline = self.encoder(x)
#         return baseline

# # Usage:
# # model = BaselineNet()
# # baseline = model(signal)
# # corrected = signal - baseline

# ################################################################################################ #
# ################################################################################################ #
# arPLS
# def analyse_lfa_v2(image_array, ...):
#     # ... steps 1-4 same ...
    
#     # OPTION 1: Skip piecewise linear entirely, straight to arPLS
#     col_means_simple = col_means  # No linear correction
#     intensity = 1.0 / col_means_simple
#     intensity_smooth = _moving_average(intensity, smooth_window)
    
#     # arPLS instead of ALS
#     intensity_corrected = baseline_arpls(intensity_smooth, lam=1e4)
    
#     # ... rest is same ...

# ################################################################################################ #
# ################################################################################################ #

# adaptive method selection:
# def analyse_lfa_v2(image_array, baseline_method='auto'):
#     """
#     baseline_method : str
#         'auto', 'arpls', 'airpls', 'morphological', 'wavelet', 'als'
#     """
#     if baseline_method == 'auto':
#         # Detect baseline complexity
#         if _has_steep_baseline(col_means):
#             method = 'arpls'
#         elif _is_smooth_baseline(col_means):
#             method = 'morphological'
#         else:
#             method = 'airpls'
    
#     # Apply selected method
#     intensity_corrected = BASELINE_METHODS[method](intensity_smooth)