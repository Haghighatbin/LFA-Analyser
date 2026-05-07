import os

class Config:
    # Artifact removal slider
    # Lower Quantile
    QUANTILE_LOW_MIN_VAL = 0.01
    QUANTILE_LOW_MAX_VAL = 0.15
    QUANTILE_LOW_VAL = 0.05
    QUANTILE_LOW_STEP = 0.01

    # Upper Quantile
    QUANTILE_HIGH_MIN_VAL = 0.85
    QUANTILE_HIGH_MAX_VAL = 0.99
    QUANTILE_HIGH_VAL = 0.95
    QUANTILE_HIGH_STEP = 0.01

    # Signal Processing
    SMOOTH_WIN_MIN_VAL = 3
    SMOOTH_WIN_MAX_VAL = 30
    SMOOTH_WIN_VAL = 4
    SMOOTH_WIN_STEP = 1

    # Baseline Region Size
    BASELINE_REGION_MIN_VAL = 10
    BASELINE_REGION_MAX_VAL = 50
    BASELINE_REGION_VAL = 20
    BASELINE_REGION_STEP = 5

    # AUC Window
    AUC_WIN_MIN_VAL = 3
    AUC_WIN_MAX_VAL = 20
    AUC_WIN_VAL = 5
    AUC_WIN_STEP = 1

    # ALS Lambda Smoothness
    ALS_LAMBDA_MIN_VAL = 0.1
    ALS_LAMBDA_MAX_VAL = 100.0
    ALS_LAMBDA_VAL = 2.5
    ALS_LAMBDA_STEP = 0.5
    ALS_LAMBDA_FORMAT = "%.1f"

    # ALS p (Asymmetry)
    ALS_P_SYM_MIN_VAL = 0.001
    ALS_P_SYM_MAX_VAL = 0.1
    ALS_P_SYM_VAL = 0.01
    ALS_P_SYM_STEP = 0.001
    ALS_P_SYM_FORMAT = "%.3f"

    # ALS n Iterations
    ALS_N_ITER_MIN_VAL = 5
    ALS_N_ITER_MAX_VAL = 50
    ALS_N_ITER_VAL = 20
    ALS_N_ITER_STEP = 5

    # Zero-Division 
    EPSILON = 0.5

    # CUT-OFF DETECTION ZONE
    LEFT_CUT_OFF = 56
    RIGHT_CUT_OFF = 142

    # Blur metric
    BLUR_METRIC = 0.0003


