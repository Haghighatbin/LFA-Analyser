"""
Test Suite for LFA Analyser Core Module

Basic tests to validate core functionality.
"""

import numpy as np
import pytest

from lfa_analyser.core import (
    analyse_lfa,
    baseline_als,
    _remove_artifacts,
    _moving_average
)


class TestCoreAnalysis:
    """Test core analysis functions."""
    
    def test_analyse_lfa_basic(self):
        """Test basic analysis with synthetic image."""
        # Create synthetic LFA image (strip with two peaks)
        width = 200
        height = 50
        image = np.ones((height, width)) * 0.3
        
        # Add test peak (left side)
        image[:, 40:50] = 0.1  # Dark = high intensity after inversion
        
        # Add control peak (right side)
        image[:, 140:150] = 0.15
        
        # Run analysis
        results = analyse_lfa(image)
        
        # Assertions
        assert 'test_peak' in results
        assert 'control_peak' in results
        assert 'ratio' in results
        assert 'intensity_profile' in results
        
        assert results['test_peak'] > 0
        assert results['control_peak'] > 0
        assert isinstance(results['ratio'], float)
    
    def test_analyse_lfa_parameters(self):
        """Test that custom parameters are accepted."""
        image = np.random.rand(50, 200) * 0.5 + 0.3
        
        results = analyse_lfa(
            image,
            quantile_low=0.03,
            quantile_high=0.97,
            smooth_window=15,
            als_lambda=10.0
        )
        
        assert results['metadata']['quantile_low'] == 0.03
        assert results['metadata']['quantile_high'] == 0.97
        assert results['metadata']['smooth_window'] == 15
        assert results['metadata']['als_lambda'] == 10.0
    
    def test_analyse_lfa_dimensions(self):
        """Test dimension validation."""
        # Too small
        small_image = np.ones((10, 30))
        
        with pytest.raises(ValueError):
            analyse_lfa(small_image)
        
        # Wrong dimensions (3D)
        wrong_dims = np.ones((50, 200, 3))
        
        with pytest.raises(ValueError):
            analyse_lfa(wrong_dims)


class TestBaselineALS:
    """Test ALS baseline correction."""
    
    def test_baseline_als_basic(self):
        """Test basic ALS functionality."""
        # Create signal with baseline drift
        x = np.linspace(0, 10, 100)
        signal = np.sin(x) + 0.1 * x  # Sine wave + linear drift
        
        corrected = baseline_als(signal, lam=100, p=0.01, niter=10)
        
        # Corrected signal should have same length
        assert len(corrected) == len(signal)
        
        # Should be finite
        assert np.all(np.isfinite(corrected))
    
    def test_baseline_als_removes_drift(self):
        """Test that ALS removes baseline drift."""
        # Pure baseline (should be flattened)
        baseline = np.linspace(0, 1, 100)
        
        corrected = baseline_als(baseline, lam=1000, p=0.01, niter=20)
        
        # Should be near zero (within numerical precision)
        assert np.abs(corrected).mean() < 0.1


class TestArtifactRemoval:
    """Test artifact removal function."""
    
    def test_remove_artifacts_basic(self):
        """Test basic artifact removal."""
        image = np.random.rand(50, 200)
        
        # Add some outliers
        image[10, 50] = 10.0  # High outlier
        image[20, 100] = -5.0  # Low outlier
        
        filtered = _remove_artifacts(image, 0.05, 0.95)
        
        # Should have NaNs where artifacts were
        assert np.sum(np.isnan(filtered)) > 0
        
        # Should maintain shape
        assert filtered.shape == image.shape
    
    def test_remove_artifacts_preserves_valid(self):
        """Test that valid values are preserved."""
        # Uniform image (no outliers)
        image = np.ones((50, 200)) * 0.5
        
        filtered = _remove_artifacts(image, 0.05, 0.95)
        
        # Should have minimal NaNs
        assert np.sum(np.isnan(filtered)) < (image.size * 0.2)


class TestMovingAverage:
    """Test moving average function."""
    
    def test_moving_average_basic(self):
        """Test basic smoothing."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        smoothed = _moving_average(signal, window=3)
        
        # Should have same length
        assert len(smoothed) == len(signal)
        
        # Should be smoother (lower variance)
        assert np.var(smoothed) < np.var(signal)
    
    def test_moving_average_preserves_mean(self):
        """Test that smoothing preserves approximate mean."""
        signal = np.random.rand(100)
        
        smoothed = _moving_average(signal, window=5)
        
        # Mean should be approximately preserved
        assert np.abs(np.mean(smoothed) - np.mean(signal)) < 0.1


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_synthetic(self):
        """Test complete analysis pipeline with synthetic data."""
        # Create realistic synthetic LFA image
        width = 300
        height = 80
        
        # Start with uniform background
        image = np.ones((height, width)) * 0.4
        
        # Add Gaussian noise
        image += np.random.normal(0, 0.05, (height, width))
        
        # Add test peak (Gaussian)
        x = np.arange(width)
        test_peak = 0.3 * np.exp(-((x - 80) ** 2) / (2 * 10 ** 2))
        image -= test_peak[np.newaxis, :]  # Subtract = darker
        
        # Add control peak
        control_peak = 0.25 * np.exp(-((x - 220) ** 2) / (2 * 10 ** 2))
        image -= control_peak[np.newaxis, :]
        
        # Clip to valid range
        image = np.clip(image, 0, 1)
        
        # Analyse
        results = analyse_lfa(image)
        
        # Verify reasonable results
        assert 0 < results['test_peak'] < 10
        assert 0 < results['control_peak'] < 10
        assert 0.5 < results['ratio'] < 2.0
        
        # Verify test peak is stronger (we made it slightly larger)
        assert results['ratio'] > 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])