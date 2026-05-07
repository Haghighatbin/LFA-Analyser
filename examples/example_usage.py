#!/usr/bin/env python3
"""
Example Usage of LFA Analyser

Demonstrates various ways to use the LFA Analyser package.
"""

import numpy as np
import matplotlib.pyplot as plt

from lfa_analyser import (
    analyse_lfa,
    load_image,
    validate_lfa_image,
    batch_load_images,
    export_results_to_csv,
    create_intensity_plot,
    print_results_summary
)


def create_synthetic_lfa():
    """
    Create a synthetic LFA image for demonstration.
    
    Returns synthetic image with two peaks (test and control).
    """
    width = 400
    height = 100
    
    # Base image
    image = np.ones((height, width)) * 0.4
    
    # Add noise
    image += np.random.normal(0, 0.03, (height, width))
    
    # Create peaks using Gaussian functions
    x = np.arange(width)
    
    # Test peak (left side, stronger)
    test_peak = 0.35 * np.exp(-((x - 120) ** 2) / (2 * 15 ** 2))
    
    # Control peak (right side, weaker)
    control_peak = 0.28 * np.exp(-((x - 280) ** 2) / (2 * 15 ** 2))
    
    # Add peaks to image (darker = higher intensity after inversion)
    image -= test_peak[np.newaxis, :]
    image -= control_peak[np.newaxis, :]
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image


def example_1_basic_analysis():
    """Example 1: Basic single image analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Single Image Analysis")
    print("="*70)
    
    # Create synthetic image
    image = create_synthetic_lfa()
    print("\n✓ Synthetic LFA image created")
    
    # Run analysis
    results = analyse_lfa(image)
    print("✓ Analysis completed")
    
    # Print results
    print_results_summary(results)
    
    # Visualise
    create_intensity_plot(results, title="Example 1: Basic Analysis", show=False)
    plt.savefig('example_1_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'example_1_plot.png'")
    plt.close()


def example_2_custom_parameters():
    """Example 2: Analysis with custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Parameters")
    print("="*70)
    
    image = create_synthetic_lfa()
    
    # Analyse with custom parameters
    results = analyse_lfa(
        image,
        quantile_low=0.03,       # More aggressive artifact removal
        quantile_high=0.97,
        smooth_window=15,        # Stronger smoothing
        als_lambda=10.0,         # Smoother baseline
        als_p=0.005              # More asymmetric
    )
    
    print("\n✓ Custom parameters applied:")
    print(f"  - Quantile range: 0.03 - 0.97")
    print(f"  - Smoothing window: 15 pixels")
    print(f"  - ALS lambda: 10.0")
    
    print(f"\nResults:")
    print(f"  Test Peak: {results['test_peak']:.4f}")
    print(f"  Control Peak: {results['control_peak']:.4f}")
    print(f"  T/C Ratio: {results['ratio']:.4f}")


def example_3_comparison():
    """Example 3: Compare default vs custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Parameter Comparison")
    print("="*70)
    
    image = create_synthetic_lfa()
    
    # Default parameters
    results_default = analyse_lfa(image)
    
    # Custom parameters
    results_custom = analyse_lfa(
        image,
        smooth_window=20,
        als_lambda=20.0
    )
    
    print("\n📊 Comparison:")
    print(f"\n  Default Parameters:")
    print(f"    Ratio: {results_default['ratio']:.4f}")
    
    print(f"\n  Custom Parameters (heavy smoothing):")
    print(f"    Ratio: {results_custom['ratio']:.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(results_default['intensity_profile'], 'b-', linewidth=2)
    ax1.set_title('Default Parameters')
    ax1.set_ylabel('Intensity')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(results_custom['intensity_profile'], 'r-', linewidth=2)
    ax2.set_title('Custom Parameters (heavy smoothing)')
    ax2.set_xlabel('Pixel Position')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_3_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved as 'example_3_comparison.png'")
    plt.close()


def example_4_batch_processing():
    """Example 4: Batch processing simulation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    # Create multiple synthetic images
    n_images = 5
    results_dict = {}
    
    print(f"\nGenerating {n_images} synthetic images...")
    
    for i in range(n_images):
        # Create image with slight variations
        image = create_synthetic_lfa()
        image += np.random.normal(0, 0.02, image.shape)  # Add variation
        
        filename = f"synthetic_strip_{i+1}.jpg"
        results_dict[filename] = analyse_lfa(image)
        
        print(f"  ✓ {filename}: Ratio = {results_dict[filename]['ratio']:.4f}")
    
    # Create summary
    from lfa_analyser.utils import create_batch_summary
    
    summary_df = create_batch_summary(results_dict, output_path='batch_summary.csv')
    
    print("\n📊 Batch Summary:")
    print(summary_df.to_string(index=False))
    print("\n✓ Summary saved as 'batch_summary.csv'")


def example_5_quality_validation():
    """Example 5: Image quality validation."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Image Quality Validation")
    print("="*70)
    
    # Good quality image
    good_image = create_synthetic_lfa()
    
    # Poor quality images
    poor_contrast = np.ones((100, 400)) * 0.5  # No contrast
    poor_contrast += np.random.normal(0, 0.01, poor_contrast.shape)
    
    overexposed = np.ones((100, 400)) * 0.95  # Overexposed
    
    print("\n1. Good Quality Image:")
    validate_lfa_image(good_image)
    
    print("\n2. Poor Contrast Image:")
    validate_lfa_image(poor_contrast)
    
    print("\n3. Overexposed Image:")
    validate_lfa_image(overexposed)


def example_6_export_formats():
    """Example 6: Different export formats."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Formats")
    print("="*70)
    
    image = create_synthetic_lfa()
    results = analyse_lfa(image)
    
    # Export to CSV
    export_results_to_csv(results, 'results_example.csv')
    print("\n✓ Exported to CSV: results_example.csv")
    
    # Export to JSON
    from lfa_analyser.utils import export_results_to_json
    export_results_to_json(results, 'results_example.json', include_profiles=True)
    print("✓ Exported to JSON: results_example.json")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LFA ANALYSER - EXAMPLE DEMONSTRATIONS")
    print("="*70)
    
    try:
        example_1_basic_analysis()
        example_2_custom_parameters()
        example_3_comparison()
        example_4_batch_processing()
        example_5_quality_validation()
        example_6_export_formats()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  - example_1_plot.png")
        print("  - example_3_comparison.png")
        print("  - batch_summary.csv")
        print("  - results_example.csv")
        print("  - results_example.json")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()