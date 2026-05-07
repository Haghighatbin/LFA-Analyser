#!/usr/bin/env python3
"""
LFA Analyser - Command Line Interface

Process LFA images via command line with batch processing support.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict
import warnings

from lfa_analyser.config import Config
from lfa_analyser.core import analyse_lfa
from lfa_analyser.preprocessing import load_image, validate_lfa_image, batch_load_images
from lfa_analyser.utils import (
    export_results_to_csv,
    export_results_to_json,
    create_intensity_plot,
    print_results_summary,
    create_batch_summary
)


def process_single_image(
    input_path: Path,
    args: argparse.Namespace
) -> Dict:
    """Process a single LFA image."""
    
    print(f"\nProcessing: {input_path.name}")
    print("-" * 60)
    
    # Load image
    try:
        image = load_image(input_path)
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None
    
    # Validate image quality
    if args.validate:
        validate_lfa_image(image)
    
    # Run analysis
    try:
        results = analyse_lfa(
            image,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
            smooth_window=args.smooth_window,
            baseline_region_size=args.baseline_region,
            als_lambda=args.als_lambda,
            als_p=args.als_p,
            als_niter=args.als_niter
        )
        print("✓ Analysis complete")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return None
    
    # Print summary
    if args.verbose:
        print_results_summary(results, input_path.name)
    else:
        print(f"\nTL1 Peak: {results['TL1_peak']:.4f}")
        print(f"TL2 Peak: {results['TL2_peak']:.4f}")
        print(f"TL1/TL2 Peak Ratio: {results['ratio']:.4f}")
        print(f"TL1 AUC: {results['TL1_auc']:.2f}")
        print(f"TL2 AUC: {results['TL2_auc']:.2f}")
        print(f"TL1/TL2 AUC Ratio: {results['auc_ratio']:.4f}")
        
    return results

def process_batch(
    input_dir: Path,
    args: argparse.Namespace
) -> Dict[str, Dict]:
    """Process all images in a directory."""
    
    print(f"\nBatch Processing: {input_dir}")
    print("=" * 60)
    
    # Load all images
    try:
        images = batch_load_images(input_dir)
        print(f"Found {len(images)} images")
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)
    
    # Process each image
    results = {}
    failed = []
    
    for i, (filename, image) in enumerate(images.items(), 1):
        print(f"\n[{i}/{len(images)}] Processing: {filename}")
        
        try:
            result = analyse_lfa(
                image,
                quantile_low=args.quantile_low,
                quantile_high=args.quantile_high,
                smooth_window=args.smooth_window,
                baseline_region_size=args.baseline_region,
                als_lambda=args.als_lambda,
                als_p=args.als_p,
                als_niter=args.als_niter
            )
            results[filename] = result
            print(f"  ✓ Ratio: {result['ratio']:.4f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(filename)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Completed: {len(results)}/{len(images)} images")
    if failed:
        print(f"Failed: {len(failed)} images")
        print("  " + ", ".join(failed))
    print("=" * 60)
    
    return results

def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="LFA Analyser - Lateral Flow Assay Image Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse single image
  lfa-analyse -i test.jpg
  
  # Batch process directory
  lfa-analyse -i ./images/ -o results.csv --plot
  
  # Custom parameters with static PNG plots
  lfa-analyse -i test.jpg --quantile-low 0.03 --smooth-window 15 --plot --plot-format png
  
  # Save detailed results
  lfa-analyse -i test.jpg -o results.json --format json --plot
        """
    )
    
    # Input/Output
    parser.add_argument(
        '-i', '--input',
        required=True,
        type=Path,
        help='Input image file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output file (default: results.csv for batch, no file for single)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    # Analysis Parameters
    params = parser.add_argument_group('Analysis Parameters')
    params.add_argument(
        '--quantile-low',
        type=float,
        default=Config.QUANTILE_LOW_VAL,
        help='Lower quantile for artifact removal (default: 0.05)'
    )
    params.add_argument(
        '--quantile-high',
        type=float,
        default=Config.QUANTILE_HIGH_VAL,
        help='Upper quantile for artifact removal (default: 0.95)'
    )
    params.add_argument(
        '--smooth-window',
        type=int,
        default=Config.SMOOTH_WIN_VAL,
        help='Moving average window size (default: 10)'
    )
    params.add_argument(
        '--baseline-region',
        type=int,
        default=Config.BASELINE_REGION_VAL,
        help='Baseline region size in pixels (default: 20)'
    )
    params.add_argument(
        '--als-lambda',
        type=float,
        default=Config.ALS_LAMBDA_VAL,
        help='ALS smoothness parameter (default: 2.5)'
    )
    params.add_argument(
        '--als-p',
        type=float,
        default=Config.ALS_P_SYM_VAL,
        help='ALS asymmetry parameter (default: 0.01)'
    )
    params.add_argument(
        '--als-niter',
        type=int,
        default=Config.ALS_N_ITER_STEP,
        help='ALS iteration count (default: 20)'
    )
    
    # Options
    options = parser.add_argument_group('Options')
    options.add_argument(
        '--plot',
        action='store_true',
        help='Generate intensity profile plots'
    )
    options.add_argument(
        '--plot-format',
        choices=['html', 'png', 'svg', 'pdf'],
        default='html',
        help='Plot output format (default: html for interactive plots)'
    )
    options.add_argument(
        '--validate',
        action='store_true',
        help='Perform image quality validation'
    )
    options.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    options.add_argument(
        '--no-warnings',
        action='store_true',
        help='Suppress warning messages'
    )
    
    args = parser.parse_args()
    
    # Suppress warnings if requested
    if args.no_warnings:
        warnings.filterwarnings('ignore')
    
    # Validate input path
    if not args.input.exists():
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
    
    # Determine if single file or batch
    is_batch = args.input.is_dir()
    
    # Process
    if is_batch:
        # Batch processing
        results = process_batch(args.input, args)
        
        # Export results
        if args.output:
            output_path = args.output
        else:
            output_path = Path('results.csv')
        
        if args.format == 'csv':
            summary = create_batch_summary(results, output_path)
            print(f"\n✓ Results saved to: {output_path}")
        else:
            export_results_to_json(list(results.values()), output_path)

        # Create plots if requested
        if args.plot:
            plot_dir = Path('plots')
            plot_dir.mkdir(exist_ok=True)
            
            print(f"\nGenerating {args.plot_format.upper()} plots...")
            
            for filename, result in results.items():
                plot_path = plot_dir / f"{Path(filename).stem}_plot.{args.plot_format}"
                create_intensity_plot(
                    result, 
                    title=filename, 
                    save_path=plot_path, 
                    show=False
                )
            
            if args.plot_format == 'html':
                print(f"✓ Interactive plots saved to: {plot_dir}/")
                print(f"  Open .html files in browser for full interactivity")
            else:
                print(f"✓ Static {args.plot_format.upper()} plots saved to: {plot_dir}/")

    else:
        # Single file processing
        results = process_single_image(args.input, args)
        
        if results is None:
            sys.exit(1)
        
        # Export if output specified
        if args.output:
            if args.format == 'csv':
                export_results_to_csv(results, args.output)
            else:
                export_results_to_json(results, args.output)
        
        # Create plot if requested
        if args.plot:
            # plot_path = args.input.stem + '_plot.png'
            plot_path = Path(f"{args.input.stem}_plot.{args.plot_format}")

            create_intensity_plot(results, save_path=plot_path, show=True)

            if args.plot_format == 'html':
                print(f"  Interactive plot will open in browser")

    print("\n✓ Analysis complete!\n")


if __name__ == '__main__':
    main()