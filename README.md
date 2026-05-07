# LFA analyser

**Lateral Flow Assay Image Analysis Tool**

LFA Analyser is a Python package for the quantitative analysis of lateral flow assay (LFA) test strip images. It supports both single-image processing and batch analysis, and can be used via a command-line interface or a web-based interface.

This project is a Python reimplementation and extension of the original methodology developed in the R-based LFA_CAMO project (https://github.com/vvasikasin/LFA_CAMO).

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
<!-- [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) -->

## Features

**Core Capabilities:**
- Automated peak detection for test and control lines
- Quantitative test/control ratio calculation
- Baseline correction using Asymmetric Least Squares (ALS)
- Robust artifact removal and signal smoothing
- Image quality validation

**Dual Interface:**
- **CLI**: Command-line tool for batch processing and automation
- **Web App**: Interactive Streamlit interface for easy visualisation

**Analysis Pipeline:**
1. Artifact removal via quantile filtering
2. Piecewise linear baseline correction
3. Signal inversion and smoothing
4. ALS baseline subtraction
5. Peak detection and quantification

## Installation

### From Source (Recommended for Development)

```bash
# Clone repository
git clone https://github.com/yourusername/lfa-analyser.git
cd lfa-analyser

# Install in development mode
pip install -e .
```

### Using pip (Once Published)

```bash
pip install lfa-analyser
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- scikit-image ≥ 0.20.0
- Matplotlib ≥ 3.7.0
- Pandas ≥ 2.0.0
- Streamlit ≥ 1.28.0

## Quick Start

### Command-Line Interface

**Analyse a single image:**
```bash
lfa-analyse -i test_strip.jpg
```

**Batch process a directory:**
```bash
lfa-analyse -i ./images/ -o results.csv
```

**Custom parameters with plotting:**
```bash
lfa-analyse -i test.jpg \
    --quantile-low 0.03 \
    --quantile-high 0.97 \
    --smooth-window 15 \
    --plot
```

**Save results in JSON format:**
```bash
lfa-analyse -i test.jpg -o results.json --format json
```

### Web Application

Launch the Streamlit interface:

```bash
streamlit run lfa_analyser/app.py
```

Or if installed via pip:
```bash
lfa-streamlit
```

Then navigate to `http://localhost:8501` in your browser.

### Python API

```python
from lfa_analyser import analyse_lfa, load_image

# Load image
image = load_image('test_strip.jpg')

# Run analysis
results = analyse_lfa(image)

# Access results
print(f"Test Peak: {results['test_peak']:.4f}")
print(f"Control Peak: {results['control_peak']:.4f}")
print(f"T/C Ratio: {results['ratio']:.4f}")
```

## Usage Examples

### 1. Basic Analysis

```python
from lfa_analyser import analyse_lfa, load_image, print_results_summary

# Load and analyse
image = load_image('sample.jpg')
results = analyse_lfa(image)

# Print formatted summary
print_results_summary(results, 'sample.jpg')
```

Output:
```
============================================================
Analysis Results: sample.jpg
============================================================

Test Peak Intensity:    0.8234
Control Peak Intensity: 0.7156
Test/Control Ratio:     1.1506

Image Size: 1200 x 400 pixels

Analysis Parameters:
  Quantile Range:     0.05 - 0.95
  Smoothing Window:   10 pixels
  ALS Lambda:         5.0
  ALS p:              0.01
============================================================
```

### 2. Batch Processing

```python
from lfa_analyser import batch_load_images, analyse_lfa, create_batch_summary

# Load all images from directory
images = batch_load_images('./data/lfa_images/')

# Process each image
results = {}
for filename, image in images.items():
    results[filename] = analyse_lfa(image)

# Create summary table
summary = create_batch_summary(results, output_path='batch_results.csv')
print(summary)
```

### 3. Custom Parameters

```python
# Fine-tune analysis parameters
results = analyse_lfa(
    image,
    quantile_low=0.03,      # More aggressive low artifact removal
    quantile_high=0.97,     # More aggressive high artifact removal
    smooth_window=15,       # Stronger smoothing
    als_lambda=10.0,        # Smoother baseline
    als_p=0.005             # More asymmetric baseline fitting
)
```

### 4. Visualisation

```python
from lfa_analyser import create_intensity_plot

# Create and save plot
create_intensity_plot(
    results,
    title="LFA Strip Analysis",
    save_path="analysis_plot.png",
    show=True
)
```

### 5. Export Results

```python
from lfa_analyser import export_results_to_csv, export_results_to_json

# Export to CSV
export_results_to_csv(results, 'results.csv')

# Export to JSON (with intensity profiles)
export_results_to_json(results, 'results.json', include_profiles=True)
```

## CLI Options

### Input/Output

| Option | Description |
|--------|-------------|
| `-i, --input` | Input image file or directory (required) |
| `-o, --output` | Output file path |
| `--format` | Output format: `csv` or `json` (default: csv) |

### Analysis Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--quantile-low` | 0.05 | Lower quantile for artifact removal |
| `--quantile-high` | 0.95 | Upper quantile for artifact removal |
| `--smooth-window` | 10 | Moving average window size (pixels) |
| `--baseline-region` | 20 | Baseline region size (pixels) |
| `--als-lambda` | 5.0 | ALS smoothness parameter |
| `--als-p` | 0.01 | ALS asymmetry parameter |
| `--als-niter` | 20 | ALS iteration count |

### Options

| Flag | Description |
|------|-------------|
| `--plot` | Generate intensity profile plots |
| `--validate` | Perform image quality validation |
| `-v, --verbose` | Verbose output |
| `--no-warnings` | Suppress warning messages |

## Analysis Methodology

The LFA analyser implements a multi-stage image processing pipeline:

### 1. Preprocessing
- Convert to grayscale
- Normalise intensity values
- Validate image quality (optional)

### 2. Artifact Removal
- Apply per-column quantile filtering
- Remove outliers (default: 5th-95th percentile)
- Preserve genuine signal while removing noise

### 3. Baseline Correction
- Extract anchor points (first, middle, last regions)
- Apply piecewise linear correction
- Compensate for uneven illumination

### 4. Signal Processing
- Invert signal (LFA peaks appear dark)
- Apply moving average smoothing
- Reduce high-frequency noise

### 5. ALS Baseline Subtraction
- Fit asymmetric baseline using sparse matrix methods
- Remove low-frequency background drift
- Preserve peak structure

### 6. Peak Detection
- Divide strip into test and control halves
- Identify maximum intensity in each region
- Calculate test/control ratio

## Output Format

### Results Dictionary

```python
{
    'test_peak': 0.8234,           # Maximum intensity in test region
    'control_peak': 0.7156,        # Maximum intensity in control region
    'ratio': 1.1506,               # Test/Control ratio
    'intensity_profile': array,    # Corrected intensity array
    'raw_profile': array,          # Pre-correction profile
    'metadata': {
        'image_size': (400, 1200),
        'quantile_low': 0.05,
        'quantile_high': 0.95,
        'smooth_window': 10,
        # ... other parameters
    }
}
```

### CSV Export Format

```csv
image_index,test_peak,control_peak,ratio,image_width,image_height
0,0.8234,0.7156,1.1506,1200,400
```

## Image Requirements

**Optimal Image Characteristics:**
- Format: JPEG, PNG, or TIFF
- Orientation: Horizontal (wider than tall)
- Resolution: ≥ 800 pixels width recommended
- Lighting: Even illumination, minimal shadows
- Focus: Sharp focus on test strips
- Background: Uniform, contrasting with strip

**Common Issues:**
- ❌ Vertical orientation → Use `auto_rotate_if_needed()`
- ❌ Poor lighting → Adjust quantile thresholds
- ❌ Out of focus → May require manual preprocessing
- ❌ Multiple strips → Crop to single strip per image

## Advanced Usage

### Image Quality Validation

```python
from lfa_analyser import validate_lfa_image

image = load_image('test.jpg')
is_valid = validate_lfa_image(image)

# Provides warnings for:
# - Incorrect orientation
# - Low contrast
# - Over/underexposure
# - Blur detection
```

### Custom Baseline Correction

```python
from lfa_analyser.core import baseline_als
import numpy as np

# Apply ALS to custom signal
signal = np.array([...])  # Your 1D signal
corrected = baseline_als(signal, lam=100, p=0.001, niter=30)
```

### Integration with Image Processing Pipeline

```python
from lfa_analyser import load_image, analyse_lfa
from skimage import filters, exposure

# Custom preprocessing
image = load_image('test.jpg')
image = filters.gaussian(image, sigma=1)  # Denoise
image = exposure.rescale_intensity(image)  # Enhance contrast

# Then analyse
results = analyse_lfa(image)
```

## Troubleshooting

### Common Issues

**1. Low Test/Control Ratio**
- Check image orientation (test line should be on left)
- Verify strip is properly illuminated
- Try adjusting `quantile_low` and `quantile_high`

**2. Analysis Fails**
- Ensure image is in supported format
- Check minimum image dimensions (≥ 60 pixels recommended)
- Validate image quality with `validate_lfa_image()`

**3. Inconsistent Results**
- Use consistent image capture conditions
- Standardise lighting and distance
- Consider batch processing with same parameters

**4. Poor Peak Detection**
- Adjust `smooth_window` (increase for noisy images)
- Tune ALS parameters (`als_lambda`, `als_p`)
- Verify strip positioning in image

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black lfa_analyser/
flake8 lfa_analyser/
```

### Project Structure

```
lfa-analyser/
├── lfa_analyser/
│   ├── __init__.py       # Package initialisation
│   ├── core.py           # Core analysis algorithms
│   ├── preprocessing.py  # Image loading and validation
│   ├── utils.py          # Export and visualisation
│   ├── cli.py            # Command-line interface
│   └── app.py            # Streamlit web application
├── tests/
│   ├── test_core.py
│   ├── test_preprocessing.py
│   └── fixtures/
├── examples/
│   └── sample_images/
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Acknowledgments

- Original R methodology by Vasin Vasikasin
- ALS baseline correction algorithm: Eilers & Boelens (2005)
- Built with: NumPy, SciPy, scikit-image, Streamlit

## Contact

## License

[TBD]

## Changelog

### Version 1.0.0 (2024)
- Initial release
- Core analysis pipeline
- CLI and Streamlit interfaces
- Batch processing support
- CSV/JSON export functionality