"""
Image Preprocessing Module

Handles image loading, format conversion, and validation.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple
from skimage import io, color
import warnings
from scipy import ndimage
from lfa_analyser.config import Config


def load_image(
    filepath: Union[str, Path], 
    target_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Load and preprocess LFA image.
    
    Parameters
    ----------
    filepath : str or Path
        Path to image file
    target_size : tuple of int, optional
        Resize to (height, width) if specified
    
    Returns
    -------
    np.ndarray
        Grayscale image as 2D numpy array with float64 dtype
    
    Raises
    ------
    FileNotFoundError
        If image file doesn't exist
    ValueError
        If image format is unsupported or corrupted
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Image not found: {filepath}")
    
    try:
        # Load image
        image = io.imread(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load image {filepath}: {e}")
    
    # Convert to grayscale if needed
    if image.ndim == 3:
        if image.shape[2] == 4:  # RGBA
            # Remove alpha channel
            image = image[:, :, :3]
        
        # Convert to grayscale
        image = color.rgb2gray(image)
    elif image.ndim > 3:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}D")
    
    # Ensure float64 for numerical stability
    if image.dtype != np.float64:
        image = image.astype(np.float64)
    
    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Resize if requested
    if target_size is not None:
        from skimage.transform import resize
        image = resize(image, target_size, anti_aliasing=True)
    
    # Validate minimum size
    if image.shape[0] < 50 or image.shape[1] < 50:
        warnings.warn(
            f"Image is too small ({image.shape}), results may be unreliable"
        )
    
    return image


def validate_lfa_image(image: np.ndarray) -> bool:
    """
    Perform basic quality checks on LFA image.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image array
    
    Returns
    -------
    bool
        True if image passes quality checks
    
    Warnings
    --------
    Issues warnings for potential quality problems.
    """
    checks_passed = True
    
    # Check dimensions
    height, width = image.shape
    if height > width:
        warnings.warn(
            f"Image is taller than wide ({height}x{width}). "
            "LFA strips are typically wider than tall. "
            "Consider rotating the image."
        )
        checks_passed = False
    
    # Check contrast
    intensity_range = image.max() - image.min()
    if intensity_range < 0.2:
        warnings.warn(
            f"Low contrast detected (range: {intensity_range:.3f}). "
            "Image may be overexposed or underexposed."
        )
        checks_passed = False
    
    # Check for saturation
    if np.sum(image >= 0.99) > (image.size * 0.05):
        warnings.warn(
            "More than 5% of pixels are saturated (white). "
            "Image may be overexposed."
        )
        checks_passed = False
    
    if np.sum(image <= 0.01) > (image.size * 0.05):
        warnings.warn(
            "More than 5% of pixels are near-black. "
            "Image may be underexposed."
        )
        checks_passed = False
    
    # Check for blur (using Laplacian variance)
    try:
        laplacian = ndimage.laplace(image)
        blur_metric = laplacian.var()
        
        if blur_metric < Config.BLUR_METRIC:
            warnings.warn(
                f"Image may be out of focus (blur metric: {blur_metric:.6f})"
            )
            checks_passed = False
    except ImportError:
        pass  # Skip blur check if scipy not available
    
    return checks_passed


def auto_rotate_if_needed(image: np.ndarray) -> np.ndarray:
    """
    Automatically rotate image if it appears to be oriented incorrectly.
    
    LFA strips should be wider than tall. If image is taller than wide,
    rotate 90 degrees.
    
    Parameters
    ----------
    image : np.ndarray
        Input grayscale image
    
    Returns
    -------
    np.ndarray
        Potentially rotated image
    """
    height, width = image.shape
    
    if height > width:
        warnings.warn("Auto-rotating image 90° (image was taller than wide)")
        return np.rot90(image, k=-1)  # Rotate 90° clockwise
    
    return image


def batch_load_images(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.tif')
) -> dict:
    """
    Load all LFA images from a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing images
    extensions : tuple of str
        File extensions to include
    
    Returns
    -------
    dict
        Dictionary mapping filename to image array
    
    Raises
    ------
    FileNotFoundError
        If directory doesn't exist
    ValueError
        If no valid images found
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    images = {}
    
    for ext in extensions:
        for filepath in directory.glob(f'*{ext}'):
            try:
                images[filepath.name] = load_image(filepath)
            except Exception as e:
                warnings.warn(f"Failed to load {filepath.name}: {e}")
                continue
    
    if not images:
        raise ValueError(
            f"No valid images found in {directory} "
            f"with extensions {extensions}"
        )
    
    return images