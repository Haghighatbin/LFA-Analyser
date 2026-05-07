#!/usr/bin/env python3
"""
LFA Analyser - Streamlit Web Application

Interactive web interface for LFA image analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go

from .core import analyse_lfa
from .preprocessing import load_image, validate_lfa_image
from .utils import format_results_table, create_intensity_plot

from lfa_analyser.config import Config


# Page configuration
st.set_page_config(
    page_title="LFA Image Analyser",
    # page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",

)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: bold !important;
        color: #2E86AB !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
        font-family: "Courier New" !important;
    }
        
    /* Universal font for all elements */
    html, body, [class*="css"] {
        font-family: "Courier New", monospace;
    }
            
    /* Explicitly target headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: "Courier New", monospace !important;
    }
    
    /* Targetting Streamlit elements */
    .stApp, .stSidebar, .stMarkdown, .stButton, .stTextInput, stInfo, stImage,
    .stSelectbox, .stMultiSelect, .stNumberInput, .stSlider,
    div[data-testid="stMarkdownContainer"], 
    div[data-testid="stText"],
    .element-container {
        font-family: "Courier New", monospace;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family:  "Courier New";
    }
    .stAlert {
        margin-top: 1rem;
        font-family:  "Courier New";
    }
</style>
""", unsafe_allow_html=True)


def load_image_from_upload(uploaded_file):
    """Load image from Streamlit upload widget."""
    try:
        # Read as PIL Image
        pil_image = Image.open(uploaded_file)
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Convert to grayscale if needed
        if img_array.ndim == 3:
            from skimage import color
            if img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            img_array = color.rgb2gray(img_array)
        
        # Normalise to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        return img_array.astype(np.float64)
    
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def export_to_csv_string(results):
    """Convert results to CSV string."""
    data = {
        'TL1 Peak': [results['TL1_peak']],
        'TL2 Peak': [results['TL2_peak']],
        'TL1/TL2 Peak Ratio': [results['ratio']],
        'TL1 AUC': [results['TL1_auc']],
        'TL2 AUC': [results['TL2_auc']],
        'TL1/TL2 AUC Ratio': [results['auc_ratio']],
        'Image Width': [results['metadata']['image_size'][1]],
        'Image Height': [results['metadata']['image_size'][0]]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# Main app
def main():
    # Header
    st.markdown('<p class="main-header"> LFA Image Analser</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Lateral Flow Assay Image Analysis Tool - v0.01</br>"
        "</p>", 
        unsafe_allow_html=True
    )
    
    # Sidebar - Parameters
    with st.sidebar:
        st.header("⚙️ Analysis Parameters")
        
        st.subheader("Artifact Removal")
        quantile_low = st.slider(
            "Lower Quantile",
            min_value=  Config.QUANTILE_LOW_MIN_VAL,
            max_value=  Config.QUANTILE_LOW_MAX_VAL,
            value=      Config.QUANTILE_LOW_VAL,
            step=       Config.QUANTILE_LOW_STEP,
            help="Threshold for removing low-intensity artifacts"
        )
        
        quantile_high = st.slider(
            "Upper Quantile",
            min_value=  Config.QUANTILE_HIGH_MIN_VAL,
            max_value=  Config.QUANTILE_HIGH_MAX_VAL,
            value=      Config.QUANTILE_HIGH_VAL,
            step=       Config.QUANTILE_HIGH_STEP,
            help="Threshold for removing high-intensity artifacts"
        )
        st.subheader("Signal Processing")
        smooth_window = st.number_input(
            "Smoothing Window",
            min_value=  Config.SMOOTH_WIN_MIN_VAL,
            max_value=  Config.SMOOTH_WIN_MAX_VAL,
            value=      Config.SMOOTH_WIN_VAL,
            step=       Config.SMOOTH_WIN_STEP,
            help="Moving average window size (pixels)"
        )
        
        baseline_region = st.number_input(
            "Baseline Region Size",
            min_value=  Config.BASELINE_REGION_MIN_VAL,
            max_value=  Config.BASELINE_REGION_MAX_VAL,
            value=      Config.BASELINE_REGION_VAL,
            step=       Config.BASELINE_REGION_STEP,
            help="Size of regions used for baseline estimation"
        )

        auc_window = st.number_input(
            "AUC Window (±pixels)",
            min_value=  Config.AUC_WIN_MIN_VAL,
            max_value=  Config.AUC_WIN_MAX_VAL,
            value=      Config.AUC_WIN_VAL,
            step=       Config.AUC_WIN_STEP,
            help="Number of pixels either side of peak to include in AUC calculation"
        )
        
        st.subheader("ALS Baseline Correction")
        als_lambda = st.number_input(
            "Lambda (Smoothness)",
            min_value=  Config.ALS_LAMBDA_MIN_VAL,
            max_value=  Config.ALS_LAMBDA_MAX_VAL,
            value=      Config.ALS_LAMBDA_VAL,
            step=       Config.ALS_LAMBDA_STEP,
            format=     Config.ALS_LAMBDA_FORMAT,
            help="Larger values = smoother baseline"
        )
        
        als_p = st.number_input(
            "p (Asymmetry)",
            min_value=  Config.ALS_P_SYM_MIN_VAL,
            max_value=  Config.ALS_P_SYM_MAX_VAL,
            value=      Config.ALS_P_SYM_VAL,
            step=       Config.ALS_P_SYM_STEP,
            format=     Config.ALS_P_SYM_FORMAT,
            help="Smaller values = more asymmetric"
        )
        
        als_niter = st.number_input(
            "Iterations",
            min_value=  Config.ALS_N_ITER_MIN_VAL,
            max_value=  Config.ALS_N_ITER_MAX_VAL,
            value=      Config.ALS_N_ITER_VAL,
            step=       Config.ALS_N_ITER_STEP,
            help="Number of ALS iterations"
        )
        
        # Options
        st.divider()
        st.subheader("Display Options")
        validate_quality = st.checkbox("Validate Image Quality", value=True)
        show_peaks = st.checkbox("Show Peak Markers", value=True)
        show_raw_overlay = st.checkbox("Show Raw Profile Overlay", value=False)

        # Reset button
        if st.button("Reset to Defaults", width='stretch'):
            st.rerun()
    
    # Main content
    st.divider()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload LFA Image(s)",
        type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
        accept_multiple_files=True,
        help="Upload one or more LFA strip images for analysis"
    )
    
    if not uploaded_files:
        # Instructions
        st.info("""
        ### 📋 Instructions
        
        1. **Upload Image(s)**: Click 'Browse files' above to select LFA strip images
        2. **Adjust Parameters**: Use the sidebar to fine-tune analysis settings
        3. **Review Results**: View quantitative metrics and intensity profiles
        4. **Export Data**: Download results as CSV for further analysis
        
        #### 💡 Tips
        - Images should show LFA strips horizontally (wider than tall)
        - Ensure good lighting and focus for best results
        - The test-line-1 (TL1) should appear in the left half, test-line-2 (TL2) in the right half
        """)
        
        # Example image placeholder
        st.markdown("### 📸 Example LFA Strip")
        example_img_path = "lfa_analyser/assets/Picture3.png"
        try:
            st.image(example_img_path,
                    caption="Example of a properly oriented LFA strip image")
        except FileNotFoundError:
                st.warning("Example image not found. Please add an example LFA strip image to `assets/example_lfa.png`")
            
    else:
        # Process each uploaded image
        for uploaded_file in uploaded_files:
            st.divider()
            st.subheader(f"📊 Analysis: {uploaded_file.name}")
            
            # Load image
            with st.spinner("Loading image..."):
                image = load_image_from_upload(uploaded_file)
            
            if image is None:
                st.error("Failed to load image. Please try another file.")
                continue
            
            # Display original image and results side-by-side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Image**")
                st.image(image, clamp=True, channels="GRAY")
                
                # Image info
                st.caption(f"Size: {image.shape[1]} x {image.shape[0]} pixels")
                
                # Quality validation
                if validate_quality:
                    with st.expander("🔍 Image Quality Check"):
                        try:
                            is_valid = validate_lfa_image(image)
                            if is_valid:
                                st.success("✓ Image passed quality checks")
                            else:
                                st.warning("⚠️ Image quality issues detected (see warnings above)")
                        except Exception as e:
                            st.warning(f"Quality check failed: {e}")
            
            with col2:
                # Run analysis
                with st.spinner("Analysing..."):
                    try:
                        results = analyse_lfa(
                            image,
                            quantile_low=quantile_low,
                            quantile_high=quantile_high,
                            smooth_window=smooth_window,
                            baseline_region_size=baseline_region,
                            als_lambda=als_lambda,
                            als_p=als_p,
                            als_niter=als_niter,
                            auc_window=auc_window
                        )
                        
                        # Display metrics
                        st.markdown("**Analysis Results**")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric(
                                "TL1 Peak",
                                f"{results['TL1_peak']:.4f}",
                                help="Maximum intensity in TL1 region"
                            )
                        
                        with metric_col2:
                            st.metric(
                                "TL2 Peak",
                                f"{results['TL2_peak']:.4f}",
                                help="Maximum intensity in TL2 region"
                            )
                        
                        with metric_col3:
                            # Color-code ratio
                            ratio = results['ratio']
                            if ratio > 0.8:
                                delta_color = "normal"
                            elif ratio > 0.5:
                                delta_color = "off"
                            else:
                                delta_color = "inverse"
                            
                            st.metric(
                                "TL1/TL2 Peak Ratio",
                                f"{ratio:.4f}",
                                delta=None,
                                help="TL1-to-TL2 Peak ratio"
                            )

                        # AUC metrics row
                        st.markdown("---")
                        auc_col1, auc_col2, auc_col3 = st.columns(3)
                        # print(results)
                        with auc_col1:
                            st.metric(
                                "TL1 AUC",
                                f"{results['TL1_auc']:.2f}",
                                help="Area under curve for TL1 region"
                            )
                        
                        with auc_col2:
                            st.metric(
                                "TL2 AUC",
                                f"{results['TL2_auc']:.2f}",
                                help="Area under curve for TL2 region"
                            )
                        
                        with auc_col3:
                            st.metric(
                                "TL1/TL2 AUC Ratio",
                                f"{results['auc_ratio']:.4f}",
                                delta=None,
                                help="TL1-to-TL2 AUC ratio"
                            )

                        # Interpretation guide
                        with st.expander("📖 Interpretation Guide"):
                            st.markdown("""
                            **TL1/TL2 Ratio Interpretation:**
                            - **> 0.8**: Strong positive result
                            - **0.5 - 0.8**: Moderate positive result
                            - **< 0.5**: Weak or negative result
                            
                            *Note: Interpretation thresholds should be validated for your specific assay.*
                            """)
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        continue
            
            # Intensity profile plot
            st.markdown("**Intensity Profile**")
            try:
                fig = create_intensity_plot(
                    results, 
                    title="Baseline-Corrected Intensity Profile",
                    show=False,  # Don't open in browser
                    show_raw=show_raw_overlay, 
                    show_peaks=show_peaks
                )
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(f"Plotting failed: {e}")
            
            # Export options
            st.markdown("**Export Results**")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                csv_data = export_to_csv_string(results)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{uploaded_file.name.split('.')[0]}_results.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            with export_col2:
                # Create JSON export
                import json
                json_data = {
                    'filename': uploaded_file.name,
                    'TL1_peak': float(results['TL1_peak']),
                    'TL2_peak': float(results['TL2_peak']),
                    'TL1/TL2 Peak Ratio': float(results['ratio']),
                    'TL1_auc': float(results['TL1_auc']),
                    'TL2_auc': float(results['TL2_auc']),
                    'TL1/TL2 AUC ratio': float(results['auc_ratio']),
                    'metadata': results['metadata']
                }
                json_string = json.dumps(json_data, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_string,
                    file_name=f"{uploaded_file.name.split('.')[0]}_results.json",
                    mime="application/json",
                    width='stretch'
                )
    
    # Footer
    st.divider()
    st.markdown("""
    <p style='text-align: center; color: #999; font-size: 0.9rem;'>
    LFA Image Analyser v0.01
    <a href='https://github.com/xxxxxx/lfa-analyser' target='_blank'>GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()