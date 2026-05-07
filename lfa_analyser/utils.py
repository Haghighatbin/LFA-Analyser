"""
Utility Functions

Helpers for exporting results, plotting, and formatting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
# from streamlit_theme import st_theme

# def _get_theme_colours():
#     """
#     Detect current Streamlit theme and return appropriate colour scheme.
    
#     Returns
#     -------
#     dict
#         Colour scheme dictionary for plot styling
#     """
#     theme = None
#     try:
#         # Detect theme from Streamlit config
#         # theme = st.session_state.get("theme", "dark") 
#         theme = st_theme()
#         if theme and theme.get("base") == "light":
#             is_dark = False
#     except:
#         # Default to light if detection fails
#         pass

#     # If we got a valid theme, cache it
#     if theme and theme.get("base"):
#         st.session_state["detected_theme"] = theme.get("base")
    
#     # Use cached theme, or default to dark if nothing detected yet
#     current_theme = st.session_state.get("detected_theme", "dark")
#     is_dark = (current_theme == "dark")

#     if is_dark:
#         return {
#             'plot_bg': '#0e1117',
#             'paper_bg': '#0e1117',
#             'grid_colour': 'rgba(255, 255, 255, 0.1)',
#             'text_colour': '#fafafa',
#             'line_colour': '#2E86AB',
#             'vline_colour': '#fafafa',
#             'legend_bg': 'rgba(14, 17, 23, 0.8)',
#         }
#     else:
#         return {
#             'plot_bg': 'white',
#             'paper_bg': 'white',
#             'grid_colour': 'rgba(128, 128, 128, 0.2)',
#             'text_colour': '#000000',
#             'line_colour': '#2E86AB',
#             'vline_colour': 'gray',
#             'legend_bg': 'rgba(255, 255, 255, 0.8)',
#         }

def _get_theme_colours():
    try:
        is_dark = st.get_option("theme.base") != "light"
    except Exception:
        is_dark = True

    if is_dark:
        return {
            'plot_bg': '#0e1117',
            'paper_bg': '#0e1117',
            'grid_colour': 'rgba(255, 255, 255, 0.1)',
            'text_colour': '#fafafa',
            'line_colour': '#2E86AB',
            'vline_colour': '#fafafa',
            'legend_bg': 'rgba(14, 17, 23, 0.8)',
        }
    else:
        return {
            'plot_bg': 'white',
            'paper_bg': 'white',
            'grid_colour': 'rgba(128, 128, 128, 0.2)',
            'text_colour': '#000000',
            'line_colour': '#2E86AB',
            'vline_colour': 'gray',
            'legend_bg': 'rgba(255, 255, 255, 0.8)',
        }

def export_results_to_csv(
    results: Union[Dict, List[Dict]], 
    output_path: Union[str, Path]
) -> None:
    """
    Export analysis results to CSV file.
    
    Parameters
    ----------
    results : dict or list of dict
        Single result or list of results from analyse_lfa
    output_path : str or Path
        Output CSV file path
    """
    output_path = Path(output_path)
    
    # Convert single result to list
    if isinstance(results, dict):
        results = [results]
    
    # Extract key metrics
    data = []
    for i, result in enumerate(results):
        row = {
            'image_index': i,
            'TL1_peak': result['TL1_peak'],
            'TL2_peak': result['TL2_peak'],
            'ratio': result['ratio'],
            'TL1_auc': result['TL1_auc'],
            'TL2_auc': result['TL2_auc'],
            'auc_ratio': result['auc_ratio'],
            'image_width': result['metadata']['image_size'][1],
            'image_height': result['metadata']['image_size'][0]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results exported to: {output_path}")

def export_results_to_json(
    results: Union[Dict, List[Dict]], 
    output_path: Union[str, Path],
    include_profiles: bool = False
) -> None:
    """
    Export analysis results to JSON file.
    
    Parameters
    ----------
    results : dict or list of dict
        Single result or list of results
    output_path : str or Path
        Output JSON file path
    include_profiles : bool
        Whether to include full intensity profiles (can be large)
    """
    output_path = Path(output_path)
    
    # Convert single result to list
    if isinstance(results, dict):
        results = [results]
    
    # Prepare data for JSON serialisation
    json_data = []
    for result in results:
        data = {
            'TL1_peak': float(result['TL1_peak']),
            'TL2_peak': float(result['TL2_peak']),
            'ratio': float(result['ratio']),
            'TL1_auc': float(result['TL1_auc']),
            'TL2_auc': float(result['TL2_auc']),
            'auc_ratio': float(result['auc_ratio']),
            'metadata': result['metadata']
        }
        
        if include_profiles:
            data['intensity_profile'] = result['intensity_profile'].tolist()
            data['raw_profile'] = result['raw_profile'].tolist()
        
        json_data.append(data)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results exported to: {output_path}")

def create_intensity_plot(
    results: Dict,
    title: str = "LFA Intensity Profile",
    save_path: Union[str, Path] = None,
    show: bool = True,
    show_raw: bool = False, 
    show_peaks: bool = True
) -> go.Figure:
    """
    Create interactive intensity profile plot using Plotly.
    
    Features:
    - Hover tooltips showing (x, y) coordinates
    - Zoom and pan capabilities
    - Peak markers with annotations
    - Professional styling with customisable colours
    - Export to HTML or static image
    
    Parameters
    ----------
    results : dict
        Results from analyse_lfa
    title : str
        Plot title
    save_path : str or Path, optional
        If provided, save plot to this path (.html or .png)
    show : bool
        Whether to display plot (creates HTML and opens in browser)
    show_raw : bool
        Whether to overlay the raw (pre-ALS) intensity profile
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure object
    """
    # Get theme colours
    colours = _get_theme_colours()
    intensity = results['intensity_profile']
    x = np.arange(len(intensity))
    
    # Create figure
    fig = go.Figure()   

    # Add raw profile if requested (optional overlay)
    if show_raw and 'raw_profile' in results:
        raw_intensity = results['raw_profile']
        fig.add_trace(go.Scatter(
            x=x,
            y=raw_intensity,
            mode='lines',
            name='Raw Intensity',
            line=dict(color='#CCCCCC', width=1.5, dash='dot'),
            hovertemplate='<b>Raw</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
        ))

    # AUC shaded regions (behind the intensity lines)
    half = len(intensity) // 2
    
    # Get peak positions and AUC window from results
    TL1_peak_idx = results.get('TL1_peak_idx', np.argmax(intensity[:half]))
    TL2_peak_idx = results.get('TL2_peak_idx', half + np.argmax(intensity[half:]))
    auc_window = results.get('metadata', {}).get('auc_window', 5)
    
    # TL1 AUC shading (window around TL1 peak)
    if TL1_peak_idx is not None:
        TL1_start = max(0, TL1_peak_idx - auc_window)
        TL1_end = min(half, TL1_peak_idx + auc_window + 1)
        fig.add_trace(go.Scatter(
            x=x[TL1_start:TL1_end],
            y=np.maximum(intensity[TL1_start:TL1_end], 0),
            fill='tozeroy',
            mode='none',
            name=f'TL1 AUC: {results["TL1_auc"]:.2f}',
            fillcolor='rgba(230, 57, 70, 0.15)',  # Semi-transparent red
            hovertemplate='<b>TL1 AUC Window</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>',
            showlegend=True
        ))
    
    # TL2 AUC shading (window around TL2 peak)
    if TL2_peak_idx is not None:
        TL2_start = max(half, TL2_peak_idx - auc_window)
        TL2_end = min(len(intensity), TL2_peak_idx + auc_window + 1)
        fig.add_trace(go.Scatter(
            x=x[TL2_start:TL2_end],
            y=np.maximum(intensity[TL2_start:TL2_end], 0),
            fill='tozeroy',
            mode='none',
            name=f'TL2 AUC: {results["TL2_auc"]:.2f}',
            fillcolor='rgba(6, 214, 160, 0.15)',  # Semi-transparent green
            hovertemplate='<b>TL2 AUC Window</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>',
            showlegend=True
        ))

    # Add corrected intensity profile
    fig.add_trace(go.Scatter(
        x=x,
        y=intensity,
        mode='lines',
        name='Corrected Intensity',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>Corrected</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
    ))
    
    # Find peak positions
    TL1_peak_idx = np.argmax(intensity[:half])
    TL2_peak_idx = half + np.argmax(intensity[half:])
    
    # Add TL1 peak marker
    TL1_peak_idx = results.get('TL1_peak_idx')
    if TL1_peak_idx is not None:
        if show_peaks:
            fig.add_trace(go.Scatter(
                x=[TL1_peak_idx],
                y=[intensity[TL1_peak_idx]],
                mode='markers',
                name=f'TL1 Peak: {results["TL1_peak"]:.4f}',
                marker=dict(color='#E63946', size=12, symbol='circle',
                            line=dict(color='white', width=2)),
                hovertemplate='<b>TL1 Peak</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
            ))
    
    # Add TL2 peak marker
    TL2_peak_idx = results.get('TL2_peak_idx')
    if TL2_peak_idx is not None:
        if show_peaks:
            fig.add_trace(go.Scatter(
                x=[TL2_peak_idx],
                y=[intensity[TL2_peak_idx]],
                mode='markers',
                name=f'TL2 Peak: {results["TL2_peak"]:.4f}',
                marker=dict(color='#06D6A0', size=12, symbol='circle',
                            line=dict(color='white', width=2)),
                hovertemplate='<b>TL2 Peak</b><br>Position: %{x}<br>Intensity: %{y:.4f}<extra></extra>'
            ))
    
    # Add midpoint vertical line
    fig.add_vline(
        x=half,
        line_dash="dash",
        line_color=colours['vline_colour'],
        opacity=0.5,
        annotation_text="Midpoint",
        annotation_position="top",
        annotation_font_color=colours['text_colour']
    )
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>TL1/TL2 Ratio: {results["ratio"]:.4f}</sub>',
            font=dict(size=16, family="Courier New, monospace"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                    text='Position (pixels)',
                    font=dict(size=13, family="Courier New, monospace", color=colours['text_colour'])
                ),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            color=colours['text_colour']
        ),
        yaxis=dict(
            title=dict(
                    text='Intensity (a.u.)',
                    font=dict(size=13, family="Courier New, monospace", color=colours['text_colour'])
                ),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            color=colours['text_colour']
        ),
        plot_bgcolor=colours['plot_bg'],
        paper_bgcolor=colours['paper_bg'],
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor=colours['legend_bg'],
            bordercolor=colours['grid_colour'],
            borderwidth=1,
            font=dict(family="Courier New, monospace", color=colours['text_colour'])
        ),
        font=dict(family="Courier New, monospace", color=colours['text_colour']),
        height=500,
        margin=dict(l=80, r=150, t=100, b=80)
    )
    
    # Add interactive tools configuration
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'lfa_intensity_profile',
            'height': 600,
            'width': 1200,
            'scale': 2
        },
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        
        if save_path.suffix == '.html':
            # Save as interactive HTML
            fig.write_html(save_path, config=config)
            print(f"Interactive plot saved to: {save_path}")
        else:
            # Save as static image (PNG, JPEG, SVG, PDF)
            fig.write_image(save_path, width=1200, height=600, scale=2)
            print(f"Plot saved to: {save_path}")
    
    # Display in browser if requested
    if show:
        fig.show(config=config)
    
    return fig

def format_results_table(results: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    Format results as a pandas DataFrame for display.
    
    Parameters
    ----------
    results : dict or list of dict
        Analysis results
    
    Returns
    -------
    pd.DataFrame
        Formatted results table
    """
    if isinstance(results, dict):
        results = [results]
    
    data = {
        'TL1 Peak': [r['TL1_peak'] for r in results],
        'TL2 Peak': [r['TL2_peak'] for r in results],
        'TL1/TL2 Peak Ratio': [r['ratio'] for r in results],
        'TL1 AUC': [r['TL1_auc'] for r in results],
        'TL2 AUC': [r['TL2_auc'] for r in results],
        'TL1/TL2 AUC Ratio': [r['auc_ratio'] for r in results],
        'Image Size': [f"{r['metadata']['image_size'][1]}x{r['metadata']['image_size'][0]}" 
                       for r in results]
    }
    
    df = pd.DataFrame(data)
    return df

def print_results_summary(results: Dict, filename: str = None) -> None:
    """
    Print formatted summary of analysis results.
    
    Parameters
    ----------
    results : dict
        Results from analyse_lfa
    filename : str, optional
        Name of analysed file
    """
    print("\n" + "="*60)
    if filename:
        print(f"Analysis Results: {filename}")
    else:
        print("Analysis Results")
    print("="*60)
    
    print(f"\TL1 Peak Intensity:    {results['TL1_peak']:.4f}")
    print(f"TL2 Peak Intensity: {results['TL2_peak']:.4f}")
    print(f"TL1/TL2 Ratio:     {results['ratio']:.4f}")

    print(f"\nTL1 AUC:               {results['TL1_auc']:.2f}")
    print(f"TL2 AUC:            {results['TL2_auc']:.2f}")
    print(f"AUC Ratio:          {results['auc_ratio']:.4f}")

    print(f"\nImage Size: {results['metadata']['image_size'][1]} x {results['metadata']['image_size'][0]} pixels")
    
    print("\nAnalysis Parameters:")
    meta = results['metadata']
    print(f"  Quantile Range:     {meta['quantile_low']:.2f} - {meta['quantile_high']:.2f}")
    print(f"  Smoothing Window:   {meta['smooth_window']} pixels")
    print(f"  ALS Lambda:         {meta['als_lambda']}")
    print(f"  ALS p:              {meta['als_p']}")
    
    print("="*60 + "\n")

def create_batch_summary(
    results_dict: Dict[str, Dict],
    output_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Create summary table from batch processing results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping filenames to results
    output_path : str or Path, optional
        Save summary to CSV if provided
    
    Returns
    -------
    pd.DataFrame
        Summary table
    """
    data = []
    for filename, result in results_dict.items():
        row = {
            'Filename': filename,
            'TL1 Peak': result['TL1_peak'],
            'TL2 Peak': result['TL2_peak'],
            'TL1/TL2 Peak Ratio': result['ratio'],
            'TL1 AUC': result['TL1_auc'],
            'TL2 AUC': result['TL2_auc'],
            'TL1/TL2 AUC Ratio': result['auc_ratio'],
            'Image Width': result['metadata']['image_size'][1],
            'Image Height': result['metadata']['image_size'][0]
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Filename')
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Batch summary exported to: {output_path}")
    
    return df