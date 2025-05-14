"""
Functions for generating comparison charts for model performance.
"""
import logging
import os
import pandas as pd
import numpy as np 
# import plotly.express as px # Comment out plotly express for now
# import plotly.graph_objects as go # Comment out plotly graph objects
# from plotly.subplots import make_subplots # Comment out plotly subplots
# import plotly.io as pio # Comment out plotly io
from . import config
from .utils import ui_utils
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from typing import Dict, List, Union, Optional, Any
import matplotlib.cm as cm
from pathlib import Path

logger = logging.getLogger(__name__)
INSPIRED_PALETTE = ["#4C72B0", "#DD8452", "#8C8C8C", "#595959", "#9370DB", "#57A057"]

sns.set_theme(
    style="ticks", 
    palette=INSPIRED_PALETTE, 
    font="sans-serif" 
)
plt.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], 
    "axes.labelsize": 11, 
    "axes.titlesize": 13, 
    "font.size": 11,      
    "legend.fontsize": 10,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10, 
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False, 
    # "grid.color": "#E0E0E0", 
    "axes.edgecolor": "#333333", # Darker edge color for axes lines, similar to reference
    "axes.linewidth": 1.2, # Make axis lines a bit thicker like in reference
    "axes.titlepad": 15, 
    "figure.facecolor": "white", 
    "savefig.facecolor": "white", 
    "xtick.direction": "out", # Ticks point outwards
    "ytick.direction": "out", # Ticks point outwards
    "xtick.major.size": 5, # Length of the major x-ticks
    "ytick.major.size": 5, # Length of the major y-ticks
    "xtick.major.width": 1.2, # Thickness of x-ticks
    "ytick.major.width": 1.2, # Thickness of y-ticks
    "xtick.minor.size": 3,  # For minor ticks, if ever used
    "ytick.minor.size": 3,  # For minor ticks, if ever used
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.bottom": True, # Ensure bottom x-ticks are on
    "ytick.left": True,   # Ensure left y-ticks are on
    # "text.usetex": False, 
})
# --- End Updated Matplotlib/Seaborn Styling ---


def _wrap_labels(ax, width, break_long_words=False):
    """Wraps labels on an axes object."""
    labels = []
    ticks = ax.get_xticks()
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)


def _safe_get(data_dict: Dict[str, Any], keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    for key in keys:
        if key in data_dict:
            return data_dict[key]
    return default


def _prepare_plot_data(all_model_run_summaries: dict) -> pd.DataFrame | None:
    """
    Prepares data from the summary dictionary into a pandas DataFrame suitable for plotting.

    Args:
        all_model_run_summaries: A dictionary containing summaries of model runs.
                                 Expected structure: {model_id: {strategy_name: {metrics...}}}

    Returns:
        A pandas DataFrame with columns like 'Model', 'Strategy', 'Metric', 'Value',
        or None if the input is empty or malformed.
    """
    plot_data = []
    numerical_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'average_latency_ms', 'total_cost', 'total_tokens', 'avg_answer_length', 'total_run_time_s']

    if not all_model_run_summaries:
        logger.warning("No model run summaries provided for plotting.")
        return None

    for model_id, strategies in all_model_run_summaries.items():
        if not isinstance(strategies, dict):
            logger.warning(f"Expected dictionary of strategies for model '{model_id}', got {type(strategies)}. Skipping.")
            continue
        for strategy_name, combined_metrics in strategies.items():
            if not isinstance(combined_metrics, dict):
                logger.warning(f"Expected dictionary of combined metrics for model '{model_id}', strategy '{strategy_name}', got {type(combined_metrics)}. Skipping.")
                continue

            if "error" in combined_metrics:
                logger.warning(f"Run for model '{model_id}', strategy '{strategy_name}' encountered an error: {combined_metrics['error']}. Skipping metrics.")
                continue

            for metric_name in numerical_metrics:
                value = combined_metrics.get(metric_name)
                if isinstance(value, (int, float)):
                     plot_data.append({
                        'Model': model_id,
                        'Strategy': strategy_name, 
                        'Metric': metric_name,
                        'Value': float(value)
                    })
                elif value is not None:
                     logger.debug(f"Metric '{metric_name}' for {model_id}/{strategy_name} has non-numeric value '{value}'. Skipping for numerical plot.")


    if not plot_data:
        logger.warning("No valid data points extracted for plotting.")
        return None

    df = pd.DataFrame(plot_data)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True) 

    if df.empty:
        logger.warning("DataFrame is empty after processing and cleaning. No plots will be generated.")
        return None

    logger.info(f"Prepared DataFrame for plotting with {len(df)} valid rows.")
    return df

def _get_strategy_type(strategy_name: str) -> str:
    """Extracts the base strategy type from the full strategy name."""
    if "Self-Consistency" in strategy_name:
        return "SC-CoT"
    elif "Self-Discover" in strategy_name:
        return "Self-Discover"
    elif "Default" in strategy_name:
        return "Default"
    else:
        return "Unknown"

def _get_strategy_param(strategy_name: str) -> str:
    """Extracts the parameter (e.g., N=3) from the strategy name, if present."""
    if "N=3" in strategy_name:
        return "N=3"
    elif "N=5" in strategy_name:
        return "N=5"
    else:
        return "" 


# --- Modify Plotting Function to use Matplotlib/Seaborn ---
def _plot_metric_by_strategy_comparison(df: pd.DataFrame, output_dir: Union[str, Path], metric: str):
    """Generates grouped bar chart comparing key strategies for each model using Seaborn."""
    metric_df = df[df['Metric'] == metric].copy()
    if metric_df.empty:
        logger.info(f"Skipping '{metric}' by strategy comparison plot: No data found for this metric.")
        return

    # Keep the original filtering for strategies to compare if needed, or adjust as necessary
    strategies_to_compare_explicit = [
        s for s in df['Strategy'].unique() 
        if "Default" in s or "Self-Discover" in s or ("Self-Consistency CoT" in s and "N=3" in s)
    ]
    if not strategies_to_compare_explicit:
         logger.info(f"Skipping '{metric}' by strategy comparison plot: No standard strategies found.")
         return

    df_comp = metric_df[metric_df['Strategy'].isin(strategies_to_compare_explicit)].copy()

    if df_comp.empty or df_comp['Model'].nunique() < 1:
        included_models = df_comp['Model'].unique().tolist() if not df_comp.empty else []
        logger.info(f"Skipping '{metric}' by strategy comparison plot: Not enough data for comparison across models. Found models: {included_models}")
        return
    
    metric_title = metric.replace('_', ' ').title()
    # Updated title for clarity, will be styled further
    title = f'{metric_title} Comparison Across Strategies and Models'

    # Dynamically adjust height based on number of strategies and models
    num_models = df_comp['Model'].nunique()
    num_strategies = df_comp['Strategy'].nunique()
    
    # Start with a base height and add per model, adjust multiplier as needed
    # The width might need adjustment if model names are long.
    plt.figure(figsize=(max(10, num_models * 2), 6 + num_strategies * 0.5))


    # Use the INSPIRED_PALETTE defined at the top of the file
    # Dynamically adjust the number of colors from the palette based on the number of unique strategies
    current_palette = INSPIRED_PALETTE[:num_strategies] if num_strategies > 0 else INSPIRED_PALETTE[:1]

    # Swapped x and y, changed orient to 'v', and hue to 'Strategy'
    ax = sns.barplot(data=df_comp, x='Model', y='Value', hue='Strategy', palette=current_palette)

    # Add data labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        try:
            if height == 0:
                label_text = '0'
            elif abs(height) < 0.001 and height != 0:
                label_text = f'{height:.2e}'
            elif abs(height) < 1:
                label_text = f'{height:.3f}'
            elif abs(height) < 100:
                label_text = f'{height:.2f}'
            else:
                label_text = f'{int(round(height))}'
        except TypeError:
            label_text = "N/A"

        ax.text(p.get_x() + p.get_width() / 2.,
                height + (ax.get_ylim()[1] * 0.01), # Position label slightly above the bar
                label_text,
                ha='center', 
                va='bottom',
                fontsize=plt.rcParams["font.size"] * 0.8)

    ax.set_xlabel("Model") # X-axis is now Model
    ax.set_ylabel(metric_title) # Y-axis is the metric value
    
    # Set title with bold font and left alignment
    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"], pad=plt.rcParams["axes.titlepad"], loc='left', fontweight='bold')

    # Add solid y-axis grid lines and ensure they are behind the bars
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.set_axisbelow(True)

    # Set y-axis limits, especially for percentage-like metrics
    if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall'] or \
       'rate' in metric.lower() or 'percentage' in metric.lower() or 'score' in metric.lower():
        current_max_val = df_comp['Value'].max() if not df_comp.empty else 1.0
        current_min_val = df_comp['Value'].min() if not df_comp.empty else 0.0
        
        # Ensure y-axis starts at 0 (or slightly below if negative values are possible and present)
        plot_min_y = 0 if current_min_val >= 0 else current_min_val * 1.1 # Adjust if negative values are meaningful

        # For scores typically between 0 and 1
        if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:
            # Ensure y-axis goes up to at least 1.0 (or a bit more)
            plot_max_y = max(1.05, current_max_val * 1.05 if current_max_val > 0 else 1.05)
        else:
            plot_max_y = current_max_val * 1.1 if current_max_val > 0 else 0.1 # For other rates/percentages
            if current_max_val == 0 and current_min_val == 0 : plot_max_y = 0.1 # Handle all zero case
        
        ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
    else:
        # For other metrics like latency, ensure it starts at 0 if all values are positive
        if not df_comp.empty and df_comp['Value'].min() >= 0:
            ax.set_ylim(bottom=0, top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None)
        elif not df_comp.empty:
            ax.set_ylim(top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None) # Allow auto bottom for negative values if any

    # Improve legend placement and title
    plt.legend(title='Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    # Rotate x-axis labels if model names are long
    plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) # Adjust rotation and size as needed

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False) 

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Updated filename to reflect new structure if desired, or keep as is
    chart_filename = output_path / f"comparison_{metric.lower()}_by_model_and_strategy.png" 
    plt.savefig(chart_filename, dpi=plt.rcParams["savefig.dpi"], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {chart_filename}")


def _plot_sc_comparison(df: pd.DataFrame, output_dir: Union[str, Path], metric: str = 'accuracy'):
    """Generates grouped bar chart comparing SC-CoT N=3 vs N=5 using Seaborn."""
    metric_df = df[df['Metric'] == metric].copy()
    if metric_df.empty:
        logger.info(f"Skipping SC '{metric}' comparison plot: No data found for this metric.")
        return

    metric_df['strategy_type'] = metric_df['Strategy'].apply(_get_strategy_type)
    metric_df['strategy_param'] = metric_df['Strategy'].apply(_get_strategy_param)
    metric_df['base_model_id'] = metric_df['Model']

    # Filter for SC-CoT strategies only
    df_comp = metric_df[metric_df['strategy_type'] == 'SC-CoT'].copy()

    # Ensure we have comparable parameters (like N=3, N=5)
    comparable_params = df_comp[df_comp['strategy_param'].str.contains(r'N=\d+', regex=True)]['strategy_param'].unique()
    if len(comparable_params) < 2:
        logger.info(f"Skipping SC '{metric}' comparison plot: Need results for at least two different N values (e.g., N=3 and N=5). Found: {comparable_params}")
        return

    # Keep only data with comparable N parameters
    df_comp = df_comp[df_comp['strategy_param'].isin(comparable_params)].copy()

    metric_title = metric.replace('_', ' ').title()
    title = f'Self-Consistency CoT {metric_title}: Comparison by Samples (N)'

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_comp, x='base_model_id', y='Value', hue='strategy_param',
                     errorbar=None) # Add error bars if needed

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)

    ax.set_xlabel("Base Model")
    ax.set_ylabel(metric_title)
    ax.set_title(title, loc='left', fontweight='bold') # Use rcParams size, add loc and fontweight

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(bottom=0)
    if metric == 'accuracy' or metric == 'f1_score' or metric == 'precision' or metric == 'recall':
         plt.ylim(top=max(1.05, df_comp['Value'].max() * 1.1 if not df_comp.empty else 1.05))
    else:
         plt.ylim(top=df_comp['Value'].max() * 1.15 if not df_comp.empty else None)

    # Move legend outside
    ax.legend(title='Samples (N)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    filename_base = f'comparison_sc_{metric}_n_samples'
    output_path = Path(output_dir) / f'{filename_base}.png'
    try:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()
# --- End SC Comparison Conversion ---


# --- Convert Scatter Plot to Seaborn ---
def _plot_scatter_tradeoff(df: pd.DataFrame, output_dir: Union[str, Path], metric_y: str, metric_x: str):
    """Generates scatter plot showing a trade-off between two metrics using Seaborn."""
    try:
        df_pivot = df.pivot_table(index=['Model', 'Strategy'], columns='Metric', values='Value').reset_index()
    except Exception as e:
        logger.error(f"Failed to pivot DataFrame for scatter plot ({metric_y} vs {metric_x}): {e}. Columns: {df.columns}, Metrics: {df['Metric'].unique()}", exc_info=True)
        return

    if metric_x not in df_pivot.columns or metric_y not in df_pivot.columns:
        logger.warning(f"Skipping {metric_y} vs {metric_x} plot: One or both metrics not found after pivoting. Available: {df_pivot.columns.tolist()}")
        return

    df_plot = df_pivot.dropna(subset=[metric_x, metric_y]).copy()
    if df_plot.empty:
        logger.warning(f"No data points with both '{metric_y}' and '{metric_x}' available for scatter plot.")
        return

    # Add strategy type for marker style and base model ID
    df_plot['strategy_type'] = df_plot['Strategy'].apply(_get_strategy_type)
    df_plot['base_model_id'] = df_plot['Model']

    # Prepare titles and labels
    metric_y_title = metric_y.replace('_', ' ').title()
    metric_x_title = metric_x.replace('_', ' ').replace(' Ms', ' (ms)').replace(' S', ' (s)').title()
    if metric_x == 'total_cost':
        metric_x_title += " ($)"
    title = f'{metric_y_title} vs. {metric_x_title} Trade-off'

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        data=df_plot,
        x=metric_x,
        y=metric_y,
        hue='base_model_id',
        style='strategy_type',
        s=100, # Adjust marker size
        alpha=0.8,
        edgecolor='k', # Add marker edge color
        linewidth=0.5
    )

    ax.set_xlabel(metric_x_title)
    ax.set_ylabel(metric_y_title)
    ax.set_title(title, loc='left', fontweight='bold')

    # Adjust y-axis for accuracy-like metrics
    if metric_y in ['accuracy', 'f1_score', 'precision', 'recall']:
        min_y = df_plot[metric_y].min()
        max_y = df_plot[metric_y].max()
        ax.set_ylim(bottom=min_y * 0.95 if min_y > 0 else -0.05,
                    top=max(1.0, max_y * 1.05) if max_y < 1 else max_y * 1.05)


    # Move legend outside
    ax.legend(title='Legend (Model / Strategy)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Optionally add text labels (can get crowded)
    # for i, point in df_plot.iterrows():
    #     ax.text(point[metric_x] * 1.01, point[metric_y], f"{point['base_model_id']}", fontsize=7)

    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename_base = f'tradeoff_{metric_y}_vs_{metric_x}'
    output_path = Path(output_dir) / f'{filename_base}.png'
    try:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()
# --- End Scatter Plot Conversion ---


# --- Remove Goodfin Overrides from _plot_total_time_comparison ---
def _plot_total_time_comparison(df: pd.DataFrame, output_dir: Union[str, Path]):
    """Plots a comparison of average latency across models and strategies using Seaborn."""
    time_df = df[df['Metric'] == 'average_latency_ms'].copy()
    if time_df.empty:
        logger.warning("No average latency data (average_latency_ms) found to plot time comparison.")
        return

    plt.figure(figsize=(12, 7))
    num_models = time_df['Model'].nunique()
    models = sorted(time_df['Model'].unique()) # Sort for consistent hue order

    # Use the globally set Seaborn theme palette
    ax = sns.barplot(data=time_df, x='Strategy', y='Value', hue='Model', hue_order=models,
                     dodge=True, errorbar=None) # Rely on global theme

    plt.title('Average Latency Comparison Across Strategies and Models', loc='left', fontweight='bold') # Use rcParams size, add loc and fontweight
    plt.ylabel('Average Latency (ms)') # Use rcParams size
    plt.xlabel('Strategy') # Use rcParams size
    plt.xticks(rotation=30, ha='right') # Use rcParams size
    plt.yticks() # Use rcParams size

    # Move legend outside
    legend = ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    # plt.setp(legend.get_title(), color=GOODFIN_TEXT_COLOR) # Remove Goodfin color
    # plt.setp(legend.get_texts(), color=GOODFIN_MUTED_TEXT_COLOR) # Remove Goodfin color

    # Add bar labels
    for container in ax.containers:
        labels = [f'{v:.0f}' if v >= 1 else f'{v:.3f}' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', padding=3,
                     fontsize=plt.rcParams['xtick.labelsize'] - 1) # Use relative size

    ax.set_ylim(bottom=0)
    sns.despine() # Keep default despine (top/right)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust rect for legend

    output_path = Path(output_dir) / "average_latency_comparison.png"
    try:
        plt.savefig(output_path, bbox_inches='tight') # Use global DPI
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()
# --- End Goodfin Override Removal ---


# --- Remove Goodfin Overrides from _plot_metric_comparison_for_strategy ---
def _plot_metric_comparison_for_strategy(df: pd.DataFrame, strategy_name: str, metrics_to_plot: list[str], output_dir: Union[str, Path]):
    """
    Generates separate bar charts comparing models for a specific strategy using Seaborn.
    Args:
        df: DataFrame containing the prepared plot data.
        strategy_name: The specific strategy name to plot comparisons for.
        metrics_to_plot: A list of metric names (strings) to generate plots for.
        output_dir: The directory to save the generated plots.
    """
    strategy_df_full = df[df['Strategy'] == strategy_name].copy()
    if strategy_df_full.empty:
        logger.warning(f"No data found for strategy '{strategy_name}'. Skipping metric comparison plots.")
        return

    num_models = strategy_df_full['Model'].nunique()
    if num_models == 0:
        logger.info(f"No models found for strategy '{strategy_name}'. Skipping model comparison plots.")
        return

    models = sorted(strategy_df_full['Model'].unique()) # Sort for consistent x-axis order

    for metric in metrics_to_plot:
        metric_df = strategy_df_full[strategy_df_full['Metric'] == metric]
        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric}' in strategy '{strategy_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(max(6, num_models * 1.5), 5)) # Adjust width based on model count

        ax = sns.barplot(data=metric_df, x='Model', y='Value', order=models,
                         hue='Model', legend=False, # Use global palette
                         errorbar=None) # Add errorbar if needed

        metric_title = metric.replace('_', ' ').title()
        # Add units for specific metrics
        if metric == 'total_cost':
            ylabel = f'{metric_title} ($)'
            label_fmt = '${:,.3f}'
        elif metric == 'average_latency_ms':
            ylabel = f'{metric_title} (ms)'
            label_fmt = '{:.0f}'
        elif metric == 'total_run_time_s':
            ylabel = f'{metric_title} (s)'
            label_fmt = '{:.1f}s'
        elif metric == 'total_tokens':
            ylabel = f'{metric_title} (tokens)'
            label_fmt = '{:,.0f}'
        else:
            ylabel = metric_title
            label_fmt = '{:.3f}'

        plot_title = f"{metric_title} for {strategy_name}"
        
        plt.title(plot_title, loc='left', fontweight='bold', wrap=True) # Add loc and fontweight
        plt.ylabel(ylabel) # Use rcParams size
        plt.xlabel('Model') # Use rcParams size
        plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) # Keep rotation 0 if names fit, use rcParams size
        plt.yticks() # Use rcParams size

        # Add bar labels
        for container in ax.containers:
            labels = [label_fmt.format(v) for v in container.datavalues]
            ax.bar_label(container, labels=labels,
                         fontsize=plt.rcParams['xtick.labelsize'] -1, padding=3) # Relative size

        # Set Y limits appropriately
        if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            ax.set_ylim(bottom=0, top=max(1.05, metric_df['Value'].max() * 1.1))
        elif metric_df['Value'].min() >= 0:
             ax.set_ylim(bottom=0, top=metric_df['Value'].max() * 1.15 if metric_df['Value'].max() > 0 else 0.1)
        # else: keep default limits

        if metric_df['Value'].max() == 0:
             ax.set_ylim(bottom=-0.001, top=0.01)
             ax.set_yticks([0])

        # _wrap_labels(ax, width=15 if num_models > 5 else 20) # Wrap labels only if many models
        sns.despine()
        plt.tight_layout()

        safe_strategy_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('/', '')
        output_path = Path(output_dir) / f"{safe_strategy_name}_strategy_{metric}_comparison.png"
        try:
            plt.savefig(output_path, bbox_inches='tight') # Use global DPI
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {output_path}: {e}")
            plt.close()
# --- End Goodfin Override Removal ---


def _plot_confusion_matrix(matrix: Union[np.ndarray, List[List[int]]], labels: List[str], model_id: str, strategy_name: str, output_dir: Union[str, Path]):
    """Generates and saves a confusion matrix heatmap using Seaborn."""
    output_path = Path(output_dir) / f"confusion_matrix_{model_id}_{strategy_name.replace(' ', '_')}.png"

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(labels):
        logger.error(f"Invalid matrix or labels for confusion matrix: {model_id}/{strategy_name}. Matrix shape: {matrix.shape}, Labels: {len(labels)}. Skipping plot.")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", # Use a standard cmap
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 10})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_title(f'Confusion Matrix: {model_id} ({strategy_name})', fontsize=12, loc='left', fontweight='bold') # add loc and fontweight
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        plt.close() # Close the figure
        logger.info(f"Saved confusion matrix: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix {output_path}: {e}")
        plt.close()


def generate_all_charts(all_model_run_summaries: dict, charts_output_dir: Union[str, Path]):
    """
    Generates all comparison charts based on the summary data using Matplotlib/Seaborn.

    Args:
        all_model_run_summaries: Dictionary containing metric summaries for different models and strategies.
        charts_output_dir: The directory path to save the generated charts.
    """
    output_dir = Path(charts_output_dir) # Ensure it's a Path object
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_plot_data(all_model_run_summaries)

    if df is None or df.empty:
        logger.warning("Plotting skipped: No valid data prepared from summaries.")
        ui_utils.print_warning("Plotting skipped: No valid data prepared from summaries.")
        return

    # --- Generate Matplotlib/Seaborn Plots ---
    logger.info("Generating plots using Matplotlib/Seaborn...")

    # Define key metrics
    primary_metric = 'accuracy'
    latency_metric = 'average_latency_ms'
    cost_metric = 'total_cost'

    # 1. Key Strategy Comparison Plots (Bar)
    key_metrics_for_strategy_comparison = [primary_metric, 'f1_score', latency_metric, cost_metric]
    for metric in key_metrics_for_strategy_comparison:
        logger.info(f"Generating strategy comparison plot for: {metric}")
        _plot_metric_by_strategy_comparison(df, output_dir, metric)

    # 2. SC-CoT N=3 vs N=5 Comparison (Bar)
    logger.info("Generating SC-CoT N sample comparison plots...")
    sc_metrics = [primary_metric, latency_metric, cost_metric]
    for metric in sc_metrics:
        _plot_sc_comparison(df, output_dir, metric=metric)


    # 3. Trade-off Scatter Plots
    logger.info("Generating trade-off scatter plots...")
    _plot_scatter_tradeoff(df, output_dir, metric_y=primary_metric, metric_x=latency_metric) # Accuracy vs Latency
    _plot_scatter_tradeoff(df, output_dir, metric_y=primary_metric, metric_x=cost_metric)    # Accuracy vs Cost
    _plot_scatter_tradeoff(df, output_dir, metric_y=latency_metric, metric_x=cost_metric)   # Latency vs Cost


    # 4. Total Time (Latency) Comparison (Bar)
    logger.info("Generating average latency comparison plot...")
    _plot_total_time_comparison(df, output_dir)


    # 5. Per-Strategy Metric Comparison (Bar)
    logger.info("Generating per-strategy metric comparison plots...")
    all_strategies = df['Strategy'].unique()
    metrics_per_strategy = [primary_metric, 'f1_score', latency_metric, cost_metric, 'total_tokens']
    for strategy in all_strategies:
         logger.debug(f"Generating plots for strategy: {strategy}")
         # Filter metrics actually available for this strategy in the dataframe
         available_metrics_for_strat = df[(df['Strategy'] == strategy) & (df['Metric'].isin(metrics_per_strategy))]['Metric'].unique()
         if available_metrics_for_strat.size > 0:
             _plot_metric_comparison_for_strategy(df, strategy, list(available_metrics_for_strat), output_dir)
         else:
             logger.debug(f"No relevant metrics found for strategy '{strategy}' to plot per-strategy comparison.")


    # 6. Confusion Matrices (Heatmap - already converted)
    logger.info("Generating confusion matrices...")
    for model_id, strategies in all_model_run_summaries.items():
        for strategy_name, results in strategies.items():
            if isinstance(results, dict) and 'confusion_matrix' in results and 'labels' in results:
                matrix = results['confusion_matrix']
                labels = results['labels']
                if matrix and labels: # Check if matrix and labels are not empty
                     _plot_confusion_matrix(matrix, labels, model_id, strategy_name, output_dir)
                else:
                     logger.warning(f"Skipping confusion matrix for {model_id}/{strategy_name}: Empty matrix or labels.")
            # else: # Don't log for every strategy that *doesn't* have a matrix
            #     logger.debug(f"Confusion matrix data not found for {model_id}/{strategy_name}. Skipping matrix plot.")

    logger.info("Finished generating Matplotlib/Seaborn plots.")

    # --- TODO: Convert remaining Plotly functions or remove them ---
    # The following plots still use Plotly and will retain the old style or fail if Plotly is removed.
    # They need to be converted to Matplotlib/Seaborn or removed.

    # Example: Placeholder for future conversion
    # def _plot_sc_comparison_seaborn(...): ... # CONVERTED
    # def _plot_time_vs_accuracy_seaborn(...): ... # CONVERTED via _plot_scatter_tradeoff
    # def _plot_accuracy_vs_cost_seaborn(...): ... # CONVERTED via _plot_scatter_tradeoff
    # def _plot_latency_vs_cost_seaborn(...): ... # CONVERTED via _plot_scatter_tradeoff
    # ... etc ... # Other plots seem to be handled now.












