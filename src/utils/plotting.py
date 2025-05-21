"""
Functions for generating comparison charts for model performance.
"""
import logging
import os
import pandas as pd
import numpy as np 
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
    "axes.edgecolor": "#333333", 
    "axes.linewidth": 1.2, 
    "axes.titlepad": 15, 
    "figure.facecolor": "white", 
    "savefig.facecolor": "white", 
    "xtick.direction": "out", 
    "ytick.direction": "out", 
    "xtick.major.size": 5, 
    "ytick.major.size": 5, 
    "xtick.major.width": 1.2, 
    "ytick.major.width": 1.2, 
    "xtick.minor.size": 3,  
    "ytick.minor.size": 3,  
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.bottom": True, 
    "ytick.left": True,   
})



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
    if "Self-Consistency CoT" in strategy_name:
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

def _plot_metric_by_strategy_comparison(df: pd.DataFrame, output_dir: Union[str, Path], metric: str):
    """Generates grouped bar chart comparing key strategies for each model using Seaborn."""
    metric_df = df[df['Metric'] == metric].copy()
    if metric_df.empty:
        logger.info(f"Skipping '{metric}' by strategy comparison plot: No data found for this metric.")
        return

    
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
    
    title = f'{metric_title} Comparison Across Strategies and Models'

    
    num_models = df_comp['Model'].nunique()
    num_strategies = df_comp['Strategy'].nunique()
    
    
    
    plt.figure(figsize=(max(10, num_models * 2), 6 + num_strategies * 0.5))


    
    
    current_palette = INSPIRED_PALETTE[:num_strategies] if num_strategies > 0 else INSPIRED_PALETTE[:1]

    
    ax = sns.barplot(data=df_comp, x='Model', y='Value', hue='Strategy', palette=current_palette)

    
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
                height + (ax.get_ylim()[1] * 0.01), 
                label_text,
                ha='center', 
                va='bottom',
                fontsize=plt.rcParams["font.size"] * 0.8)

    ax.set_xlabel("Model") 
    ax.set_ylabel(metric_title) 
    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"], pad=plt.rcParams["axes.titlepad"], loc='left', fontweight='bold')
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.set_axisbelow(True)
    if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall'] or \
       'rate' in metric.lower() or 'percentage' in metric.lower() or 'score' in metric.lower():
        current_max_val = df_comp['Value'].max() if not df_comp.empty else 1.0
        current_min_val = df_comp['Value'].min() if not df_comp.empty else 0.0
        plot_min_y = 0 if current_min_val >= 0 else current_min_val * 1.1 
        if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:
            plot_max_y = max(1.05, current_max_val * 1.05 if current_max_val > 0 else 1.05)
        else:
            plot_max_y = current_max_val * 1.1 if current_max_val > 0 else 0.1 
            if current_max_val == 0 and current_min_val == 0 : plot_max_y = 0.1   
        ax.set_ylim(bottom=plot_min_y, top=plot_max_y)
    else:
        if not df_comp.empty and df_comp['Value'].min() >= 0:
            ax.set_ylim(bottom=0, top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None)
        elif not df_comp.empty:
            ax.set_ylim(top=df_comp['Value'].max() * 1.1 if df_comp['Value'].max() > 0 else None) 

    
    plt.legend(title='Strategy', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) 
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, trim=False) 
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
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
    df_comp = metric_df[metric_df['strategy_type'] == 'SC-CoT'].copy()
    comparable_params = df_comp[df_comp['strategy_param'].str.contains(r'N=\d+', regex=True)]['strategy_param'].unique()
    if len(comparable_params) < 2:
        logger.info(f"Skipping SC '{metric}' comparison plot: Need results for at least two different N values (e.g., N=3 and N=5). Found: {comparable_params}")
        return
    df_comp = df_comp[df_comp['strategy_param'].isin(comparable_params)].copy()
    metric_title = metric.replace('_', ' ').title()
    title = f'Self-Consistency CoT {metric_title}: Comparison by Samples (N)'
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_comp, x='base_model_id', y='Value', hue='strategy_param',
                     errorbar=None) 
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
    ax.set_xlabel("Base Model")
    ax.set_ylabel(metric_title)
    ax.set_title(title, loc='left', fontweight='bold') 
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(bottom=0)
    if metric == 'accuracy' or metric == 'f1_score' or metric == 'precision' or metric == 'recall':
         plt.ylim(top=max(1.05, df_comp['Value'].max() * 1.1 if not df_comp.empty else 1.05))
    else:
         plt.ylim(top=df_comp['Value'].max() * 1.15 if not df_comp.empty else None)
    ax.legend(title='Samples (N)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    filename_base = f'comparison_sc_{metric}_n_samples'
    output_path = Path(output_dir) / f'{filename_base}.png'
    try:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()

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

    df_plot['strategy_type'] = df_plot['Strategy'].apply(_get_strategy_type)
    df_plot['base_model_id'] = df_plot['Model']
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
        s=100, 
        alpha=0.8,
        edgecolor='k', 
        linewidth=0.5
    )

    ax.set_xlabel(metric_x_title)
    ax.set_ylabel(metric_y_title)
    ax.set_title(title, loc='left', fontweight='bold')
    if metric_y in ['accuracy', 'f1_score', 'precision', 'recall']:
        min_y = df_plot[metric_y].min()
        max_y = df_plot[metric_y].max()
        ax.set_ylim(bottom=min_y * 0.95 if min_y > 0 else -0.05,
                    top=max(1.0, max_y * 1.05) if max_y < 1 else max_y * 1.05)
    
    ax.legend(title='Legend (Model / Strategy)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
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

def _plot_total_time_comparison(df: pd.DataFrame, output_dir: Union[str, Path]):
    """Plots a comparison of average latency across models and strategies using Seaborn."""
    time_df = df[df['Metric'] == 'average_latency_ms'].copy()
    if time_df.empty:
        logger.warning("No average latency data (average_latency_ms) found to plot time comparison.")
        return

    plt.figure(figsize=(12, 7))
    num_models = time_df['Model'].nunique()
    models = sorted(time_df['Model'].unique()) 
    
    ax = sns.barplot(data=time_df, x='Strategy', y='Value', hue='Model', hue_order=models,
                     dodge=True, errorbar=None) 

    plt.title('Average Latency Comparison Across Strategies and Models', loc='left', fontweight='bold') 
    plt.ylabel('Average Latency (ms)') 
    plt.xlabel('Strategy') 
    plt.xticks(rotation=30, ha='right') 
    plt.yticks() 

    legend = ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    for container in ax.containers:
        labels = [f'{v:.0f}' if v >= 1 else f'{v:.3f}' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', padding=3,
                     fontsize=plt.rcParams['xtick.labelsize'] - 1) 
    ax.set_ylim(bottom=0)
    sns.despine() 
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 

    output_path = Path(output_dir) / "average_latency_comparison.png"
    try:
        plt.savefig(output_path, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved plot: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot {output_path}: {e}")
        plt.close()

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

    models = sorted(strategy_df_full['Model'].unique()) 

    for metric in metrics_to_plot:
        metric_df = strategy_df_full[strategy_df_full['Metric'] == metric]
        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric}' in strategy '{strategy_name}'. Skipping plot.")
            continue

        plt.figure(figsize=(max(6, num_models * 1.5), 5)) 

        ax = sns.barplot(data=metric_df, x='Model', y='Value', order=models,
                         hue='Model', legend=False, 
                         errorbar=None) 

        metric_title = metric.replace('_', ' ').title()
        
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
        
        plt.title(plot_title, loc='left', fontweight='bold', wrap=True) 
        plt.ylabel(ylabel) 
        plt.xlabel('Model') 
        plt.xticks(rotation=45, ha="right", fontsize=plt.rcParams["xtick.labelsize"] * 0.9) 
        plt.yticks() 
        
        for container in ax.containers:
            labels = [label_fmt.format(v) for v in container.datavalues]
            ax.bar_label(container, labels=labels,
                         fontsize=plt.rcParams['xtick.labelsize'] -1, padding=3) 
 
        if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            ax.set_ylim(bottom=0, top=max(1.05, metric_df['Value'].max() * 1.1))
        elif metric_df['Value'].min() >= 0:
             ax.set_ylim(bottom=0, top=metric_df['Value'].max() * 1.15 if metric_df['Value'].max() > 0 else 0.1)

        if metric_df['Value'].max() == 0:
             ax.set_ylim(bottom=-0.001, top=0.01)
             ax.set_yticks([0])
        
        sns.despine()
        plt.tight_layout()

        safe_strategy_name = strategy_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace('/', '')
        output_path = Path(output_dir) / f"{safe_strategy_name}_strategy_{metric}_comparison.png"
        try:
            plt.savefig(output_path, bbox_inches='tight') 
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {output_path}: {e}")
            plt.close()

def _plot_confusion_matrix(matrix: Union[np.ndarray, List[List[int]]], labels: List[str], model_id: str, strategy_name: str, output_dir: Union[str, Path]):
    """Generates and saves a confusion matrix heatmap using Seaborn."""
    output_path = Path(output_dir) / f"confusion_matrix_{model_id}_{strategy_name.replace(' ', '_')}.png"

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(labels):
        logger.error(f"Invalid matrix or labels for confusion matrix: {model_id}/{strategy_name}. Matrix shape: {matrix.shape}, Labels: {len(labels)}. Skipping plot.")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 10})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_title(f'Confusion Matrix: {model_id} ({strategy_name})', fontsize=12, loc='left', fontweight='bold') 
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        plt.close() 
        logger.info(f"Saved confusion matrix: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix {output_path}: {e}")
        plt.close()

def save_summary_metrics_to_csv(all_model_run_summaries: dict, results_output_dir: Union[str, Path]):
    """
    Saves the aggregated summary of all metrics to a CSV file.

    Args:
        all_model_run_summaries: Dictionary containing summaries of model runs.
        results_output_dir: Path to the directory where the CSV should be saved.
    """
    if not all_model_run_summaries:
        logger.warning("No model run summaries provided. Skipping CSV generation.")
        return

    summary_data_list = []
    for base_model_id, strategies_data in all_model_run_summaries.items():
        if not isinstance(strategies_data, dict):
            logger.warning(f"Expected dict for strategies_data for model '{base_model_id}', got {type(strategies_data)}. Skipping.")
            continue
            
        for strategy_name_full, metrics in strategies_data.items():
            if not isinstance(metrics, dict):
                logger.warning(f"Expected dict for metrics for model '{base_model_id}' strategy '{strategy_name_full}', got {type(metrics)}. Skipping.")
                continue

            if "error" in metrics:
                logger.info(f"Skipping errored run for CSV: {base_model_id} - {strategy_name_full}")
                continue

            run_id = f"{base_model_id}__{strategy_name_full.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}"
            model_id_full = metrics.get("config_id_used", base_model_id) 
            
            strategy_type = _get_strategy_type(strategy_name_full)
            strategy_param_val = _get_strategy_param(strategy_name_full)
            
            display_name_parts = [base_model_id, strategy_type]
            if strategy_param_val:
                display_name_parts.append(strategy_param_val)
            display_name = " - ".join(display_name_parts)

            row = {
                "run_id": run_id,
                "model_id_full": model_id_full,
                "base_model_id": base_model_id,
                "strategy_name": strategy_name_full,
                "strategy_type": strategy_type,
                "strategy_param": strategy_param_val,
                "display_name": display_name,
                "accuracy": metrics.get("accuracy"),
                "f1_score": metrics.get("f1_score"),
                "precision": metrics.get("precision"), 
                "recall": metrics.get("recall"),       
                "avg_time_per_question_ms": metrics.get("average_latency_ms"),
                "total_run_time_s": metrics.get("total_run_time_s"),
                "total_input_tokens": metrics.get("total_input_tokens"), 
                "total_output_tokens": metrics.get("total_output_tokens"),
                "total_tokens": metrics.get("total_tokens"),             
                "avg_answer_length": metrics.get("avg_answer_length"),   
                "total_cost": metrics.get("total_cost"),
                "num_processed": metrics.get("num_processed") 
            }
            summary_data_list.append(row)

    if not summary_data_list:
        logger.warning("No valid data to save to CSV after processing summaries.")
        return

    summary_df = pd.DataFrame(summary_data_list)
    
    ordered_columns = [
        "run_id", "model_id_full", "base_model_id", "display_name", 
        "strategy_name", "strategy_type", "strategy_param",
        "accuracy", "f1_score", "precision", "recall", 
        "avg_time_per_question_ms", "total_run_time_s",
        "total_input_tokens", "total_output_tokens", "total_tokens",
        "avg_answer_length", "total_cost", "num_processed"
    ]
    for col in ordered_columns:
        if col not in summary_df.columns:
            summary_df[col] = pd.NA 
            
    summary_df = summary_df[ordered_columns]
    results_dir = Path(results_output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = results_dir / "all_runs_summary_metrics.csv"
    
    try:
        summary_df.to_csv(csv_file_path, index=False, encoding='utf-8')
        logger.info(f"Successfully saved summary metrics to {csv_file_path}")
        ui_utils.print_success(f"Aggregated metrics summary saved to: {csv_file_path}")
    except Exception as e:
        logger.error(f"Failed to save summary metrics CSV to {csv_file_path}: {e}", exc_info=True)
        ui_utils.print_error(f"Failed to save summary metrics CSV: {e}")

def generate_reasoning_vs_non_reasoning_accuracy_plot(all_model_runs_summary: Dict[str, Any], output_dir: Union[str, Path]):
    """Generates a violin plot comparing accuracy between reasoning and non-reasoning models."""
    logger.info("Generating Reasoning vs Non-Reasoning Accuracy Distribution plot...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data = []
    for model_id, strategies in all_model_runs_summary.items():
        for strategy_name, metrics in strategies.items():
            accuracy = metrics.get('accuracy')
            model_type = metrics.get('model_type') 
            if accuracy is not None and model_type is not None:
                plot_data.append({
                    'Model ID': model_id,
                    'Strategy': strategy_name,
                    'Accuracy': float(accuracy),
                    'Model Type': model_type
                })
    
    if not plot_data:
        logger.warning("No data found for Reasoning vs Non-Reasoning Accuracy plot. Skipping.")
        return

    df = pd.DataFrame(plot_data)
    
    if df.empty or 'Model Type' not in df.columns or df['Model Type'].nunique() < 2:
        logger.warning("Not enough distinct model types or data to generate Reasoning vs Non-Reasoning Accuracy plot. Skipping.")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x='Model Type', y='Accuracy', data=df, palette=INSPIRED_PALETTE[:2], inner='quartile')
    sns.stripplot(x='Model Type', y='Accuracy', data=df, color='.25', size=3, alpha=0.5, ax=ax) 

    ax.set_title('Accuracy Distribution: Reasoning vs. Non-Reasoning Models',
                 fontsize=plt.rcParams["axes.titlesize"], pad=plt.rcParams["axes.titlepad"], loc='left', fontweight='bold')
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Accuracy")
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05) 

    plt.tight_layout()
    try:
        filename_png = output_dir / "reasoning_vs_non_reasoning_accuracy.png"
        plt.savefig(filename_png, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Reasoning vs Non-Reasoning Accuracy plot to {filename_png}")
        
    except Exception as e:
        logger.error(f"Error saving Reasoning vs Non-Reasoning Accuracy plot: {e}")
    finally:
        plt.close()

def generate_all_charts(all_model_run_summaries: dict, charts_output_dir: Union[str, Path]):
    """
    Generates all comparison charts based on the provided model run summaries.
    """
    if not all_model_run_summaries:
        logger.warning("No model run summaries available. Skipping chart generation.")
        return

    
    charts_output_dir = Path(charts_output_dir)
    charts_output_dir.mkdir(parents=True, exist_ok=True)

    
    try:
        logger.info("Attempting to generate Reasoning vs Non-Reasoning Accuracy plot from generate_all_charts.")
        generate_reasoning_vs_non_reasoning_accuracy_plot(all_model_run_summaries, charts_output_dir)
    except Exception as e:
        logger.error(f"Error calling generate_reasoning_vs_non_reasoning_accuracy_plot from generate_all_charts: {e}", exc_info=True)

    
    df = _prepare_plot_data(all_model_run_summaries)

    if df is None or df.empty:
        logger.warning("Plotting skipped: No valid data prepared from summaries.")
        ui_utils.print_warning("Plotting skipped: No valid data prepared from summaries.")
        return
    
    logger.info("Generating plots using Matplotlib/Seaborn...")
    primary_metric = 'accuracy'
    latency_metric = 'average_latency_ms'
    cost_metric = 'total_cost'
    key_metrics_for_strategy_comparison = [primary_metric, 'f1_score', latency_metric, cost_metric]
    for metric in key_metrics_for_strategy_comparison:
        logger.info(f"Generating strategy comparison plot for: {metric}")
        _plot_metric_by_strategy_comparison(df, charts_output_dir, metric)

    logger.info("Generating SC-CoT N sample comparison plots...")
    sc_metrics = [primary_metric, latency_metric, cost_metric]
    for metric in sc_metrics:
        _plot_sc_comparison(df, charts_output_dir, metric=metric)
    
    logger.info("Generating trade-off scatter plots...")
    _plot_scatter_tradeoff(df, charts_output_dir, metric_y=primary_metric, metric_x=latency_metric) 
    _plot_scatter_tradeoff(df, charts_output_dir, metric_y=primary_metric, metric_x=cost_metric)    
    _plot_scatter_tradeoff(df, charts_output_dir, metric_y=latency_metric, metric_x=cost_metric)   

    logger.info("Generating average latency comparison plot...")
    _plot_total_time_comparison(df, charts_output_dir)    
    logger.info("Generating per-strategy metric comparison plots...")
    all_strategies = df['Strategy'].unique()
    metrics_per_strategy = [primary_metric, 'f1_score', latency_metric, cost_metric, 'total_tokens']
    for strategy in all_strategies:
         logger.debug(f"Generating plots for strategy: {strategy}")
         
         available_metrics_for_strat = df[(df['Strategy'] == strategy) & (df['Metric'].isin(metrics_per_strategy))]['Metric'].unique()
         if available_metrics_for_strat.size > 0:
             _plot_metric_comparison_for_strategy(df, strategy, list(available_metrics_for_strat), charts_output_dir)
         else:
             logger.debug(f"No relevant metrics found for strategy '{strategy}' to plot per-strategy comparison.")

    logger.info("Generating confusion matrices...")
    for model_id, strategies in all_model_run_summaries.items():
        for strategy_name, results in strategies.items():
            if isinstance(results, dict) and 'confusion_matrix' in results and 'labels' in results:
                matrix = results['confusion_matrix']
                labels = results['labels']
                if matrix and labels: 
                     _plot_confusion_matrix(matrix, labels, model_id, strategy_name, charts_output_dir)
                else:
                     logger.warning(f"Skipping confusion matrix for {model_id}/{strategy_name}: Empty matrix or labels.")
    
    logger.info("Finished generating Matplotlib/Seaborn plots.")

    
    
    

    
    
    
    
    
    












