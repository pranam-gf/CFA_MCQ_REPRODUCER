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
import plotly.express as px
import plotly.graph_objects as go

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
    "axes.facecolor": "#f0f0f5",  
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

MODEL_TYPE_COLOR_MAP = {
    "Reasoning": "#4C72B0",    
    "Non-Reasoning": "#DD8452" 
}

def _prepare_plotly_summary_dataframe(all_model_runs_summary_dict: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for model_id_key, strategies in all_model_runs_summary_dict.items():
        if not isinstance(strategies, dict):
            logger.warning(f"Strategies for model '{model_id_key}' is not a dict. Skipping.")
            continue
        for strategy_name_key, metrics in strategies.items():
            if not isinstance(metrics, dict):
                logger.warning(f"Metrics for model '{model_id_key}', strategy '{strategy_name_key}' is not a dict. Skipping.")
                continue
            
            record = {'Model': metrics.get('Model', model_id_key), 
                      'Strategy': metrics.get('Strategy', strategy_name_key), 
                      'display_name': metrics.get('config_id_used', f"{model_id_key} ({strategy_name_key}")}
            record.update(metrics)
            records.append(record)
    
    if not records:
        logger.warning("No records found to build Plotly summary DataFrame.")
        return pd.DataFrame()
        
    summary_df = pd.DataFrame(records)   
    numeric_cols_map = {
        'accuracy': 'Accuracy_Plotly', 
        'average_latency_ms': 'AvgLatencyMS_Plotly',
        'total_cost': 'TotalCost_Plotly',
        'total_output_tokens': 'TotalOutputTokens_Plotly',
        'avg_answer_length': 'AvgAnsLen_Plotly',
        'num_processed': 'NumProcessed_Plotly',
        'total_run_time_s': 'TotalRunTimeS_Plotly'
    }

    for original_col, plotly_col_name in numeric_cols_map.items():
        if original_col in summary_df.columns:
            summary_df[plotly_col_name] = pd.to_numeric(summary_df[original_col], errors='coerce')

    if 'model_type' in summary_df.columns:
        summary_df['ModelType_Plotly'] = summary_df['model_type']
            
    if 'Strategy' in summary_df.columns:
        summary_df['Strategy_Plotly'] = summary_df['Strategy']
    
    return summary_df

def identify_pareto_optimal(df: pd.DataFrame, x_col: str, y_col: str, x_lower_is_better: bool, y_higher_is_better: bool) -> List[bool]:
    is_pareto = [True] * len(df)    
    df_filtered = df.dropna(subset=[x_col, y_col])
    if df_filtered.empty:
        return [False] * len(df) 
        
    df_values = df_filtered[[x_col, y_col]].values 
    original_indices = df_filtered.index 
    temp_is_pareto = [True] * len(df_values)

    for i in range(len(df_values)):
        for j in range(len(df_values)):
            if i == j:
                continue
  
            x_dominates = (df_values[j, 0] < df_values[i, 0] if x_lower_is_better else df_values[j, 0] > df_values[i, 0])
            y_dominates = (df_values[j, 1] > df_values[i, 1] if y_higher_is_better else df_values[j, 1] < df_values[i, 1])
            
            x_equal_or_better = (df_values[j, 0] <= df_values[i, 0] if x_lower_is_better else df_values[j, 0] >= df_values[i, 0])
            y_equal_or_better = (df_values[j, 1] >= df_values[i, 1] if y_higher_is_better else df_values[j, 1] <= df_values[i, 1])

            if x_equal_or_better and y_equal_or_better and (x_dominates or y_dominates):
                temp_is_pareto[i] = False
                break
    
    
    final_is_pareto = [False] * len(df)
    for k, original_idx in enumerate(original_indices):
        if temp_is_pareto[k]:
            final_is_pareto[original_idx] = True
            
    return final_is_pareto

def plot_pareto_frontier(summary_df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str, output_path: Path, lower_is_better_x: bool, higher_is_better_y: bool):
    if summary_df.empty or x_col not in summary_df.columns or y_col not in summary_df.columns:
        logger.warning(f"Missing data for Pareto plot: {title}. X: {x_col}, Y: {y_col}. Cols: {summary_df.columns}")
        return

    df_plot = summary_df.copy().dropna(subset=[x_col, y_col])
    if df_plot.empty:
        logger.warning(f"No data after dropna for Pareto plot: {title}")
        return
    
    df_plot['is_pareto'] = identify_pareto_optimal(df_plot, x_col, y_col, lower_is_better_x, higher_is_better_y)
    pareto_df = df_plot[df_plot['is_pareto']].sort_values(by=[x_col if lower_is_better_x else y_col], ascending=[True if lower_is_better_x else False])

    
    df_plot['label_text'] = df_plot.apply(lambda row: row['display_name'] if row['is_pareto'] else None, axis=1)

    fig = px.scatter(df_plot, x=x_col, y=y_col,
                     text='label_text',  
                     hover_data=['Model', 'Strategy', x_col, y_col, 'Accuracy_Plotly', 'TotalCost_Plotly', 'AvgLatencyMS_Plotly'],
                     title=title,
                     labels={x_col: x_label, y_col: y_label},
                     color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_traces(marker=dict(size=8), textfont_size=6)

    if not pareto_df.empty:
        fig.add_trace(go.Scatter(x=pareto_df[x_col], y=pareto_df[y_col],
                                 mode='lines+markers',
                                 marker=dict(color='red', size=10, symbol='star'),
                                 line=dict(color='red', width=2),
                                 name='Pareto Frontier',
                                 text=pareto_df['display_name'], 
                                 hoverinfo='text'))
    
    fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Legend')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.with_suffix(".html"))
    try:
        fig.write_image(output_path.with_suffix(".png"), scale=2, width=1200, height=800)
    except Exception as e:
        logger.warning(f"Could not save PNG for {title} (Kaleido might be missing or misconfigured): {e}")
    logger.info(f"Generated Pareto frontier plot: {output_path.name}")


def plot_cost_latency_accuracy_bubble(summary_df: pd.DataFrame, output_path: Path):
    bubble_metrics_map = {
        'total_cost': 'TotalCost_Plotly',
        'average_latency_ms': 'AvgLatencyMS_Plotly',
        'accuracy': 'Accuracy_Plotly',
        'model_type': 'ModelType_Plotly' 
    }
    required_plotly_cols = list(bubble_metrics_map.values()) + ['display_name']

    if summary_df.empty or not all(col in summary_df.columns for col in required_plotly_cols):
        missing_cols = [col for col in required_plotly_cols if col not in summary_df.columns]
        logger.warning(f"Missing data for Cost-Latency-Accuracy bubble plot. Required: {required_plotly_cols}. Missing: {missing_cols}. Available: {summary_df.columns}")
        return
    
    df_plot = summary_df.dropna(subset=[bubble_metrics_map['total_cost'], bubble_metrics_map['average_latency_ms'], bubble_metrics_map['accuracy']])
    if df_plot.empty:
        logger.warning("No data for Cost-Latency-Accuracy bubble plot after dropna.")
        return

    
    df_plot['bubble_size_accuracy'] = df_plot[bubble_metrics_map['accuracy']] * 50 + 5 

    fig = px.scatter(df_plot, x=bubble_metrics_map['total_cost'], y=bubble_metrics_map['average_latency_ms'],
                     size='bubble_size_accuracy', color=bubble_metrics_map['model_type'],
                     color_discrete_map=MODEL_TYPE_COLOR_MAP, 
                     hover_name='display_name',
                     hover_data=['Model', 'Strategy', bubble_metrics_map['accuracy'], bubble_metrics_map['total_cost'], 'AvgLatencyMS_Plotly'],
                     size_max=40,
                     title='Cost vs. Latency (Bubble Size: Accuracy, Color: Model Type)',
                     labels={bubble_metrics_map['total_cost']: 'Total Cost ($)', 
                             bubble_metrics_map['average_latency_ms']: 'Average Latency per Question (ms)',
                             bubble_metrics_map['accuracy']: 'Accuracy',
                             bubble_metrics_map['model_type']: 'Model Type'})
    
    fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Model Type')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.with_suffix(".html"))
    try:
        fig.write_image(output_path.with_suffix(".png"), scale=2, width=1200, height=800)
    except Exception as e:
        logger.warning(f"Could not save PNG for Cost-Latency-Accuracy bubble plot (Kaleido might be missing): {e}")
    logger.info(f"Generated Cost-Latency-Accuracy bubble plot: {output_path.name}")


def generate_research_specific_charts(all_model_runs_summary_dict: Dict[str, Any], output_dir: Path):
    """Generates research-specific plots like Pareto frontiers and bubble charts."""
    logger.info(f"Generating research-specific charts in {output_dir}...")
    summary_df_plotly = _prepare_plotly_summary_dataframe(all_model_runs_summary_dict)

    if summary_df_plotly.empty:
        logger.warning("No data available to generate research-specific charts (Plotly DataFrame empty).")
        return

    
    cost_col = 'TotalCost_Plotly'
    accuracy_col = 'Accuracy_Plotly'
    if cost_col in summary_df_plotly.columns and accuracy_col in summary_df_plotly.columns:
        plot_pareto_frontier(summary_df_plotly, x_col=cost_col, y_col=accuracy_col, 
                             x_label='Total Cost ($)', y_label='Accuracy',
                             title='Pareto Frontier: Accuracy vs. Cost',
                             output_path=output_dir / "pareto_accuracy_vs_cost",
                             lower_is_better_x=True, higher_is_better_y=True)
    else:
        logger.warning(f"Skipping Pareto Accuracy vs Cost plot due to missing '{cost_col}' or '{accuracy_col}' columns.")

    
    latency_col_plotly = 'AvgLatencyMS_Plotly'
    if latency_col_plotly in summary_df_plotly.columns and accuracy_col in summary_df_plotly.columns:
        plot_pareto_frontier(summary_df_plotly, x_col=latency_col_plotly, y_col=accuracy_col,
                             x_label='Average Latency per Question (ms)', y_label='Accuracy',
                             title='Pareto Frontier: Accuracy vs. Latency',
                             output_path=output_dir / "pareto_accuracy_vs_latency",
                             lower_is_better_x=True, higher_is_better_y=True)
    else:
        logger.warning(f"Skipping Pareto Accuracy vs Latency plot due to missing '{latency_col_plotly}' or '{accuracy_col}' columns.")

    
    model_type_col_plotly = 'ModelType_Plotly'
    if all(col in summary_df_plotly.columns for col in [cost_col, latency_col_plotly, accuracy_col, model_type_col_plotly]):
         plot_cost_latency_accuracy_bubble(summary_df_plotly, output_dir / "cost_latency_accuracy_bubble")
    else:
        logger.warning(f"Skipping Cost-Latency-Accuracy bubble plot due to missing one or more required columns: '{cost_col}', '{latency_col_plotly}', '{accuracy_col}', '{model_type_col_plotly}'.")
    
    logger.info(f"Finished generating research-specific charts in {output_dir}.")




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
    """Generates grouped horizontal bar chart comparing SC-CoT N=3 vs N=5 using Seaborn."""
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
    
    
    
    model_order = sorted(df_comp['base_model_id'].unique())
    
    metric_title = metric.replace('_', ' ').title()
    
    num_models = len(model_order)
    plt.figure(figsize=(10, max(6, num_models * 0.5))) 

    
    
    
    
    ax = sns.barplot(data=df_comp, y='base_model_id', x='Value', hue='strategy_param',
                     order=model_order, 
                     orient='h', errorbar=None)
    
    
    for p in ax.patches:
        width = p.get_width()
        try:
            if width == 0: label_text = '0'
            elif abs(width) < 0.001 and width != 0: label_text = f'{width:.2e}'
            elif metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall'] : label_text = f'{width:.3f}'
            elif abs(width) < 100 : label_text = f'{width:.2f}'
            else: label_text = f'{int(round(width))}'
            
            if metric == 'total_cost': label_text = f'${width:,.3f}'
            elif metric == 'average_latency_ms': label_text = f'{width:.0f}'
            

        except TypeError:
            label_text = "N/A"
        
        ax.text(width + (ax.get_xlim()[1] * 0.01), 
                p.get_y() + p.get_height() / 2.,
                label_text,
                va='center',
                ha='left', 
                fontsize=plt.rcParams["ytick.labelsize"] * 0.8)

    ax.set_ylabel("Base Model") 
    ax.set_xlabel(metric_title) 
    
    if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:
        ax.set_xlim(left=0, right=max(1.0, df_comp['Value'].max() * 1.1 if not df_comp.empty else 1.0))
    elif not df_comp.empty and df_comp['Value'].min() >= 0:
        ax.set_xlim(left=0, right=df_comp['Value'].max() * 1.15 if df_comp['Value'].max() > 0 else 0.1)
    
    if not df_comp.empty and df_comp['Value'].max() == 0 and df_comp['Value'].min() == 0:
        ax.set_xlim(left=0, right=0.1)
        ax.set_xticks([0])
        
    ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray')
    ax.set_axisbelow(True)

    ax.legend(title='Samples (N)', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    sns.despine(left=True, bottom=False)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 
    
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
    """
    Generates a scatter plot for trade-off analysis (e.g., Accuracy vs. Latency).
    Uses 'Model' for hue and 'strategy_type' for style.
    """
    logger.info(f"Plotting scatter trade-off: {metric_y} vs {metric_x}")

    
    df_filtered = df.copy()
    df_filtered = df_filtered.dropna(subset=[metric_x, metric_y, 'Model', 'strategy_type'])
    if df_filtered.empty:
        logger.warning(f"No data available for scatter plot {metric_y} vs {metric_x} after filtering NaNs. Skipping.")
        return

    
    
    unique_models = df_filtered['Model'].unique()
    palette_map = {model: INSPIRED_PALETTE[i % len(INSPIRED_PALETTE)] for i, model in enumerate(unique_models)}
    
    strategy_markers = {"Default": "o", "SC-CoT": "X", "Self-Discover": "s", "Other": "D"} 
    df_filtered['strategy_type'] = df_filtered['strategy_type'].apply(lambda x: x if x in strategy_markers else "Other")


    metric_y_title = metric_y.replace('_', ' ').title()
    metric_x_title = metric_x.replace('_', ' ').title()
    if 'cost' in metric_x.lower(): 
        metric_x_title += " ($)"
    if 'latency_ms' in metric_x.lower() or 'time_s' in metric_x.lower() : 
         metric_x_title = metric_x.replace('_', ' ').replace('ms', '(ms)').replace('s', '(s)').title()


    
    legend_outside = len(unique_models) > 5 

    plt.figure(figsize=(12, 8 if legend_outside else 7)) 

    
    ax = sns.scatterplot(
        data=df_filtered, 
        x=metric_x, 
        y=metric_y, 
        hue='Model', 
        style='strategy_type', 
        s=150,  
        palette=palette_map, 
        markers=strategy_markers, 
        alpha=0.85, 
        legend="auto" 
    )

    
    
    
    
    
    
    

    ax.set_xlabel(metric_x_title)
    ax.set_ylabel(metric_y_title)
    

    
    handles, labels = ax.get_legend_handles_labels()
    
    
    valid_legend_items = [(h, l) for h, l in zip(handles, labels) if hasattr(h, 'get_label') and h.get_label() and l]
    if valid_legend_items:
        handles, labels = zip(*valid_legend_items)

        legend_fontsize = plt.rcParams.get("legend.fontsize", 10) * 0.85
        legend_title_fontsize = plt.rcParams.get("legend.fontsize", 10) * 0.9
        
        
        
        current_legend = ax.get_legend()
        if current_legend:
            current_legend.remove() 

        ax.legend(handles=handles, labels=labels,
                  title=None, 
                  bbox_to_anchor=(1.03, 1), 
                  loc='upper left', 
                  borderaxespad=0.,
                  fontsize=legend_fontsize,
                  title_fontsize=legend_title_fontsize, 
                  labelspacing=0.4, 
                  handletextpad=0.5, 
                  markerscale=0.9)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    else:
        if ax.get_legend() is not None: 
            ax.get_legend().remove()
        logger.warning(f"Could not retrieve valid legend handles/labels for {metric_y} vs {metric_x}. Legend may be missing or incomplete.")
        plt.tight_layout()

    sns.despine(ax=ax, trim=True)
    ax.grid(True, linestyle='--', alpha=0.4, color='lightgray')
    
    filename_base = f'tradeoff_{metric_y}_vs_{metric_x}'
    full_path_png = Path(output_dir) / f"{filename_base}.png"
    full_path_html = Path(output_dir) / f"{filename_base}.html" 

    try:
        plt.savefig(full_path_png, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {full_path_png}")
    except Exception as e:
        logger.error(f"Failed to save scatter plot {full_path_png}: {e}")
    
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
    Generates separate horizontal bar charts comparing models for a specific strategy using Seaborn.
    Bars will have a standard color.

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

    models_sorted_by_metric = {} 
    for metric in metrics_to_plot:
        metric_df = strategy_df_full[strategy_df_full['Metric'] == metric].copy()
        if metric_df.empty:
            logger.warning(f"No data found for metric '{metric}' in strategy '{strategy_name}'. Skipping plot.")
            continue      
        sort_ascending = True
        if metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall']:
            sort_ascending = False
        
        models_sorted_by_metric[metric] = metric_df.sort_values(by='Value', ascending=sort_ascending)['Model'].tolist()

        plt.figure(figsize=(7, max(4, num_models * 0.4))) 

        standard_color_palette = [INSPIRED_PALETTE[0]] 
        
        ax = sns.barplot(data=metric_df, y='Model', x='Value', order=models_sorted_by_metric[metric],
                         hue='Model', 
                         palette=standard_color_palette, 
                         legend=False, 
                         orient='h',
                         errorbar=None)

        metric_title = metric.replace('_', ' ').title()
        
        if metric == 'total_cost':
            xlabel = f'{metric_title} ($)'
            label_fmt_str = '${:,.3f}'
        elif metric == 'average_latency_ms':
            xlabel = f'{metric_title} (ms)'
            label_fmt_str = '{:.0f}'
        elif metric == 'total_run_time_s':
            xlabel = f'{metric_title} (s)'
            label_fmt_str = '{:.1f}s'
        elif metric == 'total_tokens':
            xlabel = f'{metric_title} (tokens)'
            label_fmt_str = '{:,.0f}'
        else: 
            xlabel = metric_title
            label_fmt_str = '{:.3f}'
        plot_title = f"{metric_title} for {strategy_name}"
        
        plt.xlabel(xlabel)
        plt.ylabel('Model') 
        plt.yticks(fontsize=plt.rcParams["ytick.labelsize"] * 0.9) 
        plt.xticks(fontsize=plt.rcParams["xtick.labelsize"] * 0.9)
        
        for p in ax.patches:
            width = p.get_width()
            try:
                if width == 0: label_text = '0'
                elif abs(width) < 0.001 and width != 0: label_text = f'{width:.2e}'
                elif metric.lower() in ['accuracy', 'f1_score', 'precision', 'recall'] : label_text = f'{width:.3f}'
                elif abs(width) < 100 : label_text = f'{width:.2f}'
                else: label_text = f'{int(round(width))}'
                
                if metric == 'total_cost': label_text = f'${width:,.3f}'
                elif metric == 'average_latency_ms': label_text = f'{width:.0f}'
                elif metric == 'total_run_time_s': label_text = f'{width:.1f}s'
                elif metric == 'total_tokens': label_text = f'{width:,.0f}'

            except TypeError:
                label_text = "N/A"
            
            ax.text(width + (ax.get_xlim()[1] * 0.01), 
                    p.get_y() + p.get_height() / 2.,
                    label_text,
                    va='center',
                    ha='left', 
                    fontsize=plt.rcParams['ytick.labelsize'] - 2)

        if metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            ax.set_xlim(left=0, right=max(1.0, metric_df['Value'].max() * 1.1) if not metric_df.empty and metric_df['Value'].max() > 0 else 1.0 )
        elif not metric_df.empty and metric_df['Value'].min() >= 0:
             ax.set_xlim(left=0, right=metric_df['Value'].max() * 1.15 if metric_df['Value'].max() > 0 else 0.1)
        
        if not metric_df.empty and metric_df['Value'].max() == 0 and metric_df['Value'].min() == 0 :
             ax.set_xlim(left=-0.001 if metric not in ['accuracy', 'f1_score', 'precision', 'recall'] else 0 , right=0.01)
             ax.set_xticks([0])
        
        ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.7, color='lightgray') 
        ax.set_axisbelow(True)
        sns.despine(left=True, bottom=False) 
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
                annot_kws={"size": 23})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
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
    
    ax = sns.violinplot(x='Model Type', y='Accuracy', data=df, hue='Model Type', 
                        palette=MODEL_TYPE_COLOR_MAP, 
                        inner='quartile', legend=False)
    sns.stripplot(x='Model Type', y='Accuracy', data=df, color='.25', size=3, alpha=0.5, ax=ax) 

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

def generate_strategy_effectiveness_boxplot(
    all_model_runs_summary_dict: Dict[str, Any], 
    output_dir: Path
):
    """Generates a box plot comparing strategy effectiveness by model type (Reasoning vs. Non-Reasoning)."""
    logger.info(f"Generating Strategy Effectiveness by Model Type box plot in {output_dir}...")
    summary_df_plotly = _prepare_plotly_summary_dataframe(all_model_runs_summary_dict)

    required_cols = ['Accuracy_Plotly', 'ModelType_Plotly', 'Strategy_Plotly']
    if summary_df_plotly.empty or not all(col in summary_df_plotly.columns for col in required_cols):
        missing = [col for col in required_cols if col not in summary_df_plotly.columns]
        logger.warning(f"Skipping Strategy Effectiveness box plot. Missing required columns: {missing}. Available: {summary_df_plotly.columns}")
        return

    df_plot = summary_df_plotly.dropna(subset=required_cols)
    if df_plot.empty:
        logger.warning("No data for Strategy Effectiveness box plot after dropna.")
        return

    fig = px.box(df_plot, 
                 x='Strategy_Plotly', 
                 y='Accuracy_Plotly', 
                 color='ModelType_Plotly',
                 color_discrete_map=MODEL_TYPE_COLOR_MAP, 
                 title="Strategy Effectiveness: Accuracy by Model Type",
                 labels={'Strategy_Plotly': "Prompting Strategy", 
                         'Accuracy_Plotly': "Accuracy", 
                         'ModelType_Plotly': "Model Type"},
                 points="all", 
                 notched=False, 
                 color_discrete_sequence=px.colors.qualitative.Plotly) 

    fig.update_layout(
        yaxis_range=[0, 1.05] 
    )
    fig.update_xaxes(categoryorder='array', categoryarray=sorted(df_plot['Strategy_Plotly'].unique())) 

    output_path_base = output_dir / "strategy_effectiveness_by_model_type"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        fig.write_html(output_path_base.with_suffix(".html"))
        fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=1200, height=700)
        logger.info(f"Generated Strategy Effectiveness box plot: {output_path_base.name}")
    except Exception as e:
        logger.warning(f"Could not save Strategy Effectiveness box plot (Kaleido might be missing or misconfigured for PNG): {e}")

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
    
    logger.info("Preparing data for trade-off scatter plots...")
    
    df_pivot = None
    if not df.empty:
        try:
            
            if all(col in df.columns for col in ['Model', 'Strategy', 'Metric', 'Value']):
                df_pivot = df.pivot_table(index=['Model', 'Strategy'], columns='Metric', values='Value').reset_index()
                df_pivot.columns.name = None 
                if 'Strategy' in df_pivot.columns:
                    df_pivot['strategy_type'] = df_pivot['Strategy'].apply(_get_strategy_type)
                else:
                    logger.warning("Pivoted DataFrame for scatter plots is missing 'Strategy' column. Cannot derive 'strategy_type'.")
                    df_pivot['strategy_type'] = 'Unknown' 
            else:
                logger.error("Input DataFrame for pivot_table is missing one or more required columns: 'Model', 'Strategy', 'Metric', 'Value'.")
                df_pivot = pd.DataFrame() 
        except Exception as e:
            logger.error(f"Failed to pivot DataFrame for scatter plots: {e}", exc_info=True)
            df_pivot = pd.DataFrame() 

    if df_pivot is not None and not df_pivot.empty:
        logger.info("Generating trade-off scatter plots...")
        plot_configs = [
            (primary_metric, latency_metric),
            (primary_metric, cost_metric),
            (latency_metric, cost_metric)
        ]
        for metric_y, metric_x in plot_configs:
            required_cols_for_scatter = [metric_y, metric_x, 'Model', 'strategy_type']
            if all(col in df_pivot.columns for col in required_cols_for_scatter):
                _plot_scatter_tradeoff(df_pivot, charts_output_dir, metric_y=metric_y, metric_x=metric_x)
            else:
                missing_cols = [col for col in required_cols_for_scatter if col not in df_pivot.columns]
                logger.warning(f"Skipping scatter plot for {metric_y} vs {metric_x}: Missing required columns in pivoted data: {missing_cols}. Available columns: {df_pivot.columns.tolist()}")
    else:
        logger.warning("Skipping trade-off scatter plots as pivoted data is empty or could not be prepared.")

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

def plot_aggregated_metrics_by_model_type(advanced_analysis_dir: Path, output_dir: Path):
    """Plots aggregated metrics (mean Accuracy, Avg Time/Q, Cost) by Model Type from model_type_aggregate_stats.csv."""
    logger.info(f"Generating Aggregated Metrics by Model Type plot in {output_dir}...")
    stats_file = advanced_analysis_dir / "model_type_aggregate_stats.csv"

    if not stats_file.exists():
        logger.warning(f"Aggregated stats file not found: {stats_file}. Skipping plot.")
        return

    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        logger.error(f"Error reading {stats_file}: {e}")
        return

    if df.empty:
        logger.warning(f"{stats_file} is empty. Skipping plot.")
        return

    
    metrics_to_plot = {
        'Accuracy_mean': 'Mean Accuracy',
        'Avg Time/Q (s)_mean': 'Mean Avg Time/Q (s)',
        'Total Cost ($)_mean': 'Mean Total Cost ($)'
    }
    plot_df_melted_list = []
    for col, name in metrics_to_plot.items():
        if col in df.columns and 'Model Type' in df.columns:
            temp_df = df[['Model Type', col]].copy()
            temp_df.rename(columns={col: 'Value'}, inplace=True)
            temp_df['Metric'] = name
            plot_df_melted_list.append(temp_df)
        else:
            logger.warning(f"Required column '{col}' or 'Model Type' not in {stats_file}. Skipping this metric for the plot.")

    if not plot_df_melted_list:
        logger.warning("No data to plot for aggregated metrics by model type after filtering.")
        return
    
    plot_df_melted = pd.concat(plot_df_melted_list)

    fig = px.bar(plot_df_melted, 
                 x='Metric', 
                 y='Value', 
                 color='Model Type', 
                 color_discrete_map=MODEL_TYPE_COLOR_MAP, 
                 barmode='group',
                 title="Mean Performance Metrics by Model Type",
                 labels={'Value': "Mean Value", 'Metric': "Performance Metric"})

    fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Model Type')
    fig.update_yaxes(rangemode='tozero') 

    output_path_base = output_dir / "mean_metrics_by_model_type"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_html(output_path_base.with_suffix(".html"))
        fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=1000, height=600)
        logger.info(f"Generated Aggregated Metrics by Model Type plot: {output_path_base.name}")
    except Exception as e:
        logger.warning(f"Could not save Aggregated Metrics by Model Type plot: {e}")

def plot_efficiency_distribution_by_model_type(advanced_analysis_dir: Path, output_dir: Path):
    """Plots distribution of efficiency metrics by Model Type from efficiency_metrics.csv."""
    logger.info(f"Generating Efficiency Metrics Distribution by Model Type plot in {output_dir}...")
    stats_file = advanced_analysis_dir / "efficiency_metrics.csv"

    if not stats_file.exists():
        logger.warning(f"Efficiency metrics file not found: {stats_file}. Skipping plot.")
        return
    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        logger.error(f"Error reading {stats_file}: {e}")
        return

    if df.empty or 'Model Type' not in df.columns:
        logger.warning(f"{stats_file} is empty or missing 'Model Type' column. Skipping plot.")
        return

    efficiency_metrics = {
        'Tokens_Per_Second': 'Tokens per Second',
        'Accuracy_Per_Dollar': 'Accuracy per Dollar',
        'Time_Per_Token_s': 'Time per Token (s)'
    }

    for col_name, readable_name in efficiency_metrics.items():
        if col_name not in df.columns:
            logger.warning(f"Efficiency metric column '{col_name}' not found in {stats_file}. Skipping this metric.")
            continue
        
        plot_df = df[['Model Type', col_name]].dropna()
        if plot_df.empty:
            logger.warning(f"No data for metric '{col_name}' after dropna. Skipping this metric.")
            continue

        fig = px.box(plot_df, 
                     x='Model Type', 
                     y=col_name, 
                     color='Model Type',
                     color_discrete_map=MODEL_TYPE_COLOR_MAP, 
                     title=f"Distribution of {readable_name} by Model Type",
                     labels={col_name: readable_name},
                     points="all")
        
        fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Model Type')
        
        

        output_path_base = output_dir / f"{col_name.lower()}_distribution_by_model_type"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_html(output_path_base.with_suffix(".html"))
            fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=800, height=600)
            logger.info(f"Generated {readable_name} distribution plot: {output_path_base.name}")
        except Exception as e:
            logger.warning(f"Could not save {readable_name} distribution plot: {e}")

def plot_efficiency_distribution_by_strategy(advanced_analysis_dir: Path, output_dir: Path):
    """Plots distribution of efficiency metrics by Strategy from efficiency_metrics.csv."""
    logger.info(f"Generating Efficiency Metrics Distribution by Strategy plot in {output_dir}...")
    stats_file = advanced_analysis_dir / "efficiency_metrics.csv"

    if not stats_file.exists():
        logger.warning(f"Efficiency metrics file not found: {stats_file}. Skipping plot.")
        return
    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        logger.error(f"Error reading {stats_file}: {e}")
        return

    if df.empty or 'Strategy' not in df.columns:
        logger.warning(f"{stats_file} is empty or missing 'Strategy' column. Skipping plot.")
        return

    efficiency_metrics = {
        'Tokens_Per_Second': 'Tokens per Second',
        'Accuracy_Per_Dollar': 'Accuracy per Dollar',
        'Time_Per_Token_s': 'Time per Token (s)'
    }

    for col_name, readable_name in efficiency_metrics.items():
        if col_name not in df.columns:
            logger.warning(f"Efficiency metric column '{col_name}' not found in {stats_file}. Skipping this metric.")
            continue
        
        plot_df = df[['Strategy', col_name]].dropna()
        if plot_df.empty:
            logger.warning(f"No data for metric '{col_name}' after dropna. Skipping this metric.")
            continue

        fig = px.box(plot_df, 
                     x='Strategy', 
                     y=col_name, 
                     color='Strategy', 
                     title=f"Distribution of {readable_name} by Strategy",
                     labels={col_name: readable_name, 'Strategy': 'Prompting Strategy'},
                     points="all",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        
        fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Strategy')
        fig.update_xaxes(categoryorder='array', categoryarray=sorted(df['Strategy'].unique()))

        output_path_base = output_dir / f"{col_name.lower()}_distribution_by_strategy"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_html(output_path_base.with_suffix(".html"))
            fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=1000, height=600) 
            logger.info(f"Generated {readable_name} distribution by Strategy plot: {output_path_base.name}")
        except Exception as e:
            logger.warning(f"Could not save {readable_name} distribution by Strategy plot: {e}")

def plot_strategy_aggregate_metrics(advanced_analysis_dir: Path, output_dir: Path):
    """Plots aggregated metrics by Strategy from strategy_aggregate_stats.csv."""
    logger.info(f"Generating Aggregated Metrics by Strategy plot in {output_dir}...")
    stats_file = advanced_analysis_dir / "strategy_aggregate_stats.csv"

    if not stats_file.exists():
        logger.warning(f"Strategy aggregate stats file not found: {stats_file}. Skipping plot.")
        return

    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        logger.error(f"Error reading {stats_file}: {e}")
        return

    if df.empty:
        logger.warning(f"{stats_file} is empty. Skipping plot.")
        return

    metrics_to_plot = {
        'Accuracy_mean': 'Mean Accuracy',
        'Avg Time/Q (s)_mean': 'Mean Avg Time/Q (s)',
        'Total Cost ($)_mean': 'Mean Total Cost ($)'
    }
    
    if 'Strategy' not in df.columns:
        logger.warning(f"'Strategy' column not in {stats_file}. Skipping this plot.")
        return

    plot_df_melted_list = []
    for col, name in metrics_to_plot.items():
        if col in df.columns:
            temp_df = df[['Strategy', col]].copy()
            temp_df.rename(columns={col: 'Value'}, inplace=True)
            temp_df['Metric'] = name
            plot_df_melted_list.append(temp_df)
        else:
            logger.warning(f"Required column '{col}' not in {stats_file}. Skipping this metric for the strategy aggregate plot.")

    if not plot_df_melted_list:
        logger.warning("No data to plot for aggregated metrics by strategy after filtering.")
        return
    
    plot_df_melted = pd.concat(plot_df_melted_list)

    fig = px.bar(plot_df_melted, 
                 x='Metric', 
                 y='Value', 
                 color='Strategy', 
                 barmode='group',
                 title="Mean Performance Metrics by Strategy",
                 labels={'Value': "Mean Value", 'Metric': "Performance Metric"},
                 color_discrete_sequence=px.colors.qualitative.Plotly) 

    fig.update_layout(title_x=0.5, title_font_size=16, legend_title_text='Strategy')
    fig.update_yaxes(rangemode='tozero')
    
    fig.update_xaxes(tickangle=-45)

    output_path_base = output_dir / "mean_metrics_by_strategy"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_html(output_path_base.with_suffix(".html"))
        fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=1000, height=600)
        logger.info(f"Generated Aggregated Metrics by Strategy plot: {output_path_base.name}")
    except Exception as e:
        logger.warning(f"Could not save Aggregated Metrics by Strategy plot: {e}")

def plot_model_aggregate_metrics(advanced_analysis_dir: Path, output_dir: Path):
    """Plots key aggregated metrics by Model from model_aggregate_stats.csv."""
    logger.info(f"Generating Aggregated Metrics by Model plots in {output_dir}...")
    stats_file = advanced_analysis_dir / "model_aggregate_stats.csv"

    if not stats_file.exists():
        logger.warning(f"Model aggregate stats file not found: {stats_file}. Skipping plots.")
        return

    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        logger.error(f"Error reading {stats_file}: {e}")
        return

    if df.empty:
        logger.warning(f"{stats_file} is empty. Skipping plots.")
        return

    
    if 'Model' not in df.columns:
        logger.warning(f"'Model' column not found in {stats_file}. Skipping model aggregate plots.")
        return

    metrics_to_plot = {
        'Accuracy_mean': 'Mean Accuracy',
        'Avg Time/Q (s)_mean': 'Mean Avg Time/Q (s)',
        'Total Cost ($)_mean': 'Mean Total Cost ($)',
        'Total API Response Time (s)_mean': 'Mean Total API Response Time (s)',
        'Total Output Tokens_mean': 'Mean Total Output Tokens'
    }

    for col_name, readable_name in metrics_to_plot.items():
        if col_name not in df.columns:
            logger.warning(f"Metric column '{col_name}' not in {stats_file}. Skipping plot for this metric.")
            continue

        plot_df = df[['Model', col_name]].copy().dropna(subset=[col_name])
        if plot_df.empty:
            logger.warning(f"No data for metric '{col_name}' after dropna. Skipping plot.")
            continue
        
        
        
        sort_ascending = True
        if col_name in ['Accuracy_mean', 'Total Output Tokens_mean']:
            sort_ascending = False
        
        plot_df = plot_df.sort_values(by=col_name, ascending=sort_ascending)

        fig = px.bar(plot_df, 
                     y='Model', 
                     x=col_name, 
                     orientation='h',
                     labels={col_name: readable_name, 'Model': 'Model'},
                     color_discrete_sequence=[INSPIRED_PALETTE[0]]) 
        
        fig.update_layout(yaxis_title="Model")
        fig.update_xaxes(rangemode='tozero') 

        output_path_base = output_dir / f"model_aggregate_{col_name.lower()}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_html(output_path_base.with_suffix(".html"))
            fig.write_image(output_path_base.with_suffix(".png"), scale=2, width=1000, height=max(600, len(plot_df) * 30)) 
            logger.info(f"Generated {readable_name} by Model plot: {output_path_base.name}")
        except Exception as e:
            logger.warning(f"Could not save {readable_name} by Model plot: {e}")

def plot_performance_tiers(all_model_runs_summary_dict: Dict[str, Any], output_dir: Path):
    """
    Generates a bar chart showing the distribution of models across performance tiers
    based on their accuracy with the 'Default (Single Pass)' strategy.
    """
    logger.info("Generating performance tier distribution plot...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_accuracies = []
    default_strategy_name = "Default (Single Pass)" 

    for model_id, strategies in all_model_runs_summary_dict.items():
        if default_strategy_name in strategies:
            accuracy = strategies[default_strategy_name].get('accuracy')
            if accuracy is not None and isinstance(accuracy, (float, int)):
                model_accuracies.append({'Model': model_id, 'Accuracy': accuracy * 100}) # Convert to percentage
            else:
                logger.debug(f"Model '{model_id}' with strategy '{default_strategy_name}' has no valid accuracy: {strategies[default_strategy_name].get('accuracy')}")
        else:
            logger.debug(f"Strategy '{default_strategy_name}' not found for model '{model_id}'. Available: {list(strategies.keys())}")

    if not model_accuracies:
        logger.warning(f"No models found with accuracy data for the '{default_strategy_name}' strategy. Cannot generate performance tier plot.")
        ui_utils.print_warning(f"No models found with accuracy data for the '{default_strategy_name}' strategy. Performance tier plot will not be generated.")
        return

    df = pd.DataFrame(model_accuracies)

    bins_revised = [-1, 49.999, 59.999, 69.999, 77.001, 101] 
    labels_revised = [
        "Sub-50% (<50%)", 
        "Developing (50-59.9%)", 
        "Mid-Tier (60-69.9%)", 
        "Frontier (70-77%)",
        "Above Frontier (>77%)" 
    ]

    df['Performance Tier'] = pd.cut(df['Accuracy'], bins=bins_revised, labels=labels_revised, right=True, include_lowest=True)

    tier_counts = df['Performance Tier'].value_counts().reindex(labels_revised).fillna(0)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("viridis", n_colors=len(labels_revised))
    bars = sns.barplot(x=tier_counts.index, y=tier_counts.values, palette=palette)
    
    plt.title('Model Performance Distribution by Accuracy Tiers (Default Strategy)', fontsize=16, pad=20)
    plt.xlabel('Performance Tier', fontsize=14, labelpad=15)
    plt.ylabel('Number of Models', fontsize=14, labelpad=15)
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(trim=True)

    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.15, 
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, color='black')

    plt.tight_layout(pad=1.5)
    
    plot_filename = output_dir / "model_performance_tiers_distribution.png"
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logger.info(f"Successfully saved performance tier plot to {plot_filename}")
        ui_utils.print_success(f"Performance tier distribution plot saved to {plot_filename}")
    except Exception as e:
        logger.error(f"Error saving performance tier plot to {plot_filename}: {e}", exc_info=True)
        ui_utils.print_error(f"Could not save performance tier plot: {e}")
    finally:
        plt.close()


    
    
    

    
    
    
    
    
    
























