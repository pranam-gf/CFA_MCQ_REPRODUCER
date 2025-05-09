"""
Functions for generating comparison charts for model performance.
"""
import logging
import os
import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import config

logger = logging.getLogger(__name__)

def generate_all_charts(all_model_results: dict, charts_output_dir: str | os.PathLike):
    """
    Generates and saves all comparison charts based on the aggregated results.
    Focuses on accuracy, response time, and token counts.

    Args:
        all_model_results: Dictionary containing results for all models.
        charts_output_dir: Path to the directory to save charts.
    """
    if not all_model_results:
        logger.warning("No model results provided. Skipping chart generation.")
        return

    logger.info(f"Generating comparison charts in {charts_output_dir}...")
    os.makedirs(charts_output_dir, exist_ok=True)

    

    plot_data = {k: v for k, v in all_model_results.items() if v.get('metrics') is not None and 'error' not in v}
    if not plot_data:
        logger.warning("No successful model results found to generate charts.")
        return

    model_names = list(plot_data.keys())
    
    
    accuracies = [v['metrics'].get('accuracy', 0.0) for v in plot_data.values()]
    
    times = [v.get('avg_response_time') if v.get('avg_response_time') is not None else np.nan for v in plot_data.values()]
    out_tokens = [v.get('total_output_tokens') if v.get('total_output_tokens') is not None else 0 for v in plot_data.values()]
    

    
    df_accuracy = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
    fig_accuracy = px.bar(
        df_accuracy, x='Model', y='Accuracy', title='Model Accuracy Comparison',
        labels={'Accuracy': 'Accuracy Score', 'Model': 'Model Name'}, color='Accuracy',
        color_continuous_scale='viridis', text='Accuracy'
    )
    fig_accuracy.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_accuracy.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1.05], height=600, width=max(900, len(model_names) * 50), title_font_size=18)
    fig_accuracy.write_image(os.path.join(charts_output_dir, 'model_comparison_accuracy.png'))
    fig_accuracy.write_html(os.path.join(charts_output_dir, 'model_comparison_accuracy.html'))
    logger.info(f"Saved accuracy chart to {charts_output_dir}")
    

    
    
    df_time = pd.DataFrame({'Model': model_names, 'Response_Time_s': times})
    
    fig_time = px.bar(
        df_time, x='Model', y='Response_Time_s', title='Average Response Time Comparison',
        labels={'Response_Time_s': 'Avg Response Time (s)', 'Model': 'Model Name'}, color='Response_Time_s',
        color_continuous_scale='blues', text='Response_Time_s'
    )
    fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig_time.update_layout(xaxis_tickangle=-45, height=600, width=max(900, len(model_names) * 50), title_font_size=18)
    fig_time.write_image(os.path.join(charts_output_dir, 'model_comparison_response_time.png'))
    fig_time.write_html(os.path.join(charts_output_dir, 'model_comparison_response_time.html'))
    logger.info(f"Saved response time chart to {charts_output_dir}")

    
    if any(t > 0 for t in out_tokens if t is not None): 
        df_tokens = pd.DataFrame({'Model': model_names, 'Output_Tokens': out_tokens})
        fig_tokens = px.bar(
            df_tokens, x='Model', y='Output_Tokens', title='Total Output Tokens Generated',
            labels={'Output_Tokens': 'Total Output Tokens', 'Model': 'Model Name'}, color='Output_Tokens',
            color_continuous_scale='cividis', text='Output_Tokens'
        )
        fig_tokens.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_tokens.update_layout(xaxis_tickangle=-45, height=600, width=max(900, len(model_names) * 50), title_font_size=18)
        fig_tokens.write_image(os.path.join(charts_output_dir, 'model_comparison_output_tokens.png'))
        fig_tokens.write_html(os.path.join(charts_output_dir, 'model_comparison_output_tokens.html'))
        logger.info(f"Saved output tokens chart to {charts_output_dir}")
    else:
        logger.info("No output token data to plot or all are zero.")

    
    
    valid_indices_for_combined = [i for i, t in enumerate(times) if pd.notna(t)]
    if valid_indices_for_combined:
        combined_model_names = [model_names[i] for i in valid_indices_for_combined]
        combined_accuracies = [accuracies[i] for i in valid_indices_for_combined]
        combined_times = [times[i] for i in valid_indices_for_combined]
        

        df_combined = pd.DataFrame({
            'Model': combined_model_names,
            'Accuracy': combined_accuracies,
            'Response_Time_s': combined_times,
            
        })

        fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combined.add_trace(
            go.Bar(x=df_combined['Model'], y=df_combined['Accuracy'], name='Accuracy', 
                   marker_color='rgba(55, 83, 109, 0.7)', text=df_combined['Accuracy'], 
                   texttemplate='%{text:.3f}', textposition='outside'),
            secondary_y=False,
        )
        fig_combined.add_trace(
            go.Scatter(x=df_combined['Model'], y=df_combined['Response_Time_s'], name='Response Time (s)', 
                       marker_color='rgba(219, 64, 82, 0.7)', mode='lines+markers', 
                       text=df_combined['Response_Time_s'], texttemplate='%{text:.2f}s', textposition="top center"),
            secondary_y=True,
        )
        fig_combined.update_layout(
            title_text="Combined Model Performance: Accuracy & Response Time", title_font_size=18,
            xaxis_tickangle=-45, height=700, width=max(1000, len(combined_model_names) * 60), xaxis_title="Model",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), barmode='group'
        )
        fig_combined.update_yaxes(title_text="Accuracy", range=[0, 1.05], secondary_y=False)
        fig_combined.update_yaxes(title_text="Response Time (s)", secondary_y=True)
        fig_combined.write_image(os.path.join(charts_output_dir, 'model_combined_metrics.png'))
        fig_combined.write_html(os.path.join(charts_output_dir, 'model_combined_metrics.html'))
        logger.info(f"Saved combined metrics chart to {charts_output_dir}")
    else:
        logger.warning("Not enough data for combined metrics chart (possibly all response times are NaN).")

    logger.info(f"All charts saved to {charts_output_dir}")












