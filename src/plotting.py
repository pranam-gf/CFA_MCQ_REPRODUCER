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
import plotly.io as pio 
logger = logging.getLogger(__name__)
GOODFIN_PRIMARY_COLOR = '#333333' 
GOODFIN_SECONDARY_COLOR = '#666666'  
GOODFIN_ACCENT_COLOR = '#999999'    
GOODFIN_BACKGROUND_COLOR = '#EDEDED'  
GOODFIN_TEXT_COLOR = '#333333'      
GOODFIN_MUTED_TEXT_COLOR = '#666666'  
GOODFIN_GRID_COLOR = '#CCCCCC'      
GOODFIN_ERROR_COLOR = '#FF4444'     
GOODFIN_FONT_FAMILY = "'Helvetica Neue', Helvetica, Arial, sans-serif"
GOODFIN_TITLE_FONT_SIZE = 20
GOODFIN_SUBTITLE_FONT_SIZE = 16
GOODFIN_AXIS_LABEL_FONT_SIZE = 14
GOODFIN_TICK_LABEL_FONT_SIZE = 12
GOODFIN_LEGEND_FONT_SIZE = 12
GOODFIN_ANNOTATION_FONT_SIZE = 10
LOGO_ASPECT_RATIO = 32 / 138
LOGO_SIZEX_PAPER = 0.30  
LOGO_SIZEY_PAPER = LOGO_SIZEX_PAPER * LOGO_ASPECT_RATIO

goodfin_template = go.layout.Template()

goodfin_template.layout = go.Layout(
    font=dict(
        family=GOODFIN_FONT_FAMILY,
        size=GOODFIN_AXIS_LABEL_FONT_SIZE,
        color=GOODFIN_TEXT_COLOR
    ),
    title=dict(
        font_size=GOODFIN_TITLE_FONT_SIZE,
        x=0.5,
        xanchor='center',
        pad=dict(t=20, b=20)
    ),
    paper_bgcolor=GOODFIN_BACKGROUND_COLOR,
    plot_bgcolor=GOODFIN_BACKGROUND_COLOR,
    xaxis=dict(
        gridcolor=GOODFIN_GRID_COLOR,
        linecolor=GOODFIN_MUTED_TEXT_COLOR, 
        zerolinecolor=GOODFIN_GRID_COLOR,
        title_font_size=GOODFIN_AXIS_LABEL_FONT_SIZE,
        tickfont_size=GOODFIN_TICK_LABEL_FONT_SIZE,
        showgrid=True,
        gridwidth=1,
        zeroline=False, 
        showline=True, 
        mirror=True 
    ),
    yaxis=dict(
        gridcolor=GOODFIN_GRID_COLOR,
        linecolor=GOODFIN_MUTED_TEXT_COLOR,
        zerolinecolor=GOODFIN_GRID_COLOR,
        title_font_size=GOODFIN_AXIS_LABEL_FONT_SIZE,
        tickfont_size=GOODFIN_TICK_LABEL_FONT_SIZE,
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        mirror=True
    ),
    legend=dict(
        font_size=GOODFIN_LEGEND_FONT_SIZE,
        bgcolor='rgba(255,255,255,0.7)', 
        bordercolor=GOODFIN_GRID_COLOR,
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1
    ),
    colorway=[GOODFIN_PRIMARY_COLOR, GOODFIN_SECONDARY_COLOR, GOODFIN_ACCENT_COLOR, GOODFIN_ERROR_COLOR],
    
    margin=dict(l=80, r=50, t=100, b=80) 
)


pio.templates.default = goodfin_template

def generate_all_charts(all_model_run_summaries: dict, charts_output_dir: str | os.PathLike):
    """
    Generates and saves all comparison charts based on the aggregated results from various runs.
    Focuses on accuracy, response time, and token counts.

    Args:
        all_model_run_summaries: Dictionary containing summary results for all model_strategy runs.
                               Keys are typically like 'model_id__strategy_name'.
        charts_output_dir: Path to the directory to save charts.
    """
    logo_base64_data_uri = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTM4IiBoZWlnaHQ9IjMyIiB2aWV3Qm94PSIwIDAgMTM4IDMyIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8ZyBjbGlwLXBhdGg9InVybCgjY2xpcDBfNDAwM181MjA4KSI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTMyLjU4NSAyNC43MzM5VjE0Ljc3ODRDMTMyLjU4NSAxMS43OTM0IDEzMS4yMzIgMTAuNDE4MSAxMjguMDI0IDEwLjQxODFDMTI0Ljg4MSAxMC40MTgxIDEyMi4zNCAxMi4yNjgzIDEyMS41NjYgMTQuMjU4NFYyNC43MzM5SDExNy4wMjJWNy4xMDI3MUgxMjEuNTY2VjEyLjE3ODRDMTIyLjQ5MSA5LjMyOTk3IDEyNS4zMDEgNi42OTIxNiAxMjkuNjMyIDYuNjkyMTZDMTMzLjkwNSA2LjY5MjE2IDEzNy4xMjkgOC45NjAzMSAxMzcuMTI5IDEzLjk1NzZWMjQuNzMzOUgxMzIuNTg1Wk0xMDkuODIyIDExLjQxNjNIMTAwLjgxMVYyNC43MzM5SDk2LjI2NjdWMTEuNDE2M0g5Mi4zNDc5VjcuOTIzMjNIOTYuMjY2N1Y2LjE0ODk5Qzk2LjI2NjcgMS44OTA5MyA5OC42MDQ0IDAuMDA1NzQwMjQgMTAyLjk3MyAwLjAwNTc0MDI0SDExNC4zNjZWMy43MzE2MkgxMDQuMDY2QzEwMS45OSAzLjY4Mjk4IDEwMC44MTEgNC4yODA1OCAxMDAuODExIDYuNzAzNDJWNy45MjMxSDExNC4zNjZWMjQuNzM0MUgxMDkuODIyVjExLjQxNjNaTTg1LjU3MzMgMjAuMTg3N1YxOS4wNzU2Qzg0LjYxMDggMjIuMjc2OCA4Mi4wNzY4IDI1LjEzMzIgNzcuNDQ2OSAyNS4xMzMyQzcyLjE4MDMgMjUuMTMzMiA2OC4xNTggMjEuNDQ0OCA2OC4xNTggMTUuOTQyNkM2OC4xNTggMTAuMzc0NCA3Mi4wNjQ3IDYuNjkyMjMgNzcuNDQ2OSA2LjY5MjIzQzgyLjEyMzggNi42OTIyMyA4NC42MjIyIDguOTg0OTEgODUuNTczMyAxMi4zMzM1Vi0wLjAwMTk1MzEySDkwLjExNzRWMjQuNzMzOUg4NS44OTQ2Qzg1Ljg5NDYgMjQuNzMzOSA4NS41NzMzIDIyLjI5IDg1LjU3MzMgMjAuMTg3N1pNNzguOTEwNCAxMC4yNjcyQzc1LjAyMjIgMTAuMjY3MiA3Mi44MDA2IDEyLjUzNCA3Mi44MDA2IDE1Ljk0MjZDNzIuODAwNiAxOS4zNDAzIDc1LjIzODEgMjEuNTY5MSA3OC45MTA0IDIxLjU2OTFDODMuMDI2IDIxLjU2OTEgODUuNTczMyAxOS4zMTM3IDg1LjU3MzMgMTUuOTQyNkM4NS41NzMzIDEyLjMxNjUgODIuOTg2NiAxMC4yNjcyIDc4LjkxMDQgMTAuMjY3MlpNNTYuMDI1MSAyNS4xNDQyQzQ5LjQ1MTggMjUuMTQ0MiA0NC45NDkxIDIxLjk0NDIgNDQuOTQ5MSAxNS45NTM2QzQ0Ljk0OTEgOS45OTg5NyA0OS40NTE4IDYuNjkyMjMgNTYuMDI1MSA2LjY5MjIzQzYyLjU1MjYgNi42OTIyMyA2Ny4wMDkyIDkuOTk4OTcgNjcuMDA5MiAxNS45NTM2QzY3LjAwOTUgMjEuOTQ0MiA2Mi41NTI2IDI1LjE0NDIgNTYuMDI1MSAyNS4xNDQyWk01Ni4wMjUxIDEwLjIzNDNDNTIuNDI3IDEwLjIzNDMgNDkuNTQ3OSAxMi4zMDE0IDQ5LjU0NzkgMTUuOTUzN0M0OS41NDc5IDE5LjYwNiA1Mi4zODEyIDIxLjYwMjUgNTYuMDI1MSAyMS42MDI1QzU5LjU3NzEgMjEuNjAyNSA2Mi40MTA3IDE5LjYwNjYgNjIuNDEwNyAxNS45NTM3QzYyLjQxMDcgMTIuMzM2NyA1OS42MjMyIDEwLjIzNDIgNTYuMDI1MSAxMC4yMzQyVjEwLjIzNDNaTTMyLjgxNjQgMjUuMTQ0MkMyNi4yNDMgMjUuMTQ0MiAyMS43NDA0IDIxLjk0NDIgMjEuNzQwNCAxNS45NTM2QzIxLjc0MDQgOS45OTg5NyAyNi4yNDMgNi42OTIyMyAzMi44MTY0IDYuNjkyMjNDMzkuMzQzOSA2LjY5MjIzIDQzLjgwMDUgOS45OTg5NyA0My44MDA1IDE1Ljk1MzZDNDMuODAwNSAyMS45NDQyIDM5LjM0MzkgMjUuMTQ0MiAzMi44MTY0IDI1LjE0NDJaTTMyLjgxNjQgMTAuMjM0M0MyOS4yMTgzIDEwLjIzNDMgMjYuMzM5MiAxMi4zMDE0IDI2LjMzOTIgMTUuOTUzN0MyNi4zMzkyIDE5LjYwNiAyOS4xNzI1IDIxLjYwMjUgMzIuODE2NCAyMS42MDI1QzM2LjM2ODMgMjEuNjAyNSAzOS4yMDE5IDE5LjYwNjYgMzkuMjAxOSAxNS45NTM3QzM5LjIwMTkgMTIuMzM2NyAzNi40MTQ0IDEwLjIzNDIgMzIuODE2NCAxMC4yMzQyVjEwLjIzNDNaTTE0LjI2NDggOC43OTU0NEMxNi42Njg1IDkuOTk0NzggMTguOTI3MiAxMS44NjE5IDE4LjkyNzIgMTQuNjg2MUMxOC45MjcyIDE5LjMzNjEgMTUuMjk2NiAyMi4wMjk3IDkuNTAwNiAyMi4wMjk3QzYuNTQ3NDYgMjIuMTMxNSA1LjE1MDc1IDIyLjQxNDMgNS4xNTA3NSAyMy41OTM4QzUuMTUwNzUgMjQuMzE2MyA1LjcwMzE0IDI0LjczNDIgNi43NzEyIDI0LjczNDJMMTIuNDgwMiAyNC43MzQ5QzE3LjE0MTggMjQuNzM0OSAxOS44ODU2IDI2LjA4MzQgMTkuODg1NiAzMC4zNDE2VjMxLjk5OEgxNS4zNDE4VjMxLjE2MTZDMTUuMzQxOCAyOC45ODM1IDEzLjg2MjggMjguMjQzMiAxMS40OTY5IDI4LjI0MzJINi4wNTkxNEMxLjg0NDc4IDI4LjI0MzIgMC41NTc3OTMgMjYuNzk4MyAwLjU1Nzc5MyAyNC44ODIxQzAuNTU3NzkzIDIyLjQ0ODMgMi44MTgxMiAyMS42MjM2IDUuODUxODQgMjEuNTk4QzIuMjM3OTkgMjAuNjU2NCAtMC4wMDAzNjYyMTEgMTguMjAxMiAtMC4wMDAzNjYyMTEgMTQuNDY0MkMtMC4wMDAzNjYyMTEgOS43MzMwMSAzLjY2Njk0IDYuNjkyMSA5LjUwMDYgNi42OTIxQzEwLjI5NzkgNi42OTE0NyAxMS4wOTMyIDYuNzcyMTMgMTEuODc0MSA2LjkzMjgyTDIxLjMyMzUgNi45MDQzM1YxMC40MzUxQzE4Ljc5ODIgMTAuNDk2NyAxNi4xMzcyIDkuNjA0MDcgMTQuMjY0OCA4Ljc5NTQ0Wk05LjQ5OTUyIDEwLjI1OTVDNi40NjQ5MiAxMC4yNTk1IDQuNDA5MjMgMTEuODMwOCA0LjQwOTIzIDE0LjUwMjRDNC40MDkyMyAxNy4xNzQgNi40MjgyOCAxOC43MTI0IDkuNDk5NTIgMTguNzEyNEMxMi40OTcyIDE4LjcxMjQgMTQuNTExOSAxNy4xNzQ0IDE0LjUxMTkgMTQuNTAyNEMxNC41MTE5IDExLjg1ODcgMTIuNTMzOSAxMC4yNTk1IDkuNDk5NTIgMTAuMjU5NVoiIGZpbGw9IiMzNzMzMzgiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80MDAzXzUyMDgiPgo8cmVjdCB3aWR0aD0iMTM4IiBoZWlnaHQ9IjMyIiBmaWxsPSJ3aGl0ZSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPg=="    
    logo_base_config = dict(
        source=logo_base64_data_uri,
        xref="paper", yref="paper",
        sizex=LOGO_SIZEX_PAPER,
        sizey=LOGO_SIZEY_PAPER,
        xanchor="right", yanchor="top",
        layer="above"            
    )

    if not all_model_run_summaries:
        logger.warning("No model run summaries provided. Skipping chart generation.")
        return

    logger.info(f"Generating comparison charts in {charts_output_dir}...")
    os.makedirs(charts_output_dir, exist_ok=True)

    
    plot_data_items = []
    for run_id, summary in all_model_run_summaries.items():
        if summary.get('metrics') is not None and 'error' not in summary:
            model_name = summary.get('model', run_id.split('__')[0] if '__' in run_id else run_id)
            strategy_name = summary.get('strategy', run_id.split('__')[1] if '__' in run_id else 'default')
            display_name = f"{model_name} ({strategy_name})"
            plot_data_items.append({
                'run_id': run_id,
                'display_name': display_name,
                'accuracy': summary['metrics'].get('accuracy', 0.0),
                'avg_time_per_question': summary.get('avg_time_per_question'),
                'total_output_tokens': summary.get('total_output_tokens')
            })

    if not plot_data_items:
        logger.warning("No successful model runs with metrics found to generate charts.")
        return
    df_plot = pd.DataFrame(plot_data_items)
    df_plot = df_plot.sort_values(by='accuracy', ascending=False)
    figure_height_accuracy = max(600, len(df_plot) * 50)
    figure_width_accuracy = max(900, len(df_plot) * 80)    
    margins = goodfin_template.layout.margin
    margin_l = margins.l if margins and margins.l is not None else 0
    margin_r = margins.r if margins and margins.r is not None else 0
    margin_t = margins.t if margins and margins.t is not None else 0
    margin_b = margins.b if margins and margins.b is not None else 0

    plot_area_width_accuracy = figure_width_accuracy - margin_l - margin_r
    plot_area_height_accuracy = figure_height_accuracy - margin_t - margin_b
    
    logo_x_accuracy, logo_y_accuracy = 1.0, 1.0 
    if plot_area_width_accuracy > 0 and plot_area_height_accuracy > 0:
        logo_x_accuracy = 1 + (margin_r / plot_area_width_accuracy)
        logo_y_accuracy = 1 + (margin_t / plot_area_height_accuracy)
    else:
        logger.warning("Plot area non-positive for accuracy chart. Defaulting logo position.")
        
    accuracy_logo_properties = {**logo_base_config, "x": logo_x_accuracy, "y": logo_y_accuracy}

    fig_accuracy = px.bar(
        df_plot, x='display_name', y='accuracy', title='<b>Model-Strategy Accuracy Comparison</b>', 
        labels={'accuracy': 'Accuracy Score', 'display_name': 'Model (Strategy)'},
        text='accuracy'
    )
    fig_accuracy.update_traces(
        texttemplate='%{text:.3f}', textposition='outside',
        marker_color=GOODFIN_PRIMARY_COLOR, 
        marker_line_color='black', 
        marker_line_width=1.5, opacity=0.8
    )
    fig_accuracy.update_layout(
        xaxis_tickangle=-45, yaxis_range=[0, max(1.05, df_plot['accuracy'].max() * 1.1 if not df_plot.empty else 1.05)],
        height=figure_height_accuracy, width=figure_width_accuracy,
        yaxis_title="Accuracy Score", xaxis_title="Model (Strategy)",
        images=[accuracy_logo_properties]
    )
    fig_accuracy.write_image(os.path.join(charts_output_dir, 'model_strategy_comparison_accuracy.png'), scale=2) 
    fig_accuracy.write_html(os.path.join(charts_output_dir, 'model_strategy_comparison_accuracy.html'))
    logger.info(f"Saved accuracy chart to {charts_output_dir}")

    df_time_plot = df_plot.dropna(subset=['avg_time_per_question'])
    if not df_time_plot.empty:
        figure_height_time = max(600, len(df_time_plot) * 50)
        figure_width_time = max(900, len(df_time_plot) * 80)

        plot_area_width_time = figure_width_time - margin_l - margin_r
        plot_area_height_time = figure_height_time - margin_t - margin_b

        logo_x_time, logo_y_time = 1.0, 1.0
        if plot_area_width_time > 0 and plot_area_height_time > 0:
            logo_x_time = 1 + (margin_r / plot_area_width_time)
            logo_y_time = 1 + (margin_t / plot_area_height_time)
        else:
            logger.warning("Plot area non-positive for time chart. Defaulting logo position.")
        
        time_logo_properties = {**logo_base_config, "x": logo_x_time, "y": logo_y_time}

        fig_time = px.bar(
            df_time_plot, x='display_name', y='avg_time_per_question', title='<b>Average Processing Time per Question</b>',
            labels={'avg_time_per_question': 'Avg Time/Q (s)', 'display_name': 'Model (Strategy)'},
            text='avg_time_per_question'
        )
        fig_time.update_traces(
            texttemplate='%{text:.2f}s', textposition='outside',
            marker_color=GOODFIN_PRIMARY_COLOR, 
            marker_line_color='black', 
            marker_line_width=1.5, opacity=0.8
        )
        fig_time.update_layout(
            xaxis_tickangle=-45, 
            height=figure_height_time, width=figure_width_time,
            yaxis_title="Average Time per Question (s)", xaxis_title="Model (Strategy)",
            yaxis_range=[0, df_time_plot['avg_time_per_question'].max() * 1.15 if not df_time_plot.empty else 10],
            images=[time_logo_properties]
        )
        fig_time.write_image(os.path.join(charts_output_dir, 'model_strategy_comparison_response_time.png'), scale=2)
        fig_time.write_html(os.path.join(charts_output_dir, 'model_strategy_comparison_response_time.html'))
        logger.info(f"Saved response time chart to {charts_output_dir}")
    else:
        logger.info("No valid response time data to plot or all are NaN.")

    
    df_token_plot = df_plot[df_plot['total_output_tokens'] > 0].dropna(subset=['total_output_tokens'])
    if not df_token_plot.empty:
        figure_height_tokens = max(600, len(df_token_plot) * 50)
        figure_width_tokens = max(900, len(df_token_plot) * 80)

        plot_area_width_tokens = figure_width_tokens - margin_l - margin_r
        plot_area_height_tokens = figure_height_tokens - margin_t - margin_b
        
        logo_x_tokens, logo_y_tokens = 1.0, 1.0
        if plot_area_width_tokens > 0 and plot_area_height_tokens > 0:
            logo_x_tokens = 1 + (margin_r / plot_area_width_tokens)
            logo_y_tokens = 1 + (margin_t / plot_area_height_tokens)
        else:
            logger.warning("Plot area non-positive for tokens chart. Defaulting logo position.")

        tokens_logo_properties = {**logo_base_config, "x": logo_x_tokens, "y": logo_y_tokens}
        
        fig_tokens = px.bar(
            df_token_plot, x='display_name', y='total_output_tokens', title='<b>Total Output Tokens Generated</b>',
            labels={'total_output_tokens': 'Total Output Tokens', 'display_name': 'Model (Strategy)'},
            text='total_output_tokens'
        )
        fig_tokens.update_traces(
            texttemplate='%{text:,.0f}', textposition='outside', 
            marker_color=GOODFIN_PRIMARY_COLOR, 
            marker_line_color='black', 
            marker_line_width=1.5, opacity=0.8
        )
        fig_tokens.update_layout(
            xaxis_tickangle=-45,
            height=figure_height_tokens, width=figure_width_tokens,
            yaxis_title="Total Output Tokens", xaxis_title="Model (Strategy)",
            yaxis_range=[0, df_token_plot['total_output_tokens'].max() * 1.15 if not df_token_plot.empty else 1000],
            images=[tokens_logo_properties]
        )
        fig_tokens.write_image(os.path.join(charts_output_dir, 'model_strategy_comparison_output_tokens.png'), scale=2)
        fig_tokens.write_html(os.path.join(charts_output_dir, 'model_strategy_comparison_output_tokens.html'))
        logger.info(f"Saved output tokens chart to {charts_output_dir}")
    else:
        logger.info("No output token data to plot or all are zero/NaN.")

    
    df_combined_plot = df_plot.dropna(subset=['avg_time_per_question', 'accuracy'])
    if not df_combined_plot.empty:
        figure_height_combined = max(700, len(df_combined_plot) * 60)
        figure_width_combined = max(1000, len(df_combined_plot) * 90)

        plot_area_width_combined = figure_width_combined - margin_l - margin_r
        plot_area_height_combined = figure_height_combined - margin_t - margin_b

        logo_x_combined, logo_y_combined = 1.0, 1.0
        if plot_area_width_combined > 0 and plot_area_height_combined > 0:
            logo_x_combined = 1 + (margin_r / plot_area_width_combined)
            logo_y_combined = 1 + (margin_t / plot_area_height_combined)
        else:
            logger.warning("Plot area non-positive for combined chart. Defaulting logo position.")

        combined_logo_properties = {**logo_base_config, "x": logo_x_combined, "y": logo_y_combined}

        fig_combined = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combined.add_trace(
            go.Bar(x=df_combined_plot['display_name'], y=df_combined_plot['accuracy'], name='Accuracy',
                   marker_color=GOODFIN_PRIMARY_COLOR, 
                   text=df_combined_plot['accuracy'], 
                   texttemplate='%{text:.3f}', textposition='outside', opacity=0.8),
            secondary_y=False,
        )
        fig_combined.add_trace(
            go.Scatter(x=df_combined_plot['display_name'], y=df_combined_plot['avg_time_per_question'], name='Avg Time/Q (s)',
                       line=dict(color=GOODFIN_SECONDARY_COLOR, width=2.5), 
                       mode='lines+markers', 
                       marker=dict(size=8, line=dict(width=1, color=GOODFIN_BACKGROUND_COLOR)),
                       text=df_combined_plot['avg_time_per_question'], texttemplate='%{text:.2f}s', textposition="top center"),
            secondary_y=True,
        )
        fig_combined.update_layout(
            title_text="<b>Combined Performance: Accuracy & Avg. Question Time</b>",
            xaxis_tickangle=-45, 
            height=figure_height_combined, width=figure_width_combined,
            xaxis_title="Model (Strategy)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), 
            barmode='group',
            images=[combined_logo_properties]
        )
        fig_combined.update_yaxes(
            title_text="<b>Accuracy Score</b>", secondary_y=False, range=[0, max(1.05, df_combined_plot['accuracy'].max() * 1.1 if not df_combined_plot.empty else 1.05)],
            showgrid=True, gridcolor=GOODFIN_GRID_COLOR, gridwidth=1,
            linecolor=GOODFIN_MUTED_TEXT_COLOR, showline=True, mirror=True
        )
        fig_combined.update_yaxes(
            title_text="<b>Avg Time/Q (s)</b>", secondary_y=True, 
            range=[0, df_combined_plot['avg_time_per_question'].max() * 1.15 if not df_combined_plot.empty else 10],
            showgrid=False, 
            linecolor=GOODFIN_MUTED_TEXT_COLOR, showline=True, mirror=True
        )
        fig_combined.write_image(os.path.join(charts_output_dir, 'model_strategy_combined_metrics.png'), scale=2)
        fig_combined.write_html(os.path.join(charts_output_dir, 'model_strategy_combined_metrics.html'))
        logger.info(f"Saved combined metrics chart to {charts_output_dir}")
    else:
        logger.warning("Not enough data for combined metrics chart (possibly all response times or accuracies are NaN).")

    logger.info(f"All charts saved to {charts_output_dir}")












