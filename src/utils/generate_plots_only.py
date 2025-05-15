import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src import config
    from src import plotting
    from src.utils import ui_utils
    from src.main import setup_logging 
except ImportError as e:
    print(f"Error importing modules. If running as 'python src/utils/generate_plots_only.py', ensure you are in the project root.")
    print(f"If running as a module 'python -m src.utils.generate_plots_only', check relative imports.")
    print(f"Import error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

def build_summary_from_csv() -> Dict[str, Any]:
    """
    Loads data from the summary CSV file and transforms it into the
    all_model_runs_summary dictionary structure needed for plotting.
    """
    all_model_runs_summary = {}
    summary_csv_path = Path(config.RESULTS_DIR) / "all_runs_summary_metrics.csv"

    logger.info(f"Loading summary data from CSV: {summary_csv_path}")

    if not summary_csv_path.exists():
        logger.error(f"Summary CSV file not found: {summary_csv_path}. Cannot generate plots.")
        ui_utils.print_error(f"Summary CSV file not found: {summary_csv_path}")
        return {}

    try:
        summary_df = pd.read_csv(summary_csv_path)
    except Exception as e:
        logger.error(f"Error reading summary CSV {summary_csv_path}: {e}", exc_info=True)
        ui_utils.print_error(f"Error reading summary CSV {summary_csv_path}: {e}")
        return {}

    if summary_df.empty:
        logger.warning(f"Summary CSV {summary_csv_path} is empty. No data to plot.")
        ui_utils.print_warning(f"Summary CSV {summary_csv_path} is empty.")
        return {}

    logger.info(f"Loaded {len(summary_df)} rows from {summary_csv_path}. Processing...")

    for _, row in summary_df.iterrows():
        base_model_id = row.get('base_model_id')
        strategy_name = row.get('strategy_name')
        
        if pd.isna(base_model_id) or pd.isna(strategy_name):
            logger.warning(f"Skipping row due to missing Model or Strategy: {row.to_dict()}")
            continue
        accuracy = row.get('Accuracy')
        avg_time_q_s = row.get('Avg Time/Q (s)')
        total_output_tokens = row.get('Total Output Tokens')
        avg_ans_len = row.get('Avg Ans Len')
        total_cost = row.get('Total Cost ($)')
        num_processed = row.get('Num Processed')
        config_id_used = row.get('Config ID Used')
        total_run_time_s = row.get('Total API Response Time (s)')
        model_type = row.get('Model Type')
        average_latency_ms = None
        if pd.notna(avg_time_q_s):
            try:
                average_latency_ms = float(avg_time_q_s) * 1000
            except ValueError:
                logger.warning(f"Could not convert 'Avg Time/Q (s)' to float for {base_model_id}/{strategy_name}. Value: {avg_time_q_s}")


        metrics_dict = {
            'accuracy': float(accuracy) if pd.notna(accuracy) else None,
            'average_latency_ms': average_latency_ms,
            'total_output_tokens': int(total_output_tokens) if pd.notna(total_output_tokens) else None,
            'avg_answer_length': float(avg_ans_len) if pd.notna(avg_ans_len) else None,
            'total_cost': float(total_cost) if pd.notna(total_cost) else None,
            'num_processed': int(num_processed) if pd.notna(num_processed) else None,
            'config_id_used': str(config_id_used) if pd.notna(config_id_used) else None,
            'total_run_time_s': float(total_run_time_s) if pd.notna(total_run_time_s) else None,
            'model_type': str(model_type) if pd.notna(model_type) else None,
        }        
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}


        if base_model_id not in all_model_runs_summary:
            all_model_runs_summary[base_model_id] = {}
        
        if strategy_name in all_model_runs_summary[base_model_id]:
             logger.warning(f"Duplicate entry found for Model: {base_model_id}, Strategy: {strategy_name} from CSV. Overwriting with later entry.")

        all_model_runs_summary[base_model_id][strategy_name] = metrics_dict

    if not all_model_runs_summary:
        logger.warning("No valid summary data could be built from the CSV.")
    else:
        logger.info(f"Successfully built summary for {len(all_model_runs_summary)} models from CSV.")
        
    return all_model_runs_summary

def main():
    setup_logging() 
    logger.info("Starting plot generation from summary CSV file...")
    print("Starting plot generation from summary CSV file...")
    
    charts_output_dir = Path(config.RESULTS_DIR) / "json_plot_charts"
    charts_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Charts will be saved to: {charts_output_dir}")

    summary_building_anim = ui_utils.LoadingAnimation(message="Building summary from CSV file")
    summary_building_anim.start()
    all_model_runs_summary = build_summary_from_csv()
    summary_building_anim.stop()

    if not all_model_runs_summary:
        ui_utils.print_error("No summary data could be built from CSV. Exiting plot generation.")
        return
    
    logger.info("Generating charts...")
    print("\\nGenerating charts...")
    chart_gen_anim = ui_utils.LoadingAnimation(message="Generating comparison charts")
    chart_gen_anim.start()
    try:
        plotting.generate_all_charts(all_model_runs_summary, charts_output_dir)
        chart_gen_anim.stop()
        ui_utils.print_success(f"Charts generated successfully in {charts_output_dir}")
        logger.info(f"Charts generated successfully in {charts_output_dir}")
    except Exception as e:
        chart_gen_anim.stop()
        logger.error(f"An error occurred during chart generation: {e}", exc_info=True)
        ui_utils.print_error(f"Chart generation failed: {e}")

    logger.info("Plot generation process finished.")
    ui_utils.print_info("Plot generation process finished.")

if __name__ == "__main__":
    main() 