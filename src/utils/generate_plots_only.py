import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import sys
import json
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src import config
    from src import plotting
    from src.utils import ui_utils
    from src.main import setup_logging 
    from src.evaluations import classification as classification_eval
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
        base_model_id = row.get('Model')
        strategy_name = row.get('Strategy')
        
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

def parse_filename_local(filename: Path):
    match = re.match(r"response_data_(.*?)__(.*?).json", filename.name)
    if match:
        config_id = match.group(1)
        strategy_key = match.group(2)
        if strategy_key.endswith('.json'):
            strategy_key = strategy_key[:-5]
        return config_id, strategy_key
    logger.warning(f"Local_parse: Could not parse config_id and strategy from filename: {filename.name}")
    return None, None

def get_pretty_strategy_name_local(strategy_key_from_filename: str) -> str:
    if strategy_key_from_filename == "default":
        return "Default (Single Pass)"
    elif strategy_key_from_filename == "cotn3": 
        return "Self-Consistency CoT (N=3 samples)"
    elif strategy_key_from_filename == "cotn5": 
        return "Self-Consistency CoT (N=5 samples)"
    elif strategy_key_from_filename == "sd":    
        return "Self-Discover"
    
    logger.warning(f"Local_pretty_name: No pretty name mapping for strategy key: '{strategy_key_from_filename}'. Using key as name.")
    return strategy_key_from_filename

def _generate_confusion_matrices(base_output_dir: Path):
    """
    Generates confusion matrix plots from individual response_data_*.json files
    found in results/json subdirectories (results/json/default, results/json/cotn3, etc.).
    """
    logger.info("Attempting to generate confusion matrices...")
    confusion_matrices_output_dir = base_output_dir / "confusion_matrices"
    confusion_matrices_output_dir.mkdir(parents=True, exist_ok=True)
    
    base_json_dir = config.RESULTS_DIR / "json"
    
    strategy_dirs_map = {
        "default": base_json_dir / "default",
        "cotn3": base_json_dir / "cotn3",
        "cotn5": base_json_dir / "cotn5",
        "sd": base_json_dir / "sd"
    }

    found_any_jsons = False
    for strategy_key, specific_strategy_dir in strategy_dirs_map.items():
        if specific_strategy_dir.exists() and specific_strategy_dir.is_dir():
            logger.info(f"Scanning for CM data in: {specific_strategy_dir}")
            for json_file_path in specific_strategy_dir.glob("response_data_*.json"):
                found_any_jsons = True
                logger.debug(f"Processing for CM: {json_file_path.name}")                
                config_id, file_strategy_key_from_name = parse_filename_local(json_file_path) 
                
                if not config_id or not file_strategy_key_from_name:
                    logger.warning(f"Could not parse {json_file_path.name}, skipping for CM.")
                    continue

                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f) 
                    if not results_data or not isinstance(results_data, list):
                        logger.warning(f"No data or invalid data format in {json_file_path.name}. Skipping.")
                        continue
                    classification_metrics = classification_eval.evaluate_classification(results_data)
                    matrix = classification_metrics.get('confusion_matrix') 
                    labels = classification_metrics.get('labels')
                    
                   
                    pretty_strategy_name = get_pretty_strategy_name_local(file_strategy_key_from_name)

                    if matrix is not None and labels and len(matrix) > 0: 
                        logger.info(f"Plotting confusion matrix for {config_id} - {pretty_strategy_name}")
                        plotting._plot_confusion_matrix(
                            matrix=matrix, 
                            labels=labels,
                            model_id=config_id,
                            strategy_name=pretty_strategy_name,
                            output_dir=confusion_matrices_output_dir
                        )
                    else:
                        logger.warning(f"No valid confusion matrix data in {json_file_path.name} (parsed as {config_id}/{pretty_strategy_name}) after evaluation.")

                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {json_file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {json_file_path.name} for confusion matrix: {e}", exc_info=True)
        else:
            logger.warning(f"Strategy directory for CMs not found or is not a dir: {specific_strategy_dir}")

    if not found_any_jsons:
        logger.warning(f"No response_data_*.json files found in any of the specified strategy data directories {list(strategy_dirs_map.values())}. Skipping confusion matrices.")
    else:
        logger.info(f"Finished attempt to generate confusion matrices into {confusion_matrices_output_dir}")

def main():
    setup_logging() 
    logger.info("Starting plot generation from summary CSV file...")
    print("Starting plot generation from summary CSV file...")
    
    base_charts_output_dir = Path(config.RESULTS_DIR) / "CSV_PLOTS"
    base_charts_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base charts directory: {base_charts_output_dir}")

    trade_off_plots_dir = base_charts_output_dir / "trade_off_analysis"
    trade_off_plots_dir.mkdir(parents=True, exist_ok=True)

    comparative_plots_dir = base_charts_output_dir / "comparative_performance"
    comparative_plots_dir.mkdir(parents=True, exist_ok=True)
    
    model_type_plots_dir = base_charts_output_dir / "model_type_analysis"
    model_type_plots_dir.mkdir(parents=True, exist_ok=True)

    derived_metrics_plots_dir = base_charts_output_dir / "derived_metrics_analysis"
    derived_metrics_plots_dir.mkdir(parents=True, exist_ok=True)

    advanced_analysis_csv_dir = Path(config.RESULTS_DIR) / "advanced_analysis"

    summary_building_anim = ui_utils.LoadingAnimation(message="Building summary from CSV file")
    summary_building_anim.start()
    all_model_runs_summary = build_summary_from_csv()
    summary_building_anim.stop()

    if not all_model_runs_summary:
        ui_utils.print_error("No summary data could be built from CSV. Exiting plot generation.")
        return
    
    logger.info("Generating charts...")
    print("\\\\nGenerating charts...")
    chart_gen_anim = ui_utils.LoadingAnimation(message="Generating comparison charts")
    chart_gen_anim.start()
    try:
        _generate_confusion_matrices(base_charts_output_dir)
        plotting.generate_research_specific_charts(all_model_runs_summary, trade_off_plots_dir)
        plotting.generate_all_charts(all_model_runs_summary, comparative_plots_dir) 
        if hasattr(plotting, 'generate_reasoning_vs_non_reasoning_accuracy_plot'):
            plotting.generate_reasoning_vs_non_reasoning_accuracy_plot(all_model_runs_summary, model_type_plots_dir)
        
        if hasattr(plotting, 'generate_strategy_effectiveness_boxplot'):
            plotting.generate_strategy_effectiveness_boxplot(all_model_runs_summary, model_type_plots_dir)

        if advanced_analysis_csv_dir.exists():
            if hasattr(plotting, 'plot_aggregated_metrics_by_model_type'):
                plotting.plot_aggregated_metrics_by_model_type(advanced_analysis_csv_dir, derived_metrics_plots_dir)
            if hasattr(plotting, 'plot_efficiency_distribution_by_model_type'):
                plotting.plot_efficiency_distribution_by_model_type(advanced_analysis_csv_dir, derived_metrics_plots_dir)
            if hasattr(plotting, 'plot_efficiency_distribution_by_strategy'):
                plotting.plot_efficiency_distribution_by_strategy(advanced_analysis_csv_dir, derived_metrics_plots_dir)
            if hasattr(plotting, 'plot_strategy_aggregate_metrics'):
                plotting.plot_strategy_aggregate_metrics(advanced_analysis_csv_dir, derived_metrics_plots_dir)
            if hasattr(plotting, 'plot_model_aggregate_metrics'):
                plotting.plot_model_aggregate_metrics(advanced_analysis_csv_dir, derived_metrics_plots_dir)
        else:
            logger.warning(f"Advanced analysis CSV directory not found: {advanced_analysis_csv_dir}. Skipping plots from derived metrics.")

        chart_gen_anim.stop()
        ui_utils.print_success(f"Charts generated successfully in {base_charts_output_dir} and its subdirectories.")
        logger.info(f"Charts generated successfully in {base_charts_output_dir} and its subdirectories.")
    except Exception as e:
        chart_gen_anim.stop()
        logger.error(f"An error occurred during chart generation: {e}", exc_info=True)
        ui_utils.print_error(f"Chart generation failed: {e}")

    logger.info("Plot generation process finished.")
    ui_utils.print_info("Plot generation process finished.")

if __name__ == "__main__":
    main() 