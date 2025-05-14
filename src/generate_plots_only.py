import json
import logging
import re
from pathlib import Path
from typing import Dict, Any
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src import config
    from src.evaluations import classification as classification_eval
    from src.evaluations import resource_metrics as resource_eval
    from src import plotting
    from src.utils import ui_utils
    from src.main import AVAILABLE_STRATEGIES 
    from src.main import setup_logging 
except ImportError as e:
    print(f"Error importing modules. Make sure you are in the project root directory.")
    print(f"If running 'python src/generate_plots_only.py', try 'python -m src.generate_plots_only' instead.")
    print(f"Import error: {e}")
    sys.exit(1)


logger = logging.getLogger(__name__)
FILENAME_PATTERN = re.compile(r"response_data_(.*?)__(.*?)\.json")

def get_base_model_id_and_strategy_name(config_id_from_file: str, strategy_key_from_file: str) -> tuple[str, str | None]:
    """
    Determines the base model ID (for summary grouping) and the full strategy name.
    Replicates logic from main._run_model_evaluations.
    """
    base_model_id = config_id_from_file
    strategy_details = AVAILABLE_STRATEGIES.get(strategy_key_from_file)
    strategy_name = strategy_details['name'] if strategy_details else None

    if strategy_name:
        is_cot_strategy = "cot" in strategy_key_from_file.lower()
        is_self_discover_strategy = strategy_key_from_file == "self_discover"        
        if is_cot_strategy and config_id_from_file.endswith('-cot'):
             base_model_id = config_id_from_file[:-len('-cot')]
        elif is_self_discover_strategy and config_id_from_file.endswith('-self-discover'):
             base_model_id = config_id_from_file[:-len('-self-discover')]
    else:
        logger.warning(f"Could not find strategy details for key: {strategy_key_from_file}")

    return base_model_id, strategy_name


def build_summary_from_results() -> Dict[str, Any]:
    """
    Scans the results directory, loads JSON data, calculates metrics,
    and builds the summary dictionary needed for plotting.
    """
    all_model_runs_summary = {}
    results_dir = config.RESULTS_DIR
    logger.info(f"Scanning for result files in: {results_dir}")

    found_files = list(results_dir.glob("response_data_*.json"))
    if not found_files:
        logger.warning(f"No 'response_data_*.json' files found in {results_dir}. Cannot generate plots.")
        return {}

    logger.info(f"Found {len(found_files)} result files. Processing...")

    for result_file in found_files:
        match = FILENAME_PATTERN.match(result_file.name)
        if not match:
            logger.warning(f"Skipping file with unexpected name format: {result_file.name}")
            continue

        config_id_from_file = match.group(1)
        strategy_key_from_file = match.group(2)
        
        base_model_id, strategy_name = get_base_model_id_and_strategy_name(config_id_from_file, strategy_key_from_file)

        if not strategy_name:
            logger.warning(f"Skipping file {result_file.name}: Could not determine strategy name from key '{strategy_key_from_file}'.")
            continue

        logger.info(f"Processing {result_file.name} (Model: {base_model_id}, Strategy: {strategy_name})")

        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                llm_run_results_data = json.load(f)

            if not llm_run_results_data or not isinstance(llm_run_results_data, list):
                 logger.warning(f"Skipping {result_file.name}: File is empty or not a valid JSON list.")
                 continue
            model_type_for_cost = "unknown"
            if "gpt" in config_id_from_file.lower() or "o1" in config_id_from_file.lower() or "o3" in config_id_from_file.lower() or "o4" in config_id_from_file.lower():
                model_type_for_cost = "openai"
            elif "claude" in config_id_from_file.lower():
                model_type_for_cost = "anthropic" 

            
            current_model_config_info = {
                "config_id": config_id_from_file,
                "type": model_type_for_cost 
            }
            classification_metrics = classification_eval.evaluate_classification(llm_run_results_data)
            resource_usage_metrics = resource_eval.calculate_resource_usage(llm_run_results_data, current_model_config_info)
            combined_metrics = {
                **classification_metrics,
                **resource_usage_metrics,
                "total_run_time_s": None, 
                "num_processed": len(llm_run_results_data),
                "results_file": str(result_file),
                "config_id_used": config_id_from_file 
            }
            if base_model_id not in all_model_runs_summary:
                all_model_runs_summary[base_model_id] = {}
            
            if strategy_name in all_model_runs_summary[base_model_id]:
                 logger.warning(f"Duplicate entry found for Model: {base_model_id}, Strategy: {strategy_name}. Overwriting with data from {result_file.name}. Check for duplicate result files.")

            all_model_runs_summary[base_model_id][strategy_name] = combined_metrics

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {result_file.name}. Skipping.")
        except Exception as e:
            logger.error(f"Failed to process file {result_file.name}: {e}", exc_info=True)
            if base_model_id not in all_model_runs_summary:
                 all_model_runs_summary[base_model_id] = {}
            all_model_runs_summary[base_model_id][strategy_name] = {"error": f"Failed to process: {e}"}


    logger.info("Finished processing result files.")
    return all_model_runs_summary

def main():
    setup_logging() 
    logger.info("Starting plot generation from existing results...")
    print("Starting plot generation from existing results...")
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_building_anim = ui_utils.LoadingAnimation(message="Building summary from result files")
    summary_building_anim.start()
    all_model_runs_summary = build_summary_from_results()
    summary_building_anim.stop()

    if not all_model_runs_summary:
        ui_utils.print_error("No summary data could be built. Exiting plot generation.")
        return
    
    logger.info("Generating charts...")
    print("\nGenerating charts...")
    chart_gen_anim = ui_utils.LoadingAnimation(message="Generating comparison charts")
    chart_gen_anim.start()
    try:
        plotting.generate_all_charts(all_model_runs_summary, config.CHARTS_DIR)
        chart_gen_anim.stop()
        ui_utils.print_success(f"Charts generated successfully in {config.CHARTS_DIR}")
        logger.info(f"Charts generated successfully in {config.CHARTS_DIR}")
    except Exception as e:
        chart_gen_anim.stop()
        logger.error(f"An error occurred during chart generation: {e}", exc_info=True)
        ui_utils.print_error(f"Chart generation failed: {e}")

    logger.info("Plot generation process finished.")
    ui_utils.print_info("Plot generation process finished.")


if __name__ == "__main__":
    main() 