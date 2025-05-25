"""
Main script for the CFA MCQ Reproducer pipeline.

Orchestrates the loading of data, processing with LLMs via different strategies,
_evaluation, and generation of comparison charts.
"""
import json
import logging
import numpy as np
from . import config
from .evaluations import classification as classification_eval
from .evaluations import resource_metrics as resource_eval
from .evaluations import cost_evaluation
from . import plotting
from .utils import ui_utils
from . import configs
import questionary
import time
from .strategies import default as default_strategy
from .strategies import self_consistency as sc_strategy
from .strategies import self_discover as sd_strategy
from .prompts.cot import COHERENT_CFA_COT
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging():
    """Configures root logger and adds a file handler for warnings."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    warning_log_path = config.RESULTS_DIR / "model_warnings.log"
    file_handler_warnings = logging.FileHandler(warning_log_path)
    file_handler_warnings.setLevel(logging.WARNING) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler_warnings.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler_warnings)
    logger.info(f"Warning messages (and above) will be logged to: {warning_log_path}")

AVAILABLE_STRATEGIES = {
    "default": {
        "function": default_strategy.run_default_strategy,
        "name": "Default (Single Pass)",
        "params": {}
    },
    "self_consistency_cot_n3": {
        "function": sc_strategy.run_self_consistency_strategy,
        "name": "Self-Consistency CoT (N=3 samples)",
        "params": {"n_samples": 3, "cot_template": COHERENT_CFA_COT}
    },
    "self_consistency_cot_n5": {
        "function": sc_strategy.run_self_consistency_strategy,
        "name": "Self-Consistency CoT (N=5 samples)",
        "params": {"n_samples": 5, "cot_template": COHERENT_CFA_COT}
    },
    "self_discover": {
        "function": sd_strategy.run_self_discover_strategy,
        "name": "Self-Discover",
        "params": {}
    }
}

def _run_analysis_script(module_string: str, message: str, proj_root: Path) -> bool:
    """Helper to run an analysis script as a module."""
    ui_utils.print_info(message)
    python_executable = sys.executable
    try:
        process_result = subprocess.run(
            [python_executable, "-m", module_string],
            capture_output=True, text=True, check=False, cwd=proj_root
        )
        if process_result.returncode == 0:
            ui_utils.print_success(f"Script {module_string} completed successfully.")
            logger.info(f"{module_string} completed successfully.")
            if process_result.stdout: logger.info(f"{module_string} stdout:\n{process_result.stdout}")
            if process_result.stderr: logger.warning(f"{module_string} stderr (on success):\n{process_result.stderr}")
            return True
        else:
            ui_utils.print_error(f"Script {module_string} encountered an error.")
            logger.error(f"{module_string} failed. RC: {process_result.returncode}. Stderr:\n{process_result.stderr}")
            if process_result.stdout: logger.error(f"{module_string} stdout (on error):\n{process_result.stdout}")
            return False
    except Exception as e:
        ui_utils.print_error(f"Error running {module_string}: {e}")
        logger.error(f"Error running {module_string}: {e}", exc_info=True)
        return False

def _run_model_evaluations(
    processed_data: list, 
    selected_model_ids: list[str] | None, 
    strategy_key: str,
    all_model_runs_summary: dict,
    use_cache: bool
) -> None:
    """
    Helper function to run evaluation for selected models and a specific strategy.
    Updates the all_model_runs_summary dictionary in place.
    """
    chosen_strategy = AVAILABLE_STRATEGIES[strategy_key]
    strategy_func = chosen_strategy["function"]
    strategy_params = chosen_strategy["params"]
    strategy_name_for_file = strategy_key.replace(" ", "_").lower()
    ui_utils.print_info(f"Preparing for strategy: {chosen_strategy['name']}")
    
    strategy_folder_map = {
        "default": "default",
        "self_consistency_cot_n3": "cotn3",
        "self_consistency_cot_n5": "cotn5",
        "self_discover": "sd"
    }
    
    strategy_folder_name = strategy_folder_map.get(strategy_key, strategy_name_for_file) 
    
    
    json_output_dir = config.RESULTS_DIR / "json" / strategy_folder_name
    json_output_dir.mkdir(parents=True, exist_ok=True) 

    relevant_model_configs_source = []
    if "cot" in strategy_key.lower(): 
        print(f"Using CoT-specific model configurations for strategy '{chosen_strategy['name']}'.")
        logger.info(f"Using CoT-specific model configurations from src.configs.cot_config")
        relevant_model_configs_source = configs.COT_CONFIGS
    elif strategy_key == "self_discover":
        print(f"Using Self-Discover-specific model configurations for strategy '{chosen_strategy['name']}'.")
        logger.info(f"Using Self-Discover model configurations from src.configs.self_discover_config")
        relevant_model_configs_source = configs.SELF_DISCOVER_CONFIGS
    else:
        print(f"Using default model configurations for strategy '{chosen_strategy['name']}'.")
        logger.info(f"Using default model configurations from src.configs.default_config")
        relevant_model_configs_source = configs.DEFAULT_CONFIGS

    active_model_configs = []
    if not selected_model_ids or "__ALL__" in selected_model_ids:
        active_model_configs = relevant_model_configs_source
        if not active_model_configs:
            print(f"Warning: No models found in the configuration set for strategy '{chosen_strategy['name']}'.")
            logger.warning(f"No models found in the configuration set for strategy '{chosen_strategy['name']}'.")
    else:
        filtered_configs = []
        for m_strategy_config in relevant_model_configs_source:
            config_id_from_strategy = m_strategy_config.get('config_id', m_strategy_config.get('model_id'))

            base_id_for_comparison = config_id_from_strategy
            is_cot_strategy_check = "cot" in strategy_key.lower()
            is_self_discover_strategy_check = strategy_key == "self_discover"

            if is_cot_strategy_check and config_id_from_strategy.endswith('-cot'):
                 base_id_for_comparison = config_id_from_strategy[:-len('-cot')]
            elif is_self_discover_strategy_check and config_id_from_strategy.endswith('-self-discover'):
                 base_id_for_comparison = config_id_from_strategy[:-len('-self-discover')]
            if base_id_for_comparison and base_id_for_comparison in selected_model_ids:
                 filtered_configs.append(m_strategy_config)
        
        active_model_configs = filtered_configs

        if not active_model_configs:
            print(f"Warning: None of the selected models ({selected_model_ids}) are present in the configuration set for strategy '{chosen_strategy['name']}'.")
            logger.warning(f"None of the selected models ({selected_model_ids}) are present in the configuration set for strategy '{chosen_strategy['name']}'.")

    logger.info(f"Found {len(active_model_configs)} model configurations to run for strategy {strategy_key}.")

    for model_config_item in active_model_configs:
        config_id_loop = model_config_item.get("config_id", model_config_item.get("model_id"))
        model_type_loop = model_config_item.get("type")
        
        base_model_id_for_summary = config_id_loop
        is_cot_strategy = "cot" in strategy_key.lower()
        is_self_discover_strategy = strategy_key == "self_discover"
        if is_cot_strategy and config_id_loop.endswith('-cot'):
             base_model_id_for_summary = config_id_loop[:-len('-cot')]
        elif is_self_discover_strategy and config_id_loop.endswith('-self-discover'):
             base_model_id_for_summary = config_id_loop[:-len('-self-discover')]

        run_identifier_log = f"{config_id_loop}__{strategy_name_for_file}"
        logger.info(f"\n{'='*20} Processing Model: {config_id_loop} ({model_type_loop}) with Strategy: {chosen_strategy['name']} {'='*20}")
        data_for_this_run = processed_data
        credentials_ok = True

        model_type = model_config_item.get('type')

        if model_type == 'mistral_official':
            if not config.MISTRAL_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing Mistral API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (Mistral Official): Missing Mistral API key.")
                credentials_ok = False
        elif model_type == 'openai':
            if not config.OPENAI_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing OpenAI API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (OpenAI): Missing OpenAI API key.")
                credentials_ok = False
        elif model_type == 'gemini':
             if not config.GEMINI_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing Gemini API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (Gemini): Missing Gemini API key.")
                credentials_ok = False
        elif model_type == 'xai':
            if not config.XAI_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing xAI API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (xAI): Missing xAI API key.")
                credentials_ok = False
        elif model_type == 'writer':
             if not config.WRITER_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing Writer API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (Writer): Missing Writer API key.")
                credentials_ok = False
        elif model_type == 'groq':
            if not config.GROQ_API_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing Groq API key.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (Groq): Missing Groq API key.")
                credentials_ok = False

        if not credentials_ok:
            if base_model_id_for_summary not in all_model_runs_summary:
                all_model_runs_summary[base_model_id_for_summary] = {}
            all_model_runs_summary[base_model_id_for_summary][chosen_strategy['name']] = {
                "error": "Missing credentials", "num_processed": 0, "total_run_time": 0.0
            }
            
            run_identifier_log = f"{config_id_loop}__{strategy_name_for_file}"
            
            model_response_filename = json_output_dir / f"response_data_{run_identifier_log}.json"
            logger.info(f"Credential check failed. Expected output for {run_identifier_log} would be at {model_response_filename}")
            continue

        run_identifier_log = f"{config_id_loop}__{strategy_name_for_file}" 
        model_response_filename = json_output_dir / f"response_data_{run_identifier_log}.json"
        llm_run_results_data_from_cache = None
        is_from_cache = False

        if use_cache and model_response_filename.exists():
            ui_utils.print_info(f"Found cached results for {run_identifier_log}. Attempting to load...")
            logger.info(f"Attempting to load cached results for {run_identifier_log} from {model_response_filename}")
            try:
                with open(model_response_filename, 'r', encoding='utf-8') as f:
                    loaded_data_temp = json.load(f)
                if not isinstance(loaded_data_temp, list):
                    logger.warning(f"Cached file {model_response_filename} for {run_identifier_log} does not contain a list. Invalidating for this run.")
                    ui_utils.print_warning(f"Cached file for {run_identifier_log} is invalid. Will re-run.")
                else:
                    llm_run_results_data_from_cache = loaded_data_temp
                    is_from_cache = True
                    logger.info(f"Successfully loaded {len(llm_run_results_data_from_cache)} items from cached results for {run_identifier_log}.")
                    ui_utils.print_success(f"Using cached results for {run_identifier_log}.")
            except json.JSONDecodeError as e_json:
                logger.error(f"JSONDecodeError loading cached results for {run_identifier_log} from {model_response_filename}: {e_json}. Will re-run.", exc_info=True)
                ui_utils.print_error(f"Error decoding cached JSON for {run_identifier_log}. Will re-run.")
            except Exception as e_load:
                logger.error(f"Failed to load cached results for {run_identifier_log} from {model_response_filename}: {e_load}. Will re-run.", exc_info=True)
                ui_utils.print_error(f"Error loading cached results for {run_identifier_log}. Will re-run.")

        run_start_time = time.time()
        llm_run_results_data = None 
        processing_animation = None 

        try:
            if is_from_cache and llm_run_results_data_from_cache is not None:
                llm_run_results_data = llm_run_results_data_from_cache
            else:
                logger.info(f"Processing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")
                print(f"\nProcessing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")
                processing_animation = ui_utils.LoadingAnimation(message=f"Processing with {config_id_loop} ({chosen_strategy['name']})")
                processing_animation.start()
                
                llm_run_results_data = strategy_func(data_for_this_run, model_config_item, **strategy_params)
                
                if processing_animation: processing_animation.stop()
                ui_utils.print_success(f"Completed processing {len(data_for_this_run)} questions with {config_id_loop} ({chosen_strategy['name']})")
                
                if llm_run_results_data:
                    try:
                        with open(model_response_filename, 'w', encoding='utf-8') as f:
                            json.dump(llm_run_results_data, f, indent=4)
                        logger.info(f"LLM responses for {run_identifier_log} saved to {model_response_filename}")
                        ui_utils.print_success(f"Results for {run_identifier_log} saved.")
                    except Exception as e_save:
                        logger.error(f"Failed to save LLM results for {run_identifier_log} to {model_response_filename}: {e_save}", exc_info=True)
                        ui_utils.print_error(f"Failed to save results for {run_identifier_log}.")
                elif not is_from_cache:
                    logger.warning(f"No data returned from strategy_func for {run_identifier_log}, so nothing to save.")

            run_end_time = time.time()
            total_run_time_seconds = run_end_time - run_start_time
            
            resource_usage_metrics = {}
            if llm_run_results_data:
                resource_usage_metrics = resource_eval.calculate_resource_usage(llm_run_results_data, model_config_item)
            else: 
                resource_usage_metrics = resource_eval.calculate_resource_usage([], model_config_item)
            
            if llm_run_results_data:
                logger.info(f"Performing final evaluation on {run_identifier_log} results...")
                eval_loading_message = f"Evaluating results for {run_identifier_log}"
                if is_from_cache:
                    eval_loading_message = f"Evaluating cached results for {run_identifier_log}"
                
                eval_loading = ui_utils.LoadingAnimation(message=eval_loading_message)
                eval_loading.start()
                classification_metrics = classification_eval.evaluate_classification(llm_run_results_data)
                eval_loading.stop()

                individual_results = {
                    **classification_metrics,
                    **resource_usage_metrics,
                    "total_run_time_s": total_run_time_seconds, 
                    "num_processed": len(llm_run_results_data),
                    "results_file": str(model_response_filename),
                    "config_id_used": config_id_loop 
                }

                if base_model_id_for_summary not in all_model_runs_summary:
                    all_model_runs_summary[base_model_id_for_summary] = {}
                all_model_runs_summary[base_model_id_for_summary][chosen_strategy['name']] = individual_results

                accuracy = classification_metrics.get('accuracy', 0.0)
                logger.info(f"Evaluation summary for {run_identifier_log}: Accuracy: {accuracy:.4f}")
                ui_utils.print_success(f"Evaluation complete for {run_identifier_log}: Accuracy: {accuracy:.4f}")
            else:
                logger.warning(f"No results generated or loaded for {run_identifier_log}. Skipping evaluation.")
                if not is_from_cache and processing_animation and processing_animation._thread and processing_animation._thread.is_alive():
                    processing_animation.stop() 
                
                if base_model_id_for_summary not in all_model_runs_summary:
                    all_model_runs_summary[base_model_id_for_summary] = {}
                all_model_runs_summary[base_model_id_for_summary][chosen_strategy['name']] = {
                    "error": "No results generated or loaded",
                    "num_processed": 0,
                    "total_run_time_s": total_run_time_seconds,
                    "config_id_used": config_id_loop
                }

        except Exception as model_proc_err:
            run_end_time = time.time()
            total_run_time_seconds = run_end_time - run_start_time
            if not is_from_cache and processing_animation and processing_animation._thread and processing_animation._thread.is_alive():
                processing_animation.stop()
            logger.error(f"An error occurred while processing {run_identifier_log}: {model_proc_err}", exc_info=True)
            ui_utils.print_error(f"Error processing {run_identifier_log}: {model_proc_err}")
            
            if base_model_id_for_summary not in all_model_runs_summary:
                all_model_runs_summary[base_model_id_for_summary] = {}
            all_model_runs_summary[base_model_id_for_summary][chosen_strategy['name']] = {
                "error": str(model_proc_err),
                "num_processed": 0,
                "total_run_time_s": total_run_time_seconds,
                "config_id_used": config_id_loop
            }

def _check_existing_json_results(base_json_dir: Path) -> bool:
    """Checks if any response_data_*.json files exist in strategy subfolders."""
    strategy_subfolders = ["default", "cotn3", "cotn5", "sd"]
    for folder_name in strategy_subfolders:
        strategy_dir = base_json_dir / folder_name
        if strategy_dir.exists() and strategy_dir.is_dir():
            if any(strategy_dir.glob("response_data_*.json")):
                logger.info(f"Found existing JSON results in {strategy_dir}")
                return True
    logger.info(f"No existing JSON results found in any subfolders of {base_json_dir}")
    return False

def main():
    setup_logging() 
    project_root = Path(__file__).resolve().parent.parent
    logger.info(f"Project root directory: {project_root}")
    logger.info("Starting CFA MCQ Reproducer pipeline...")
    print("Starting CFA MCQ Reproducer pipeline...")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "json").mkdir(parents=True, exist_ok=True) 
    processed_data = None
    if config.FILLED_JSON_PATH.exists():
        logger.info(f"Loading processed data from: {config.FILLED_JSON_PATH}")
        print(f"Loading data from: {config.FILLED_JSON_PATH}")
        data_loading = ui_utils.LoadingAnimation(message="Loading data")
        data_loading.start()

        try:
            with open(config.FILLED_JSON_PATH, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            data_loading.stop()

            if not isinstance(processed_data, list) or not processed_data:
                ui_utils.print_error(f"{config.FILLED_JSON_PATH} is empty or not a valid JSON list.")
                logger.error(f"{config.FILLED_JSON_PATH} is empty or not a valid JSON list. Exiting.")
                return

            ui_utils.print_success(f"Successfully loaded {len(processed_data)} entries.")
            logger.info(f"Successfully loaded {len(processed_data)} entries.")

        except json.JSONDecodeError:
            data_loading.stop()
            ui_utils.print_error(f"Error decoding JSON from {config.FILLED_JSON_PATH}.")
            logger.error(f"Error decoding JSON from {config.FILLED_JSON_PATH}. Exiting.", exc_info=True)
            return
        except Exception as e:
            data_loading.stop()
            ui_utils.print_error(f"Error reading {config.FILLED_JSON_PATH}: {e}")
            logger.error(f"Error reading {config.FILLED_JSON_PATH}: {e}. Exiting.", exc_info=True)
            return
    else:
        ui_utils.print_error(f"Input data file not found: {config.FILLED_JSON_PATH}")
        logger.error(f"Input data file not found: {config.FILLED_JSON_PATH}. Exiting.")
        return

    base_json_results_dir = config.RESULTS_DIR / "json"
    run_new_evaluations = True

    if _check_existing_json_results(base_json_results_dir):
        print("\n" + "="*30 + " GoodFin CFA MCQ Reproducer - Analysis Options " + "="*30)
        ui_utils.print_info("Existing LLM evaluation results (JSON files) were found.")
        
        analysis_choices = [
            {"name": "LLM Benchmark (Comprehensive Summary & All Plots)", "value": "llm_benchmark"},
            {"name": "Deep Dive Analysis (Advanced Metrics & All Plots)", "value": "deep_dive"},
            {"name": "Concise Performance Summary (Key Insights & Targeted Plots)", "value": "concise_summary"},
            {"name": "Generate All Plots Only (from current summary data)", "value": "plots_only"},
            {"name": "Proceed with New LLM Evaluations", "value": "proceed_new_eval"},
            {"name": "Exit", "value": "exit"},
        ]
        
        analysis_choice = questionary.select(
            "Choose an action:",
            choices=analysis_choices
        ).ask()

        scripts_to_run = []
        if analysis_choice == "llm_benchmark":
            scripts_to_run = [
                ("src.utils.update_summary_from_json", "Updating summary CSV with existing JSONs..."),
                ("src.utils.generate_plots_only", "Generating all plots from summary data...")
            ]
            run_new_evaluations = False
        elif analysis_choice == "deep_dive":
            scripts_to_run = [
                ("src.utils.update_summary_from_json", "Updating summary CSV with existing JSONs..."),
                ("src.utils.generate_advanced_analysis", "Generating advanced analysis data..."),
                ("src.utils.generate_plots_only", "Generating all plots (including advanced)...")
            ]
            run_new_evaluations = False
        elif analysis_choice == "concise_summary":
            scripts_to_run = [
                ("src.utils.update_summary_from_json", "Updating summary CSV with existing JSONs..."),
                ("src.evaluations.analyze_summary_metrics", "Generating concise performance summary & plots...")
            ]
            run_new_evaluations = False
        elif analysis_choice == "plots_only":
            scripts_to_run = [
                ("src.utils.generate_plots_only", "Generating all plots from current summary data...")
            ]
            run_new_evaluations = False
        elif analysis_choice == "proceed_new_eval":
            logger.info("User opted to proceed with new LLM evaluations from the analysis menu.")
            ui_utils.print_info("Proceeding with new LLM evaluations.")
            # run_new_evaluations remains True
        elif analysis_choice == "exit":
            ui_utils.print_info("Exiting.")
            return
        elif analysis_choice is None: # User cancelled the selection
            ui_utils.print_warning("No analysis option selected. Exiting.")
            return
        
        if not run_new_evaluations and scripts_to_run:
            all_scripts_succeeded = True
            for module_str, msg in scripts_to_run:
                if not _run_analysis_script(module_str, msg, project_root):
                    all_scripts_succeeded = False
                    # Potentially break here or log and continue
            if all_scripts_succeeded:
                ui_utils.print_success("Selected analysis tasks completed.")
            else:
                ui_utils.print_warning("Some analysis tasks encountered errors.")
            
            print(f"Relevant outputs (if any) can be found in: {config.RESULTS_DIR / 'CSV_PLOTS'} and {config.RESULTS_DIR / 'advanced_analysis'}")
            return # Exit after running selected analysis

    if run_new_evaluations:
        all_model_runs_summary = {}
        logger.info("Proceeding with standard LLM evaluation pipeline.")
        logger.info("Determining evaluation run type...")

        
        use_cache_if_available = questionary.confirm(
            "Do you want to use existing cached results (if available)? Choosing 'No' will re-run all evaluations.",
            default=True
        ).ask()

        if use_cache_if_available is None: 
            ui_utils.print_warning("Cache preference not selected. Exiting.")
            return
        
        if use_cache_if_available:
            ui_utils.print_info("Attempting to use cached results where available.")
            logger.info("User opted to use cached results.")
        else:
            ui_utils.print_info("Running all evaluations from scratch. Existing cached results will be ignored and overwritten.")
            logger.info("User opted to run evaluations from scratch.")

        run_mode = questionary.select(
            "Select run mode:",
            choices=[
                {"name": "Run Full Evaluation (All Models, Default + CoT SC N=3/5 + Self-Discover Strategies)", "value": "full"},
                {"name": "Custom Run (Select Models and Single Strategy)", "value": "custom"},
            ]
        ).ask()

        if not run_mode:
            ui_utils.print_warning("No run mode selected. Exiting.")
            return

        if run_mode == "full":
            logger.info("Full evaluation mode selected.")
            print("\nRunning Full Evaluation...")
            
            full_eval_strategies = [
                "default", 
                "self_consistency_cot_n3", 
                "self_consistency_cot_n5",
                "self_discover"
            ]
            
            for strategy_key_full in full_eval_strategies:
                print(f"\n--- Starting Full Eval: Strategy '{AVAILABLE_STRATEGIES[strategy_key_full]['name']}' ---")
                
                _run_model_evaluations(
                    processed_data=processed_data,
                    selected_model_ids=None, 
                    strategy_key=strategy_key_full,
                    all_model_runs_summary=all_model_runs_summary,
                    use_cache=use_cache_if_available
                )
                print(f"--- Completed Full Eval: Strategy '{AVAILABLE_STRATEGIES[strategy_key_full]['name']}' ---")

        elif run_mode == "custom":
            logger.info("Custom run mode selected.")
            print("\nStarting Custom Run Configuration...")
            
            print("\nPreparing available LLM models...")
            model_loading_anim = ui_utils.LoadingAnimation(message="Loading model configurations") 
            model_loading_anim.start()

            
            
            unique_model_configs_for_listing = configs.DEFAULT_CONFIGS 
            
            model_choices = [
                {
                    "name": f"{m.get('config_id', m.get('model_id'))} ({m.get('type')})", 
                    "value": m.get('config_id', m.get('model_id')) 
                }
                for m in unique_model_configs_for_listing 
            ]
            model_choices.insert(0, {"name": "[Run All Available Models]", "value": "__ALL__"})
            model_loading_anim.stop()
            ui_utils.print_info(f"Found {len(unique_model_configs_for_listing)} unique model configurations for selection.")

            print("\nPlease select which LLM models to run:")
            selected_model_ids = questionary.checkbox(
                "Select models (space to select, arrows to move, enter to confirm):",
                choices=model_choices,
                validate=lambda a: True if a else "Select at least one model."
            ).ask()
            if not selected_model_ids:
                ui_utils.print_warning("No models selected. Exiting.")
                return
            strategy_choices = [
                {"name": details["name"], "value": key}
                for key, details in AVAILABLE_STRATEGIES.items()
            ] 

            print("\nPlease select which prompting strategies to use:")
            selected_strategy_keys = questionary.checkbox(
                "Select prompting strategies (space to select, arrows to move, enter to confirm):",
                choices=strategy_choices,
                validate=lambda a: True if a else "Select at least one strategy."
            ).ask()

            if not selected_strategy_keys:
                ui_utils.print_warning("No prompting strategies selected. Exiting.")
                return
            
            if selected_model_ids and selected_strategy_keys:
                 for selected_strategy_key in selected_strategy_keys:
                     print(f"\n--- Starting Custom Run: Strategy '{AVAILABLE_STRATEGIES[selected_strategy_key]['name']}' ---")
                     _run_model_evaluations(
                         processed_data=processed_data,
                         selected_model_ids=selected_model_ids, 
                         strategy_key=selected_strategy_key,
                         all_model_runs_summary=all_model_runs_summary,
                         use_cache=use_cache_if_available
                     )
                     print(f"--- Completed Custom Run: Strategy '{AVAILABLE_STRATEGIES[selected_strategy_key]['name']}' ---")
            else:
                
                 logger.warning("Skipping custom run execution due to missing selections.")
                 ui_utils.print_warning("Skipping custom run execution due to missing selections.")

        
        if all_model_runs_summary:
            logger.info("\nGenerating final summary and charts...")
            print("\nGenerating final summary and charts...")
            
            
            generate_plots_confirm = questionary.confirm(
                "Do you want to generate plots now? (This may take a few minutes)",
                default=True
            ).ask()

            if generate_plots_confirm is None:
                ui_utils.print_warning("Plot generation preference not selected. Skipping plot generation.")
                logger.info("User did not select a preference for plot generation. Skipping.")
            elif generate_plots_confirm:
                ui_utils.print_info(f"Proceeding with plot generation. Output will be in {config.RESULTS_DIR / 'CSV_PLOTS'}.")
                logger.info("User opted to generate plots. Attempting to run src.utils.generate_plots_only...")
                try:
                    
                    python_executable = sys.executable
                    
                    process_result = subprocess.run(
                        [python_executable, "-m", "src.utils.generate_plots_only"],
                        capture_output=True,
                        text=True,
                        check=False  
                    )
                    if process_result.returncode == 0:
                        ui_utils.print_success("Plot generation script completed successfully.")
                        logger.info("Plot generation script (src.utils.generate_plots_only) completed successfully.")
                        if process_result.stdout:
                            logger.info(f"Plotting script stdout:\n{process_result.stdout}")
                        if process_result.stderr: 
                            logger.warning(f"Plotting script stderr (on success):\n{process_result.stderr}")
                    else:
                        ui_utils.print_error("Plot generation script encountered an error.")
                        logger.error(f"Plot generation script (src.utils.generate_plots_only) failed with return code {process_result.returncode}.")
                        if process_result.stdout:
                            logger.error(f"Plotting script stdout (on error):\n{process_result.stdout}")
                        if process_result.stderr:
                            logger.error(f"Plotting script stderr (on error):\n{process_result.stderr}")
                    ui_utils.print_info(f"Plots (if generated) are in {config.RESULTS_DIR / 'CSV_PLOTS'}")

                except FileNotFoundError:
                    ui_utils.print_error(f"Error: Could not find the Python executable '{sys.executable}' to run the plotting script.")
                    logger.error(f"FileNotFoundError: Python executable '{sys.executable}' not found when trying to run plotting script.", exc_info=True)
                except Exception as e:
                    ui_utils.print_error(f"An unexpected error occurred while trying to run the plotting script: {e}")
                    logger.error(f"Unexpected error running plotting script: {e}", exc_info=True)
            else:
                ui_utils.print_info("Skipping plot generation as per user request.")
                logger.info("User opted not to generate plots at this time.")
                
        else:
           logger.warning("No models were processed in this run.")
           ui_utils.print_warning("No models were processed.")
        
        logger.info("CFA MCQ Reproducer pipeline finished.")
        logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")
        ui_utils.print_info(f"CFA MCQ Reproducer pipeline finished. Results saved in: {config.RESULTS_DIR}")
        
        if all_model_runs_summary and run_new_evaluations: # Ensure summary is printed only if new evals were run
            logger.info("\n" + "="*30 + " Overall Run Comparison " + "="*30)
            summary_loading = ui_utils.LoadingAnimation(message="Preparing results summary")
            summary_loading.start()
            
            header = "| {:<45} | {:<25} | {:>8} | {:>12} | {:>15} | {:>19} | {:>18} | {:>15} |".format(
                "Model", "Strategy", "Accuracy", "Avg Time/Q (s)", "Total Time (s)", "Total Output Tokens", "Avg Ans Len", "Total Cost ($)"
            )
            logger.info(header)
            
            separator = "|" + "-"*47 + "|" + "-"*27 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*17 + "|" + "-"*21 + "|" + "-"*20 + "|" + "-"*17 + "|"
            logger.info(separator)

            summary_rows_for_print = []
            for model_id, strategy_data in sorted(all_model_runs_summary.items()): 
                for strategy, data in strategy_data.items():
                    model_disp = model_id
                    strategy_disp = strategy
                    if "error" in data:    
                        row_str = "| {:<45} | {:<25} | {:^8} | {:^12} | {:^15} | {:^19} | {:^18} | {:^15} |".format(
                            model_disp, strategy_disp, "FAILED", f"({data.get('error', 'Unknown')})", "---", "---", "---", "---" 
                        )
                    else:          
                        total_time_s = data.get('total_run_time_s')
                        total_time_str = f"{total_time_s:.2f}" if total_time_s is not None else "N/A"
                        time_s = data.get('average_latency_ms')
                        time_str = f"{time_s:.2f}" if time_s is not None else "N/A"
                        tokens = data.get('total_output_tokens')
                        token_str = f"{tokens:.0f}" if tokens is not None else "N/A"
                        ans_len = data.get('avg_answer_length')
                        ans_len_str = f"{ans_len:.1f}" if ans_len is not None else "N/A"
                        acc = data.get('accuracy', 0.0)
                        cost = data.get('total_cost')
                        cost_str = f"${cost:.4f}" if cost is not None and cost > 0 else ("$0.0000" if cost == 0 else "N/A")
                        row_str = "| {:<45} | {:<25} | {:>8.4f} | {:>12} | {:>15} | {:>19} | {:>18} | {:>15} |".format(
                            model_disp, strategy_disp, acc, time_str, total_time_str, token_str, ans_len_str, cost_str
                        )
                    logger.info(row_str)
                    summary_rows_for_print.append(row_str)
            final_separator = separator
            logger.info(final_separator)
            summary_loading.stop()
            print("\n" + "="*30 + " Overall Run Comparison " + "="*30)
            print(header)
            print(separator) 
            for row in summary_rows_for_print:
                print(row)
            print(final_separator) 
        else:
           logger.warning("No models were processed in this run.")
           ui_utils.print_warning("No models were processed.")
        
        logger.info("CFA MCQ Reproducer pipeline finished.")
        logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")
        ui_utils.print_info(f"CFA MCQ Reproducer pipeline finished. Results saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()