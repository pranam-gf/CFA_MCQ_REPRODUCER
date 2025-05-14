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

logger = logging.getLogger(__name__)

def setup_logging():
    """Configures root logger and adds a file handler for warnings."""
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def _run_model_evaluations(
    processed_data: list, 
    selected_model_ids: list[str] | None, 
    strategy_key: str,
    all_model_runs_summary: dict
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
        is_cot_strategy = "cot" in strategy_key.lower()
        is_self_discover_strategy = strategy_key == "self_discover"
        for m in relevant_model_configs_source:
            config_id = m.get('config_id', m.get('model_id'))
            base_model_id = config_id 

            if is_cot_strategy and config_id.endswith('-cot'):
                 base_model_id = config_id[:-len('-cot')]
            elif is_self_discover_strategy and config_id.endswith('-self-discover'):
                 base_model_id = config_id[:-len('-self-discover')]

            if base_model_id in selected_model_ids:
                 filtered_configs.append(m)
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

        if model_type == 'bedrock' or model_type == 'sagemaker':
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                logger.error(f"Skipping {config_id_loop}: Missing AWS credentials.")
                ui_utils.print_warning(f"Skipping {config_id_loop} (Bedrock/SageMaker): Missing AWS credentials.")
                credentials_ok = False
        elif model_type == 'mistral_official':
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
            continue

        run_start_time = time.time()
        try:
            logger.info(f"Processing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")
            print(f"\nProcessing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")
            processing_animation = ui_utils.LoadingAnimation(message=f"Processing with {config_id_loop} ({chosen_strategy['name']})")
            processing_animation.start()
            llm_run_results_data = strategy_func(data_for_this_run, model_config_item, **strategy_params)
            run_end_time = time.time()
            total_run_time_seconds = run_end_time - run_start_time
            processing_animation.stop()
            ui_utils.print_success(f"Completed processing {len(data_for_this_run)} questions with {config_id_loop} ({chosen_strategy['name']})")

            
            resource_usage_metrics = {}
            if llm_run_results_data:
                resource_usage_metrics = resource_eval.calculate_resource_usage(llm_run_results_data, model_config_item)
            else: 
                resource_usage_metrics = resource_eval.calculate_resource_usage([], model_config_item)

            model_response_filename = config.RESULTS_DIR / f"response_data_{run_identifier_log}.json"
            try:
                with open(model_response_filename, 'w', encoding='utf-8') as f:
                    json.dump(llm_run_results_data, f, indent=4)
                logger.info(f"LLM responses for {run_identifier_log} saved to {model_response_filename}")
                ui_utils.print_success(f"Results for {run_identifier_log} saved.")
            except Exception as e_save:
                logger.error(f"Failed to save LLM results for {run_identifier_log} to {model_response_filename}: {e_save}", exc_info=True)
                ui_utils.print_error(f"Failed to save results for {run_identifier_log}.")

            if llm_run_results_data:
                logger.info(f"Performing final evaluation on {run_identifier_log} results...")
                print(f"Evaluating results for {run_identifier_log}...")
                eval_loading = ui_utils.LoadingAnimation(message=f"Evaluating {run_identifier_log} results")
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
                logger.warning(f"No results generated by {run_identifier_log}. Skipping evaluation.")
                
                if base_model_id_for_summary not in all_model_runs_summary:
                    all_model_runs_summary[base_model_id_for_summary] = {}
                all_model_runs_summary[base_model_id_for_summary][chosen_strategy['name']] = {
                    "error": "No results generated",
                    "num_processed": 0,
                    "total_run_time_s": total_run_time_seconds,
                    "config_id_used": config_id_loop
                }

        except Exception as model_proc_err:
            run_end_time = time.time()
            total_run_time_seconds = run_end_time - run_start_time
            if 'processing_animation' in locals() and processing_animation._thread and processing_animation._thread.is_alive():
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

def main():
    setup_logging() 
    logger.info("Starting CFA MCQ Reproducer pipeline...")
    print("Starting CFA MCQ Reproducer pipeline...")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
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

    all_model_runs_summary = {}
    logger.info("Determining evaluation run type...")

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
                all_model_runs_summary=all_model_runs_summary 
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

        print("\nPlease select which prompting strategy to use:")
        selected_strategy_key = questionary.select(
            "Select prompting strategy:",
            choices=strategy_choices
        ).ask()

        if not selected_strategy_key:
            ui_utils.print_warning("No prompting strategy selected. Exiting.")
            return
        
        if selected_model_ids and selected_strategy_key:
             print(f"\n--- Starting Custom Run: Strategy '{AVAILABLE_STRATEGIES[selected_strategy_key]['name']}' ---")
             _run_model_evaluations(
                 processed_data=processed_data,
                 selected_model_ids=selected_model_ids, 
                 strategy_key=selected_strategy_key,
                 all_model_runs_summary=all_model_runs_summary 
             )
             print(f"--- Completed Custom Run: Strategy '{AVAILABLE_STRATEGIES[selected_strategy_key]['name']}' ---")
        else:
            
             logger.warning("Skipping custom run execution due to missing selections.")
             ui_utils.print_warning("Skipping custom run execution due to missing selections.")

    
    if all_model_runs_summary:
        logger.info("\nGenerating final summary and charts...")
        print("\nGenerating final summary and charts...")
        plotting.generate_all_charts(all_model_runs_summary, config.CHARTS_DIR) 
        ui_utils.print_success(f"Charts generated successfully in {config.CHARTS_DIR}")
    else:
       logger.warning("No models were processed in this run.")
       ui_utils.print_warning("No models were processed.")
    
    logger.info("CFA MCQ Reproducer pipeline finished.")
    logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")
    ui_utils.print_info(f"CFA MCQ Reproducer pipeline finished. Results saved in: {config.RESULTS_DIR}")
    
    if all_model_runs_summary:
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
        plotting.generate_all_charts(all_model_runs_summary, config.CHARTS_DIR) 
        ui_utils.print_success(f"Charts generated successfully in {config.CHARTS_DIR}")
    else:
       logger.warning("No models were processed in this run.")
       ui_utils.print_warning("No models were processed.")
    
    logger.info("CFA MCQ Reproducer pipeline finished.")
    logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")
    ui_utils.print_info(f"CFA MCQ Reproducer pipeline finished. Results saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()