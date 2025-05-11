"""
Main script for the CFA MCQ Reproducer pipeline.

Orchestrates the loading of data, processing with LLMs via different strategies,
_evaluation, and generation of comparison charts.
"""
import json
import logging
import numpy as np
from . import config
from . import evaluation
from . import plotting
from . import ui_utils
from . import configs
import questionary


from .strategies import default as default_strategy
from .strategies import self_consistency as sc_strategy

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
        "params": {"n_samples": 3}
    },
    "self_consistency_cot_n5": {
        "function": sc_strategy.run_self_consistency_strategy,
        "name": "Self-Consistency CoT (N=5 samples)",
        "params": {"n_samples": 5}
    }
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

    print("\nPreparing available LLM models...")
    model_loading_anim = ui_utils.LoadingAnimation(message="Loading model configurations") 
    model_loading_anim.start()

    # Decide which config list to use based on the selected strategy later
    # For now, we'll prepare to merge or select them.
    
    # Temporary: For initial model listing, we can show all available for simplicity,
    # or decide on a primary list. Let's use DEFAULT_CONFIGS as the base for selection for now.
    # We will refine this to be strategy-dependent.
    temp_all_model_configs = configs.DEFAULT_CONFIGS + configs.COT_CONFIGS
    # Create a set of config_ids to avoid duplicates if a model is in both lists (though unlikely with current naming)
    seen_config_ids = set()
    unique_model_configs_for_listing = []
    for model_conf in temp_all_model_configs:
        conf_id = model_conf.get('config_id', model_conf.get('model_id'))
        if conf_id not in seen_config_ids:
            unique_model_configs_for_listing.append(model_conf)
            seen_config_ids.add(conf_id)

    model_choices = [
        {
            "name": f"{m.get('config_id', m.get('model_id'))} ({m.get('type')})",
            "value": m.get('config_id', m.get('model_id'))
        }
        for m in unique_model_configs_for_listing # Use the de-duplicated list
    ]
    model_choices.insert(0, {"name": "[Run All Available Models]", "value": "__ALL__"})
    model_loading_anim.stop()
    ui_utils.print_info(f"Found {len(unique_model_configs_for_listing)} unique model configurations across default and CoT sets.")

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
    
    chosen_strategy = AVAILABLE_STRATEGIES[selected_strategy_key]
    strategy_func = chosen_strategy["function"]
    strategy_params = chosen_strategy["params"]
    strategy_name_for_file = selected_strategy_key.replace(" ", "_").lower()
    ui_utils.print_info(f"Using strategy: {chosen_strategy['name']}")

    # Determine which set of model configs to use based on strategy
    if "cot" in selected_strategy_key.lower(): # Simple check for CoT strategies
        print(f"Strategy '{chosen_strategy['name']}' selected. Using CoT-specific model configurations.")
        logger.info(f"Using CoT-specific model configurations from src.configs.cot_config")
        relevant_model_configs_source = configs.COT_CONFIGS
    else:
        print(f"Strategy '{chosen_strategy['name']}' selected. Using default model configurations.")
        logger.info(f"Using default model configurations from src.configs.default_config")
        relevant_model_configs_source = configs.DEFAULT_CONFIGS

    all_model_runs_summary = {} 
    logger.info("Starting LLM processing and evaluation loop...")

    # Filter selected models from the chosen configuration source
    if "__ALL__" in selected_model_ids:
        active_model_configs = relevant_model_configs_source
        if not active_model_configs:
            print(f"Warning: No models found in the selected configuration set for strategy '{chosen_strategy['name']}'.")
            logger.warning(f"No models found in the selected configuration set for strategy '{chosen_strategy['name']}'.")
    else:
        active_model_configs = [
            m for m in relevant_model_configs_source
            if m.get('config_id', m.get('model_id')) in selected_model_ids
        ]
        if not active_model_configs:
             # This could happen if user selected models from the combined list but the strategy's specific config list doesn't have them.
            print(f"Warning: None of the selected models are present in the configuration set for strategy '{chosen_strategy['name']}'. Selected IDs: {selected_model_ids}")
            logger.warning(f"None of the selected models are present in the configuration set for strategy '{chosen_strategy['name']}'. Selected IDs: {selected_model_ids}")

    for model_config_item in active_model_configs:
        config_id_loop = model_config_item.get("config_id", model_config_item.get("model_id"))
        model_type_loop = model_config_item.get("type")
        run_identifier = f"{config_id_loop}__{strategy_name_for_file}" 
        logger.info(f"\n{'='*20} Processing Model: {config_id_loop} ({model_type_loop}) with Strategy: {chosen_strategy['name']} {'='*20}")
        data_for_this_run = processed_data 
        credentials_ok = True
        # These API key checks are now dynamic based on the active_model_configs being processed
        models_of_type_bedrock = any(m.get('type') == 'bedrock' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_sagemaker = any(m.get('type') == 'sagemaker' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_openai = any(m.get('type') == 'openai' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_gemini = any(m.get('type') == 'gemini' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_xai = any(m.get('type') == 'xai' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_writer = any(m.get('type') == 'writer' for m in active_model_configs if m.get('config_id') == config_id_loop)
        models_of_type_groq = any(m.get('type') == 'groq' for m in active_model_configs if m.get('config_id') == config_id_loop)

        if (models_of_type_bedrock or models_of_type_sagemaker) and \
           (not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY):
            logger.error(f"Skipping {config_id_loop}: Missing AWS credentials.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (Bedrock/SageMaker): Missing AWS credentials.")
            credentials_ok = False
        elif models_of_type_openai and not config.OPENAI_API_KEY:
            logger.error(f"Skipping {config_id_loop}: Missing OpenAI API key.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (OpenAI): Missing OpenAI API key.")
            credentials_ok = False
        elif models_of_type_gemini and not config.GEMINI_API_KEY:
            logger.error(f"Skipping {config_id_loop}: Missing Gemini API key.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (Gemini): Missing Gemini API key.")
            credentials_ok = False
        elif models_of_type_xai and not config.XAI_API_KEY:
            logger.error(f"Skipping {config_id_loop}: Missing xAI API key.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (xAI): Missing xAI API key.")
            credentials_ok = False
        elif models_of_type_writer and not config.WRITER_API_KEY:
            logger.error(f"Skipping {config_id_loop}: Missing Writer API key.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (Writer): Missing Writer API key.")
            credentials_ok = False
        elif models_of_type_groq and not config.GROQ_API_KEY:
            logger.error(f"Skipping {config_id_loop}: Missing Groq API key.")
            ui_utils.print_warning(f"Skipping {config_id_loop} (Groq): Missing Groq API key.")
            credentials_ok = False

        if not credentials_ok:
            all_model_runs_summary[run_identifier] = {"error": "Missing credentials", "num_processed": 0, "model": config_id_loop, "strategy": chosen_strategy['name']}
            continue

        try:
            logger.info(f"Processing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")
            print(f"\nProcessing {len(data_for_this_run)} questions with {config_id_loop} using {chosen_strategy['name']}...")            
            processing_animation = ui_utils.LoadingAnimation(message=f"Processing with {config_id_loop} ({chosen_strategy['name']})")
            processing_animation.start()
            llm_run_results_data = strategy_func(data_for_this_run, model_config_item, **strategy_params)
            processing_animation.stop()
            ui_utils.print_success(f"Completed processing {len(data_for_this_run)} questions with {config_id_loop} ({chosen_strategy['name']})")

            avg_time_per_question = None 
            total_in_tokens = 0
            total_out_tokens = 0
            avg_answer_len = 0.0 

            if llm_run_results_data: 
                valid_times = [r.get('response_time') for r in llm_run_results_data if r.get('response_time') is not None]
                avg_time_per_question = np.mean(valid_times) if valid_times else None

                valid_in_tokens = [r.get('input_tokens') for r in llm_run_results_data if r.get('input_tokens') is not None]
                total_in_tokens = sum(valid_in_tokens) if valid_in_tokens else 0

                valid_out_tokens = [r.get('output_tokens') for r in llm_run_results_data if r.get('output_tokens') is not None]
                total_out_tokens = sum(valid_out_tokens) if valid_out_tokens else 0
                
                num_results_for_len = len(llm_run_results_data)
                
                avg_answer_len = np.mean([r.get('answer_length', 0) for r in llm_run_results_data]) if num_results_for_len > 0 else 0.0

            model_response_filename = config.RESULTS_DIR / f"response_data_{run_identifier}.json"
            try:
                with open(model_response_filename, 'w', encoding='utf-8') as f:
                    json.dump(llm_run_results_data, f, indent=4)
                logger.info(f"LLM responses for {run_identifier} saved to {model_response_filename}")
                ui_utils.print_success(f"Results for {run_identifier} saved.")
            except Exception as e_save:
                logger.error(f"Failed to save LLM results for {run_identifier} to {model_response_filename}: {e_save}", exc_info=True)
                ui_utils.print_error(f"Failed to save results for {run_identifier}.")

            if llm_run_results_data:
                logger.info(f"Performing final evaluation on {run_identifier} results...")
                print(f"Evaluating results for {run_identifier}...")
                eval_loading = ui_utils.LoadingAnimation(message=f"Evaluating {run_identifier} results")
                eval_loading.start()
                
                classification_metrics = evaluation.evaluate_classification(llm_run_results_data)
                eval_loading.stop()

                all_model_runs_summary[run_identifier] = {
                    "model": config_id_loop,
                    "strategy": chosen_strategy['name'],
                    "metrics": classification_metrics,
                    "avg_time_per_question": avg_time_per_question, 
                    "total_input_tokens": total_in_tokens,
                    "total_output_tokens": total_out_tokens,
                    "avg_answer_length": avg_answer_len,
                    "num_processed": len(llm_run_results_data),
                    "results_file": str(model_response_filename)
                }
                accuracy = classification_metrics.get('accuracy', 0.0)
                logger.info(f"Evaluation summary for {run_identifier}: Accuracy: {accuracy:.4f}")
                ui_utils.print_success(f"Evaluation complete for {run_identifier}: Accuracy: {accuracy:.4f}")
            else:
                logger.warning(f"No results generated by {run_identifier}. Skipping evaluation.")
                all_model_runs_summary[run_identifier] = {
                    "error": "No results generated", 
                    "num_processed": 0,
                    "model": config_id_loop, 
                    "strategy": chosen_strategy['name']
                }
        except Exception as model_proc_err:  
            if 'processing_animation' in locals() and processing_animation._thread and processing_animation._thread.is_alive():
                processing_animation.stop()
            logger.error(f"An error occurred while processing {run_identifier}: {model_proc_err}", exc_info=True)
            ui_utils.print_error(f"Error processing {run_identifier}: {model_proc_err}")
            all_model_runs_summary[run_identifier] = {"error": str(model_proc_err), "num_processed": 0, "model": config_id_loop, "strategy": chosen_strategy['name']}    
    logger.info("\n" + "="*30 + " Overall Run Comparison " + "="*30)
    summary_loading = ui_utils.LoadingAnimation(message="Preparing results summary")
    summary_loading.start()
    header = "| {:<45} | {:<25} | {:>8} | {:>12} | {:>19} | {:>18} |".format(
        "Model", "Strategy", "Accuracy", "Avg Time/Q (s)", "Total Output Tokens", "Avg Ans Len"
    )
    logger.info(header)
    logger.info("|" + "-"*47 + "|" + "-"*27 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*21 + "|" + "-"*20 + "|")

    summary_rows_for_print = []
    for run_id, summary_data in all_model_runs_summary.items():
        model_disp = summary_data.get("model", run_id.split("__")[0] if "__" in run_id else run_id)
        strategy_disp = summary_data.get("strategy", run_id.split("__")[1] if "__" in run_id else "N/A")
        if "error" in summary_data:
            row_str = "| {:<45} | {:<25} | {:^8} | {:^12} | {:^19} | {:^18} |".format(
                model_disp, strategy_disp, "FAILED", "---", "---", "---"
            )
        else:
            metrics = summary_data.get("metrics", {})
            acc = metrics.get('accuracy', 0.0)
            time_s = summary_data.get('avg_time_per_question') 
            time_str = f"{time_s:.2f}" if time_s is not None else "N/A"
            tokens = summary_data.get('total_output_tokens')
            token_str = f"{tokens:.0f}" if tokens is not None else "N/A"
            ans_len = summary_data.get('avg_answer_length')
            ans_len_str = f"{ans_len:.1f}" if ans_len is not None else "N/A"

            row_str = "| {:<45} | {:<25} | {:>8.4f} | {:>12} | {:>19} | {:>18} |".format(
                model_disp, strategy_disp, acc, time_str, token_str, ans_len_str
            )
        logger.info(row_str)
        summary_rows_for_print.append(row_str)
    logger.info("="*(47+27+10+14+21+20+7))
    summary_loading.stop()
    print("\n" + "="*30 + " Overall Run Comparison " + "="*30)
    print(header)
    print("|" + "-"*47 + "|" + "-"*27 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*21 + "|" + "-"*20 + "|")
    for row in summary_rows_for_print:
        print(row)
    print("="*(47+27+10+14+21+20+7))

    if any('metrics' in res for res in all_model_runs_summary.values()):
        print("\nGenerating comparison charts...")
        chart_loading = ui_utils.LoadingAnimation(message="Generating comparison charts")
        chart_loading.start()  
        plotting.generate_all_charts(all_model_runs_summary, config.CHARTS_DIR)
        chart_loading.stop()
        ui_utils.print_success(f"Charts generated successfully in {config.CHARTS_DIR}")
    else:
        logger.warning("No successful model runs with metrics to plot.")
        ui_utils.print_warning("No model results with metrics to plot.")
    logger.info("CFA MCQ Reproducer pipeline finished.")
    logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")
    ui_utils.print_info(f"CFA MCQ Reproducer pipeline finished. Results saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()