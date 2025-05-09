"""
Main script for the CFA MCQ Reproducer pipeline.

Orchestrates the loading of data, processing with LLMs, evaluation,
and generation of comparison charts.
"""

import json
import logging
import numpy as np 
from pathlib import Path
from . import config 
from . import llm_clients
from . import evaluation
from . import plotting

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting CFA MCQ Reproducer pipeline...")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True) 
    processed_data = None
    if config.FILLED_JSON_PATH.exists():
        logger.info(f"Loading processed data from: {config.FILLED_JSON_PATH}")
        try:
            with open(config.FILLED_JSON_PATH, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            if not isinstance(processed_data, list) or not processed_data:
                logger.error(f"{config.FILLED_JSON_PATH} is empty or not a valid JSON list. Exiting.")
                return
            logger.info(f"Successfully loaded {len(processed_data)} entries.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {config.FILLED_JSON_PATH}. Exiting.", exc_info=True)
            return
        except Exception as e:
            logger.error(f"Error reading {config.FILLED_JSON_PATH}: {e}. Exiting.", exc_info=True)
            return
    else:
        logger.error(f"Input data file not found: {config.FILLED_JSON_PATH}. Exiting.")
        return

    
    all_model_results_summary = {}
    logger.info("Starting LLM processing and evaluation loop...")

    for model_config_item in config.ALL_MODEL_CONFIGS:
        config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
        model_type = model_config_item.get("type")
        logger.info(f"\n{'='*20} Processing Model: {config_id} ({model_type}) {'='*20}")

        
        credentials_ok = True
        if (model_type == "bedrock" or model_type == "sagemaker") and \
           (not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY):
            logger.error(f"Skipping {config_id}: Missing AWS credentials.")
            credentials_ok = False
        elif model_type == "openai" and not config.OPENAI_API_KEY:
            logger.error(f"Skipping {config_id}: Missing OpenAI API key.")
            credentials_ok = False
        elif model_type == "gemini" and not config.GEMINI_API_KEY:
            logger.error(f"Skipping {config_id}: Missing Gemini API key.")
            credentials_ok = False
        elif model_type == "xai" and not config.XAI_API_KEY:
            logger.error(f"Skipping {config_id}: Missing xAI API key.")
            credentials_ok = False
        elif model_type == "writer" and not config.WRITER_API_KEY:
            logger.error(f"Skipping {config_id}: Missing Writer API key.")
            credentials_ok = False
        
        if not credentials_ok:
            all_model_results_summary[config_id] = {"error": "Missing credentials", "num_processed": 0}
            continue

        try:
            logger.info(f"Processing {len(processed_data)} questions with {config_id}...")
            
            llm_run_results_data = llm_clients.process_questions_with_llm(processed_data, model_config_item)

            avg_time = None
            total_in_tokens = 0
            total_out_tokens = 0
            avg_answer_len = 0.0

            if llm_run_results_data:
                valid_times = [r.get('response_time') for r in llm_run_results_data if r.get('response_time') is not None]
                avg_time = np.mean(valid_times) if valid_times else None

                valid_in_tokens = [r.get('input_tokens') for r in llm_run_results_data if r.get('input_tokens') is not None]
                total_in_tokens = sum(valid_in_tokens) if valid_in_tokens else 0

                valid_out_tokens = [r.get('output_tokens') for r in llm_run_results_data if r.get('output_tokens') is not None]
                total_out_tokens = sum(valid_out_tokens) if valid_out_tokens else 0
                
                
                num_results_for_len = len(llm_run_results_data)
                avg_answer_len = np.mean([r.get('answer_length', 0) for r in llm_run_results_data]) if num_results_for_len > 0 else 0.0
            
            model_response_filename = config.RESULTS_DIR / f"response_data_{config_id}.json"
            try:
                with open(model_response_filename, 'w', encoding='utf-8') as f:
                    json.dump(llm_run_results_data, f, indent=4)
                logger.info(f"LLM responses for {config_id} saved to {model_response_filename}")
            except Exception as e_save:
                logger.error(f"Failed to save LLM results for {config_id} to {model_response_filename}: {e_save}", exc_info=True)

            if llm_run_results_data:
                logger.info(f"Performing final evaluation on {config_id} results...")                
                classification_metrics = evaluation.evaluate_classification(llm_run_results_data)
                
                all_model_results_summary[config_id] = {
                    "metrics": classification_metrics,
                    "avg_response_time": avg_time,
                    "total_input_tokens": total_in_tokens,
                    "total_output_tokens": total_out_tokens,
                    "avg_answer_length": avg_answer_len,
                    "num_processed": len(llm_run_results_data),
                    "results_file": str(model_response_filename)
                }
                logger.info(f"Evaluation summary for {config_id}: Accuracy: {classification_metrics.get('accuracy'):.4f}")
            else:
                logger.warning(f"No results generated by {config_id}. Skipping evaluation.")
                all_model_results_summary[config_id] = {
                    "error": "No results generated", "num_processed": 0
                }
        except Exception as model_proc_err:
            logger.error(f"An error occurred while processing model {config_id}: {model_proc_err}", exc_info=True)
            all_model_results_summary[config_id] = {"error": str(model_proc_err), "num_processed": 0}
            

    
    logger.info("\n" + "="*30 + " Overall Model Comparison " + "="*30)
    
    header = "| {:<30} | {:>8} | {:>12} | {:>19} | {:>18} |".format(
        "Model", "Accuracy", "Avg Time (s)", "Total Output Tokens", "Avg Answer Length"
    )
    logger.info(header)
    logger.info("|" + "-"*32 + "|" + "-"*10 + "|" + "-"*14 + "|" + "-"*21 + "|" + "-"*20 + "|")

    for config_id_summary, summary_results in all_model_results_summary.items():
        if "error" in summary_results:
            row = "| {:<30} | {:^8} | {:^12} | {:^19} | {:^18} |".format(
                config_id_summary, "FAILED", "---", "---", "---"
            )
        else:
            metrics = summary_results.get("metrics", {})
            acc = metrics.get('accuracy', 0.0)
            time_s = summary_results.get('avg_response_time')
            time_str = f"{time_s:.2f}" if time_s is not None else "N/A"
            tokens = summary_results.get('total_output_tokens')
            token_str = f"{tokens:.0f}" if tokens is not None else "N/A"
            ans_len = summary_results.get('avg_answer_length')
            ans_len_str = f"{ans_len:.1f}" if ans_len is not None else "N/A"
            
            row = "| {:<30} | {:>8.4f} | {:>12} | {:>19} | {:>18} |".format(
                config_id_summary, acc, time_str, token_str, ans_len_str
            )
        logger.info(row)
    logger.info("="*110)

    
    if all_model_results_summary:
        plotting.generate_all_charts(all_model_results_summary, config.CHARTS_DIR)
    else:
        logger.warning("No model results to plot.")

    logger.info("CFA MCQ Reproducer pipeline finished.")
    logger.info(f"Results and charts saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main() 