import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import re
from pathlib import Path
import pandas as pd
from src import config 
from src.evaluations import classification as classification_eval
from src.evaluations import resource_metrics as resource_eval
from src.evaluations import cost_evaluation
from src.configs import default_config, cot_config, self_discover_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALL_MODEL_CONFIG_LISTS = {
    "default": default_config.ALL_MODEL_CONFIGS if hasattr(default_config, 'ALL_MODEL_CONFIGS') else [],
    "cot": cot_config.ALL_MODEL_CONFIGS_COT if hasattr(cot_config, 'ALL_MODEL_CONFIGS_COT') else [], 
    "self_discover": self_discover_config.SELF_DISCOVER_CONFIGS if hasattr(self_discover_config, 'SELF_DISCOVER_CONFIGS') else []
}

REASONING_MODEL_CONFIG_IDS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash", 
    "o3-mini",
    "o4-mini",
    "grok-3",
    "grok-3-mini-beta-high-effort", 
    "deepseek-r1",   
    "gemini-2.5-pro-cot",
    "gemini-2.5-flash-cot",
    "claude-3.7-sonnet-cot",
    "grok-3-cot",
    "deepseek-r1-cot",   
    "gpt-4o-self-discover",
    "gpt-o3-self-discover", 
    "grok-3-self-discover",
    "gemini-2.5-pro-self-discover",
    "gemini-2.5-flash-self-discover",
    "deepseek-r1-self-discover",  
    "grok-3-mini-beta-self-discover-high-effort" 
]

REASONING_MODEL_CONFIG_IDS = sorted(list(set(REASONING_MODEL_CONFIG_IDS)))

def get_model_type(config_id_from_filename: str) -> str:
    """Determines if a model is 'Reasoning' or 'Non-Reasoning'."""
    if config_id_from_filename in REASONING_MODEL_CONFIG_IDS:
        return "Reasoning"
    return "Non-Reasoning"

def get_model_config(config_id_from_filename: str):
    """Fetches the model configuration item based on config_id."""
    for _, config_list in ALL_MODEL_CONFIG_LISTS.items():
        for model_conf in config_list:
            if model_conf.get("config_id", model_conf.get("model_id")) == config_id_from_filename:
                return model_conf
    logger.warning(f"Model configuration not found for config_id: {config_id_from_filename}")
    return None

def get_pretty_strategy_name(strategy_key_from_filename: str) -> str:
    """Converts a strategy key from filename to a pretty display name."""
    if strategy_key_from_filename == "default":
        return "Default (Single Pass)"
    elif strategy_key_from_filename == "self_consistency_cot_n3":
        return "Self-Consistency CoT (N=3 samples)"
    elif strategy_key_from_filename == "self_consistency_cot_n5":
        return "Self-Consistency CoT (N=5 samples)"
    elif strategy_key_from_filename == "self_discover":
        return "Self-Discover"
    logger.warning(f"No pretty name mapping for strategy key: {strategy_key_from_filename}. Using key as name.")
    return strategy_key_from_filename

def get_base_model_id(config_id_used: str) -> str:
    """Extracts a base model ID for display purposes."""
    if config_id_used.endswith('-cot'):
        return config_id_used[:-len('-cot')]
    elif config_id_used.endswith('-self-discover'):
        return config_id_used[:-len('-self-discover')]
    return config_id_used

def parse_filename(filename: Path):
    """Parses config_id and strategy_key from the JSON filename."""
    match = re.fullmatch(r"response_data_(.*?)__(.*?)\.json", filename.name)
    if match:
        config_id = match.group(1)
        strategy_key = match.group(2)
        return config_id, strategy_key
    logger.warning(f"Could not parse config_id and strategy from filename: {filename.name}")
    return None, None

def main():
    logger.info("Starting summary CSV update process from JSON results...")
    
    base_json_dir = config.RESULTS_DIR / "json"

    results_output_dir = Path(config.RESULTS_DIR)
    if not results_output_dir.exists():
        results_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created results directory: {results_output_dir}")
        
    summary_csv_path = results_output_dir / "all_runs_summary_metrics.csv"

    csv_columns = [
        'Model', 'Strategy', 'Accuracy', 'Avg Time/Q (s)', 
        'Total API Response Time (s)', 'Total Output Tokens', 
        'Avg Ans Len', 'Total Cost ($)', 'Num Processed', 
        'Config ID Used', 'Source JSON', 'Model Type'
    ]

    if summary_csv_path.exists():
        try:
            summary_df = pd.read_csv(summary_csv_path)
            for col in csv_columns:
                if col not in summary_df.columns:
                    summary_df[col] = None 
            logger.info(f"Loaded existing summary CSV: {summary_csv_path}")
        except pd.errors.EmptyDataError:
            logger.warning(f"Summary CSV {summary_csv_path} is empty. Initializing new DataFrame.")
            summary_df = pd.DataFrame(columns=csv_columns)
        except Exception as e:
            logger.error(f"Error loading summary CSV {summary_csv_path}: {e}. Initializing new DataFrame.")
            summary_df = pd.DataFrame(columns=csv_columns)
    else:
        logger.info("Summary CSV not found. Initializing new DataFrame.")
        summary_df = pd.DataFrame(columns=csv_columns)

    json_files = []
    if base_json_dir.exists() and base_json_dir.is_dir():
        logger.info(f"Recursively scanning directory: {base_json_dir} for response_data_*.json files.")
        json_files = list(base_json_dir.rglob("response_data_*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {base_json_dir} and its subdirectories.")
    else:
        logger.warning(f"Base JSON directory not found or is not a directory: {base_json_dir}")
            
    if not json_files:
        logger.info(f"No response_data_*.json files found in {base_json_dir} and its subdirectories.")
        if not summary_df.empty or not summary_csv_path.exists():
            try:
                for col in csv_columns:
                    if col not in summary_df.columns:
                        summary_df[col] = None
                summary_df.to_csv(summary_csv_path, index=False)
                logger.info(f"Summary CSV (possibly empty or with new columns) saved to {summary_csv_path}")
            except Exception as e:
                logger.error(f"Error saving summary CSV to {summary_csv_path}: {e}")
        return

    logger.info(f"Found a total of {len(json_files)} JSON files to process.")
    
    new_rows = []

    for json_file_path in json_files:
        logger.info(f"Processing {json_file_path.name}...")
        config_id_used, strategy_key_from_filename = parse_filename(json_file_path)

        if not config_id_used or not strategy_key_from_filename:
            continue

        model_config_item = get_model_config(config_id_used)
        if not model_config_item:
            logger.warning(f"Skipping {json_file_path.name} as model config for '{config_id_used}' was not found.")
            continue
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            if not isinstance(results_data, list):
                logger.warning(f"Data in {json_file_path.name} is not a list. Skipping.")
                continue
            if not results_data:
                logger.warning(f"Data in {json_file_path.name} is empty. Skipping.")
                continue
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_file_path.name}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error reading {json_file_path.name}: {e}. Skipping.")
            continue

        num_processed = len(results_data)
        classification_metrics = classification_eval.evaluate_classification(results_data)
        resource_usage_metrics = resource_eval.calculate_resource_usage(results_data, model_config_item)
        cost_metrics = cost_evaluation.calculate_model_cost(results_data, model_config_item)

        accuracy = classification_metrics.get('accuracy', 0.0)
        avg_latency_ms = resource_usage_metrics.get('average_latency_ms', 0.0)
        avg_time_q_s = avg_latency_ms / 1000.0 if avg_latency_ms is not None else 0.0
        
        total_api_response_time_s = sum(item.get('response_time', 0) for item in results_data if isinstance(item.get('response_time'), (int, float)))

        total_output_tokens = resource_usage_metrics.get('total_output_tokens', 0)
        avg_ans_len = resource_usage_metrics.get('avg_answer_length', 0.0)
        total_cost = cost_metrics.get('total_cost', 0.0)
        
        pretty_strategy = get_pretty_strategy_name(strategy_key_from_filename)
        base_model = get_base_model_id(config_id_used)
        model_type = get_model_type(config_id_used)

        new_row_data = {
            'Model': base_model,
            'Strategy': pretty_strategy,
            'Accuracy': accuracy,
            'Avg Time/Q (s)': avg_time_q_s,
            'Total API Response Time (s)': total_api_response_time_s,
            'Total Output Tokens': total_output_tokens,
            'Avg Ans Len': avg_ans_len,
            'Total Cost ($)': total_cost,
            'Num Processed': num_processed,
            'Config ID Used': config_id_used,
            'Source JSON': json_file_path.name,
            'Model Type': model_type
        }
        new_rows.append(new_row_data)

    if not new_rows:
        logger.info("No new data processed from JSON files to update the summary.")
        if not summary_df.empty:
             summary_df.to_csv(summary_csv_path, index=False)
             logger.info(f"Original summary CSV (potentially with new columns) saved to {summary_csv_path}")
        return
    updates_df = pd.DataFrame(new_rows)
    if not summary_df.empty:
        summary_df = summary_df.set_index(['Config ID Used', 'Strategy'], drop=False)
        updates_df = updates_df.set_index(['Config ID Used', 'Strategy'], drop=False)

        for idx, row_to_upsert in updates_df.iterrows():
            if idx in summary_df.index:
                existing_row = summary_df.loc[idx]
                if isinstance(existing_row, pd.DataFrame): existing_row = existing_row.iloc[0]
                existing_accuracy = existing_row.get('Accuracy', 0.0)
                existing_num_processed = existing_row.get('Num Processed', 0)
                new_accuracy = row_to_upsert.get('Accuracy', 0.0)
                new_num_processed = row_to_upsert.get('Num Processed', 0)

                should_update = False
                if new_accuracy > existing_accuracy:
                    should_update = True
                elif new_accuracy == existing_accuracy and new_num_processed > existing_num_processed:
                    should_update = True
                
                if should_update:
                    logger.info(f"Updating existing entry for {idx}: Acc {existing_accuracy}->{new_accuracy}, NumProc {existing_num_processed}->{new_num_processed}")
                    for col in row_to_upsert.index: 
                         if col in summary_df.columns:
                            summary_df.loc[idx, col] = row_to_upsert[col]
                else:
                    logger.info(f"Skipping update for {idx}: New (Acc {new_accuracy}, NumProc {new_num_processed}) not better than existing (Acc {existing_accuracy}, NumProc {existing_num_processed}).")
            else:
                pass 
        summary_df = summary_df.reset_index(drop=True)        
        updates_df = updates_df.reset_index(drop=True)
        summary_df['_merge_key'] = summary_df['Config ID Used'] + "_" + summary_df['Strategy']
        updates_df['_merge_key'] = updates_df['Config ID Used'] + "_" + updates_df['Strategy']
         
        if 'Model Type' not in summary_df.columns and not summary_df.empty:
            summary_df['Model Type'] = updates_df.set_index('_merge_key')['Model Type'].reindex(summary_df['_merge_key']).values

        new_entries_df = updates_df[~updates_df['_merge_key'].isin(summary_df['_merge_key'])]
        
        final_df = pd.concat([summary_df, new_entries_df], ignore_index=True)
        final_df = final_df.drop(columns=['_merge_key'])
        
    else: 
        final_df = updates_df.reset_index(drop=True)

    final_df = final_df.reindex(columns=csv_columns)
    sort_by_columns = []
    if 'Strategy' in final_df.columns:
        sort_by_columns.append('Strategy')
    if 'Model' in final_df.columns:
        sort_by_columns.append('Model')
    if 'Config ID Used' in final_df.columns:
        sort_by_columns.append('Config ID Used')

    if 'Strategy' in sort_by_columns and 'Model' in sort_by_columns:
        logger.info(f"Sorting final DataFrame by {sort_by_columns}.")
        final_df = final_df.sort_values(by=sort_by_columns, ascending=True)
        final_df = final_df.reset_index(drop=True) 
    else:
        logger.warning(
            f"Skipping primary sort as 'Strategy' or 'Model' column is missing. "
            f"Available columns for sorting: {sort_by_columns}. All columns: {final_df.columns.tolist()}"
        )

    try:
        final_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Summary CSV updated successfully: {summary_csv_path}")
    except Exception as e:
        logger.error(f"Error saving updated summary CSV to {summary_csv_path}: {e}")

if __name__ == "__main__":
    main()

