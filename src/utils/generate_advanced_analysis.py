import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src import config
except ImportError as e:
    print(f"Error importing src.config: {e}. Ensure generate_advanced_analysis.py is in src/utils/ and project root is in sys.path.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_aggregate_stats(summary_df: pd.DataFrame, output_dir: Path):
    """Calculates and saves aggregate statistics by Model and Strategy."""
    logger.info("Calculating aggregate statistics by Model and Strategy...")

    numeric_cols = ['Accuracy', 'Avg Time/Q (s)', 'Total Cost ($)', 'Total API Response Time (s)', 'Total Output Tokens']
    aggregations = ['mean', 'median', 'std']

    
    model_stats = summary_df.groupby('Model')[numeric_cols].agg(aggregations)
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats_path = output_dir / "model_aggregate_stats.csv"
    model_stats.to_csv(model_stats_path)
    logger.info(f"Saved model aggregate stats to {model_stats_path}")

    
    strategy_stats = summary_df.groupby('Strategy')[numeric_cols].agg(aggregations)
    strategy_stats.columns = ['_'.join(col).strip() for col in strategy_stats.columns.values]
    strategy_stats_path = output_dir / "strategy_aggregate_stats.csv"
    strategy_stats.to_csv(strategy_stats_path)
    logger.info(f"Saved strategy aggregate stats to {strategy_stats_path}")


def calculate_efficiency_metrics(summary_df: pd.DataFrame, output_dir: Path):
    """Calculates and saves efficiency metrics for each run."""
    logger.info("Calculating efficiency metrics...")
    cols_to_copy = ['Model', 'Strategy', 'Accuracy', 'Total Cost ($)', 'Total Output Tokens', 'Total API Response Time (s)']
    if 'Model Type' in summary_df.columns:
        cols_to_copy.append('Model Type')
    else:
        logger.warning("'Model Type' column not found in summary_df for efficiency metrics calculation. Output will not include it.")
        
    eff_df = summary_df[cols_to_copy].copy()
    eff_df['Tokens_Per_Second'] = eff_df['Total Output Tokens'] / eff_df['Total API Response Time (s)']
    eff_df['Tokens_Per_Second'] = eff_df['Tokens_Per_Second'].replace([np.inf, -np.inf], np.nan)    
    eff_df['Accuracy_Per_Dollar'] = eff_df['Accuracy'] / eff_df['Total Cost ($)']
    eff_df['Accuracy_Per_Dollar'] = eff_df['Accuracy_Per_Dollar'].replace([np.inf, -np.inf], np.nan)
    eff_df['Time_Per_Token_s'] = eff_df['Total API Response Time (s)'] / eff_df['Total Output Tokens']
    eff_df['Time_Per_Token_s'] = eff_df['Time_Per_Token_s'].replace([np.inf, -np.inf], np.nan)
    
    output_cols = ['Model', 'Strategy', 'Tokens_Per_Second', 'Accuracy_Per_Dollar', 'Time_Per_Token_s']
    if 'Model Type' in eff_df.columns: 
        output_cols.insert(2, 'Model Type') 

    eff_output_df = eff_df[output_cols]
    
    efficiency_metrics_path = output_dir / "efficiency_metrics.csv"
    eff_output_df.to_csv(efficiency_metrics_path, index=False)
    logger.info(f"Saved efficiency metrics to {efficiency_metrics_path}")

def calculate_model_type_comparison(summary_df: pd.DataFrame, output_dir: Path):
    """Calculates and saves aggregate statistics by Model Type."""
    logger.info("Calculating aggregate statistics by Model Type...")
    
    if 'Model Type' not in summary_df.columns:
        logger.warning("'Model Type' column not found in summary_df. Skipping model type comparison.")
        return

    numeric_cols = ['Accuracy', 'Avg Time/Q (s)', 'Total Cost ($)']
    aggregations = ['mean', 'median', 'std']
    
    model_type_stats = summary_df.groupby('Model Type')[numeric_cols].agg(aggregations)
    model_type_stats.columns = ['_'.join(col).strip() for col in model_type_stats.columns.values]
    
    model_type_stats_path = output_dir / "model_type_aggregate_stats.csv"
    model_type_stats.to_csv(model_type_stats_path)
    logger.info(f"Saved model type aggregate stats to {model_type_stats_path}")

def main():
    logger.info("Starting advanced analysis generation...")
    results_dir = Path(config.RESULTS_DIR)
    summary_csv_path = results_dir / "all_runs_summary_metrics.csv"
    advanced_analysis_dir = results_dir / "advanced_analysis"
    advanced_analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Advanced analysis output directory: {advanced_analysis_dir}")

    if not summary_csv_path.exists():
        logger.error(f"Summary CSV file not found: {summary_csv_path}. Cannot perform advanced analysis.")
        return
    
    try:
        summary_df = pd.read_csv(summary_csv_path)
        logger.info(f"Successfully loaded summary CSV from {summary_csv_path} with {len(summary_df)} rows.")
    except pd.errors.EmptyDataError:
        logger.error(f"Summary CSV file {summary_csv_path} is empty. Cannot perform advanced analysis.")
        return
    except Exception as e:
        logger.error(f"Error loading summary CSV {summary_csv_path}: {e}")
        return

    calculate_aggregate_stats(summary_df.copy(), advanced_analysis_dir)
    calculate_efficiency_metrics(summary_df.copy(), advanced_analysis_dir)
    calculate_model_type_comparison(summary_df.copy(), advanced_analysis_dir)

    logger.info("Advanced analysis generation finished.")

if __name__ == "__main__":
    main() 