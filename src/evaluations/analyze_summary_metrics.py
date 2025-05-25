import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
charts_output_dir = os.path.join(script_dir, "..", "..", "results", "analysis_charts")
csv_output_dir = os.path.join(script_dir, "..", "..", "results", "IJCAI_RESULTS")
os.makedirs(charts_output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

def load_data(csv_path):
    """Loads the summary metrics CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {os.path.basename(csv_path)}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return None

def overall_performance_analysis(df, charts_dir, csv_dir):
    """Analyzes overall performance and saves plots and CSVs."""
    if df is None:
        return

    print("\n--- Overall Performance Analysis ---")

    # Top 10 models by Accuracy
    top_10_accuracy = df.sort_values(by="Accuracy", ascending=False).head(10)
    print("\nTop 10 Models by Accuracy:")
    print(top_10_accuracy[["Model", "Strategy", "Accuracy", "Total Cost ($)", "Avg Time/Q (s)"]])
    top_10_accuracy.to_csv(os.path.join(csv_dir, "top_10_accuracy.csv"), index=False)
    print(f"Saved top_10_accuracy.csv to {csv_dir}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_10_accuracy, x="Model", y="Accuracy", hue="Strategy")
    plt.title("Top 10 Models by Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "top_10_accuracy.png"))
    plt.close()
    print(f"Saved top_10_accuracy.png to {charts_dir}")

    # Lowest 5 models by Cost
    lowest_5_cost = df.sort_values(by="Total Cost ($)", ascending=True).head(5)
    print("\nLowest 5 Models by Total Cost ($):")
    print(lowest_5_cost[["Model", "Strategy", "Accuracy", "Total Cost ($)"]])
    lowest_5_cost.to_csv(os.path.join(csv_dir, "lowest_5_cost.csv"), index=False)
    print(f"Saved lowest_5_cost.csv to {csv_dir}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=lowest_5_cost, x="Model", y="Total Cost ($)", hue="Strategy")
    plt.title("Lowest 5 Models by Total Cost ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "lowest_5_cost.png"))
    plt.close()
    print(f"Saved lowest_5_cost.png to {charts_dir}")

    # Fastest 5 models by Average Time per Question
    fastest_5_time = df.sort_values(by="Avg Time/Q (s)", ascending=True).head(5)
    print("\nFastest 5 Models by Avg Time/Q (s):")
    print(fastest_5_time[["Model", "Strategy", "Accuracy", "Avg Time/Q (s)"]])
    fastest_5_time.to_csv(os.path.join(csv_dir, "fastest_5_time.csv"), index=False)
    print(f"Saved fastest_5_time.csv to {csv_dir}")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=fastest_5_time, x="Model", y="Avg Time/Q (s)", hue="Strategy")
    plt.title("Fastest 5 Models by Avg Time/Q (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "fastest_5_time.png"))
    plt.close()
    print(f"Saved fastest_5_time.png to {charts_dir}")


def model_type_analysis(df, charts_dir, csv_dir):
    """Analyzes performance based on model type (Reasoning vs. Non-Reasoning) and saves plots and CSVs."""
    if df is None or "Model Type" not in df.columns:
        print("Model Type column not found. Skipping model type analysis.")
        return

    print("\n--- Model Type Analysis (Reasoning vs. Non-Reasoning) ---")
    
    # Average accuracy by model type
    avg_accuracy_by_type = df.groupby("Model Type")["Accuracy"].mean().reset_index()
    print("\nAverage Accuracy by Model Type:")
    print(avg_accuracy_by_type)
    avg_accuracy_by_type.to_csv(os.path.join(csv_dir, "avg_accuracy_by_model_type.csv"), index=False)
    print(f"Saved avg_accuracy_by_model_type.csv to {csv_dir}")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_accuracy_by_type, x="Model Type", y="Accuracy")
    plt.title("Average Accuracy by Model Type")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_accuracy_by_model_type.png"))
    plt.close()
    print(f"Saved avg_accuracy_by_model_type.png to {charts_dir}")

    # Average cost by model type
    avg_cost_by_type = df.groupby("Model Type")["Total Cost ($)"].mean().reset_index()
    print("\nAverage Total Cost ($) by Model Type:")
    print(avg_cost_by_type)
    avg_cost_by_type.to_csv(os.path.join(csv_dir, "avg_cost_by_model_type.csv"), index=False)
    print(f"Saved avg_cost_by_model_type.csv to {csv_dir}")
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_cost_by_type, x="Model Type", y="Total Cost ($)")
    plt.title("Average Total Cost ($) by Model Type")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_cost_by_model_type.png"))
    plt.close()
    print(f"Saved avg_cost_by_model_type.png to {charts_dir}")

    # Average time by model type
    avg_time_by_type = df.groupby("Model Type")["Avg Time/Q (s)"].mean().reset_index()
    print("\nAverage Time/Q (s) by Model Type:")
    print(avg_time_by_type)
    avg_time_by_type.to_csv(os.path.join(csv_dir, "avg_time_by_model_type.csv"), index=False)
    print(f"Saved avg_time_by_model_type.csv to {csv_dir}")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_time_by_type, x="Model Type", y="Avg Time/Q (s)")
    plt.title("Average Time/Q (s) by Model Type")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_time_by_model_type.png"))
    plt.close()
    print(f"Saved avg_time_by_model_type.png to {charts_dir}")

    # Boxplot of Accuracy by Model Type and Strategy
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x="Model Type", y="Accuracy", hue="Strategy")
    plt.title("Accuracy Distribution by Model Type and Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "accuracy_boxplot_model_type_strategy.png"))
    plt.close()
    print(f"Saved accuracy_boxplot_model_type_strategy.png to {charts_dir}")

def prompt_strategy_effectiveness(df, charts_dir, csv_dir):
    """Analyzes the effectiveness of different prompting strategies and saves plots and CSVs."""
    if df is None:
        return

    print("\n--- Prompt Strategy Effectiveness Analysis ---")

    # Average accuracy by strategy
    avg_accuracy_by_strategy = df.groupby("Strategy")["Accuracy"].mean().sort_values(ascending=False).reset_index()
    print("\nAverage Accuracy by Strategy:")
    print(avg_accuracy_by_strategy)
    avg_accuracy_by_strategy.to_csv(os.path.join(csv_dir, "avg_accuracy_by_strategy.csv"), index=False)
    print(f"Saved avg_accuracy_by_strategy.csv to {csv_dir}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_accuracy_by_strategy, x="Strategy", y="Accuracy")
    plt.title("Average Accuracy by Prompt Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_accuracy_by_strategy.png"))
    plt.close()
    print(f"Saved avg_accuracy_by_strategy.png to {charts_dir}")

    # Average cost by strategy
    avg_cost_by_strategy = df.groupby("Strategy")["Total Cost ($)"].mean().sort_values(ascending=True).reset_index()
    print("\nAverage Total Cost ($) by Strategy:")
    print(avg_cost_by_strategy)
    avg_cost_by_strategy.to_csv(os.path.join(csv_dir, "avg_cost_by_strategy.csv"), index=False)
    print(f"Saved avg_cost_by_strategy.csv to {csv_dir}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_cost_by_strategy, x="Strategy", y="Total Cost ($)")
    plt.title("Average Total Cost ($) by Prompt Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_cost_by_strategy.png"))
    plt.close()
    print(f"Saved avg_cost_by_strategy.png to {charts_dir}")

    # Average time by strategy
    avg_time_by_strategy = df.groupby("Strategy")["Avg Time/Q (s)"].mean().sort_values(ascending=True).reset_index()
    print("\nAverage Time/Q (s) by Strategy:")
    print(avg_time_by_strategy)
    avg_time_by_strategy.to_csv(os.path.join(csv_dir, "avg_time_by_strategy.csv"), index=False)
    print(f"Saved avg_time_by_strategy.csv to {csv_dir}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_time_by_strategy, x="Strategy", y="Avg Time/Q (s)")
    plt.title("Average Time/Q (s) by Prompt Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "avg_time_by_strategy.png"))
    plt.close()
    print(f"Saved avg_time_by_strategy.png to {charts_dir}")

def efficiency_tradeoffs_analysis(df, charts_dir, csv_dir):
    """Analyzes efficiency tradeoffs (Accuracy vs. Cost, Accuracy vs. Time) and saves plots and CSVs."""
    if df is None:
        return

    print("\n--- Efficiency Tradeoffs Analysis ---")

    # Accuracy vs. Cost
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x="Total Cost ($)", y="Accuracy", hue="Strategy", size="Avg Time/Q (s)", alpha=0.7)
    plt.title("Efficiency Tradeoff: Accuracy vs. Total Cost ($)")
    plt.xlabel("Total Cost ($)")
    plt.ylabel("Accuracy")
    plt.xscale('log') # Using log scale for cost if values vary widely
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "accuracy_vs_cost_tradeoff.png"))
    plt.close()
    print(f"Saved accuracy_vs_cost_tradeoff.png to {charts_dir}")

    # Accuracy vs. Avg Time/Q (s)
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x="Avg Time/Q (s)", y="Accuracy", hue="Strategy", size="Total Cost ($)", alpha=0.7)
    plt.title("Efficiency Tradeoff: Accuracy vs. Avg Time/Q (s)")
    plt.xlabel("Avg Time/Q (s)")
    plt.ylabel("Accuracy")
    plt.xscale('log')
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "accuracy_vs_time_tradeoff.png"))
    plt.close()
    print(f"Saved accuracy_vs_time_tradeoff.png to {charts_dir}")
    df['EfficiencyScore'] = df['Accuracy'] / (df['Total Cost ($)'] + df['Avg Time/Q (s)'] / 10) # Example metric
    
    high_efficiency_models = df[
        (df['Accuracy'] > df['Accuracy'].median()) &
        (df['Total Cost ($)'] < df['Total Cost ($)'].quantile(0.4)) &
        (df['Avg Time/Q (s)'] < df['Avg Time/Q (s)'].quantile(0.4))
    ].sort_values(by='EfficiencyScore', ascending=False)
    print("\nModels in High Efficiency Tier (example criteria):")
    print(high_efficiency_models[['Model', 'Strategy', 'Accuracy', 'Total Cost ($)', 'Avg Time/Q (s)', 'EfficiencyScore']].head())
    high_efficiency_models.to_csv(os.path.join(csv_dir, "high_efficiency_models.csv"), index=False)
    print(f"Saved high_efficiency_models.csv to {csv_dir}")

    high_performance_models = df[
        (df['Accuracy'] > df['Accuracy'].quantile(0.75)) &
        (df['Total Cost ($)'] < df['Total Cost ($)'].quantile(0.75)) &
        (df['Avg Time/Q (s)'] < df['Avg Time/Q (s)'].quantile(0.75))
    ].sort_values(by='Accuracy', ascending=False)
    print("\nModels in High Performance Tier (example criteria):")
    print(high_performance_models[['Model', 'Strategy', 'Accuracy', 'Total Cost ($)', 'Avg Time/Q (s)']].head())
    high_performance_models.to_csv(os.path.join(csv_dir, "high_performance_models.csv"), index=False)
    print(f"Saved high_performance_models.csv to {csv_dir}")

    premium_performance_models = df[df['Accuracy'] > df['Accuracy'].quantile(0.9)].sort_values(by='Accuracy', ascending=False)
    print("\nModels in Premium Performance Tier (highest accuracy):")
    print(premium_performance_models[['Model', 'Strategy', 'Accuracy', 'Total Cost ($)', 'Avg Time/Q (s)']].head())
    premium_performance_models.to_csv(os.path.join(csv_dir, "premium_performance_models.csv"), index=False)
    print(f"Saved premium_performance_models.csv to {csv_dir}")


if __name__ == "__main__":
    # Use the absolute path provided by the user
    csv_file_path = "/Users/pranamshetty/Work/Paper_1/CFA_MCQ_REPRODUCER/results/all_runs_summary_metrics.csv"
    summary_df = load_data(csv_file_path)

    if summary_df is not None:
        # Create a sub-directory for this specific analysis run if needed
        # For now, saving directly to output_dir
        
        overall_performance_analysis(summary_df, charts_output_dir, csv_output_dir)
        model_type_analysis(summary_df, charts_output_dir, csv_output_dir)
        prompt_strategy_effectiveness(summary_df, charts_output_dir, csv_output_dir)
        efficiency_tradeoffs_analysis(summary_df, charts_output_dir, csv_output_dir)

        print(f"\nAnalysis complete. Charts saved in {charts_output_dir}")
        print(f"Analysis complete. CSVs saved in {csv_output_dir}")
    else:
        print("Could not perform analysis due to data loading issues.") 