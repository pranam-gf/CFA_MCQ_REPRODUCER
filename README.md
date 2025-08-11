<p align="center">
  <a href="https://www.goodfin.com/" target="_blank">
    <img src="img/gf_logo.svg" alt="GoodFin Logo" width="200" style="background-color:white;"/>
  </a>
</p>

# CFA MCQ Question Reproducer

This project processes CFA MCQ questions using various Large Language Models (LLMs)
and evaluates their performance. It features an interactive UI with loading animations
and progress indicators to provide real-time feedback during processing.

## Benchmark Overview

This comprehensive LLM benchmark evaluates the performance of state-of-the-art language models on CFA multiple-choice questions, measuring both accuracy and efficiency metrics.

<table>
<thead>
<tr>
<th align="center">Benchmark Statistics</th>
<th align="center">Details</th>
<th align="center">Count</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>Models</strong></td>
<td>Claude-3.7-Sonnet, Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-Opus-4, Claude-Opus-4.1, Claude-Sonnet-4, Mistral-Large, Palmyra-fin, GPT-oss-20B, GPT-oss-120B, GPT-5, GTP-5-mini, GPT-5-nano, GPT-4o, o3-mini, o4-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, Grok-3, Grok-3-mini-beta (high/low effort), Grok-4, Gemini-2.5-Pro, Gemini-2.5-Flash, Deepseek-R1, Llama-4-Maverick, Llama-4-Scout, Llama-3.3-70B, Llama-3.1-8B-instant, Kimi-K2, Qwen3-32B, 
</td>
<td align="center"><strong>24+</strong></td>
</tr>
<tr>
<td align="center"><strong>Strategies</strong></td>
<td>Default, Chain of Thought (CoT), Self-Discover, Self-Consistency with different sample sizes (N=3, N=5)</td>
<td align="center"><strong>3-5</strong></td>
</tr>
<tr>
<td align="center"><strong>Metrics</strong></td>
<td>Accuracy, Precision, Recall, F1 score, Average Latency (ms), Total Input Tokens, Total Output Tokens, Total Tokens, Average Answer Length, Total Cost, Total Run Time</td>
<td align="center"><strong>11</strong></td>
</tr>
<tr>
<td align="center"><strong>Visualizations</strong></td>
<td>Model-Strategy Accuracy Comparison, Average Processing Time, Total Output Tokens, Combined Performance (Accuracy & Time), Accuracy/F1 Comparison (Default vs SC-CoT), Self-Consistency CoT Comparison (N=3 vs N=5), Accuracy/F1 vs Time Trade-off, Accuracy/F1 vs Cost Trade-off, Latency vs Cost Trade-off, Total Run Time Comparison, Default Strategy Metric Comparisons (Accuracy, F1, Latency), Confusion Matrices, **Pareto Frontier (Accuracy vs. Cost, Accuracy vs. Latency)**, **Cost-Latency-Accuracy Bubble Plot**</td>
<td align="center"><strong>18+</strong></td>
</tr>
</tbody>
</table>

This benchmark analyzes over 24 state-of-the-art LLMs across multiple reasoning strategies, measuring 11 performance metrics, and generating 18+ detailed visualization plots to evaluate and compare their performance on CFA multiple-choice questions.

## Project Structure

```
CFA_MCQ_REPRODUCER/
├── data/
│   ├── default/           # Raw JSON outputs (Default Strategy)
│   ├── cotn3/             # Raw JSON outputs (CoT N=3 Strategy)
│   ├── cotn5/             # Raw JSON outputs (CoT N=5 Strategy)
│   ├── sd/                # Raw JSON outputs (Self-Discover Strategy)
│   └── updated_data.json  # Input MCQ data (questions, correct answers)
├── results/
│   ├── analysis_charts/   # Output charts from the analysis script (src/evaluations/analyze_summary_metrics.py)
│   ├── IJCAI_RESULTS/     # CSV files of analysis tables (src/evaluations/analyze_summary_metrics.py)
│   ├── comparison_charts/ # Output charts from direct model runs (src/main.py)
│   ├── CSV_PLOTS/         # Comprehensive output charts from summary CSV (src/utils/generate_plots_only.py)
│   │   ├── comparative_performance/ # Standard comparison charts
│   │   ├── confusion_matrices/      # Confusion matrices for each run
│   │   ├── derived_metrics_analysis/ # Plots from advanced_analysis CSVs (efficiency, etc.)
│   │   ├── model_type_analysis/     # Plots comparing model categories (e.g., Reasoning vs. Non-Reasoning)
│   │   └── trade_off_analysis/      # Pareto, bubble, and other trade-off plots
│   └── *.json             # Raw JSON outputs for each model run
├── src/
│   ├── __init__.py
│   ├── config.py          # API keys, file paths, model configs, global settings
│   ├── llm_clients.py     # Functions for interacting with LLM APIs
│   ├── evaluations/       # Directory for performance evaluation modules
│   │   ├── classification.py # Classification metrics (accuracy, F1, etc.)
│   │   ├── resource_metrics.py # Resource usage (tokens, latency)
│   │   ├── cost_evaluation.py  # Cost estimation for various LLM providers
│   │   ├── analyze_summary_metrics.py # Script to perform deeper analysis on aggregated results and generate specific charts/tables
│   │   └── __init__.py
│   ├── plotting.py        # Chart generation functions
│   ├── prompts/           # Directory for LLM prompt templates
│   │   └── default.py     # Default prompt templates
│   │   └── cot.py         # Chain of Thought prompt templates
│   ├── utils/             # Utility functions
│   │   ├── ui_utils.py    # UI utilities (loading animations, colored output)
│   │   ├── prompt_utils.py # Prompt generation and parsing utilities
│   │   └── __init__.py    # (Likely, or add if not present)
│   └── main.py            # Main script to run the pipeline
├── .env                   # Local environment variables (API keys). Not version controlled.
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## Setup

1.  **Clone the repository (if applicable) or ensure you have this directory structure.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Create a `.env` file in the `CFA_MCQ_REPRODUCER` directory (i.e., at the same level as `src/` and `data/`).
    Add your API keys to this file. Example `.env` content:

    ```env
    OPENAI_API_KEY="your_openai_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    XAI_API_KEY="your_xai_api_key_for_grok"
    WRITER_API_KEY="your_writer_api_key"
    GROQ_API_KEY="your_groq_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    ```
    The script will load these variables. Alternatively, you can set them as system environment variables.

5.  **Prepare Input Data:**
    - Place your MCQ data file into the `CFA_MCQ_REPRODUCER/data/` directory. The script currently expects a file named `updated_data.json` (configurable in `src/config.py`).
    - **Required Data Structure:** The JSON file must contain a list of objects, where each object represents a question and should include keys like `question`, `options` (a dictionary like `{"A": "...", "B": "..."}`), `correct_answer` (the letter key, e.g., "A"), and potentially `explanation` or other metadata.

## Running the Pipeline

Navigate to the `CFA_MCQ_REPRODUCER` directory in your terminal.
Run the main script as a module:

```bash
python -m src.main
```

This will:
- Load data from `data/updated_data.json` with a loading animation.
- **Check for existing JSON results in `results/json/`**: If found, it will ask if you want to skip evaluations and proceed directly to generating plots from these existing results. If yes, it will update the summary CSV and generate all plots, then exit.
- **Prompt you to choose a run mode (if not skipping to plots):**
  - **Full Evaluation:** Runs all available models with the Default, Self-Consistency CoT (N=3), and Self-Consistency CoT (N=5) strategies automatically.
  - **Custom Run:** Allows you to interactively select specific models and a single strategy to run.
- Process each question with the selected configurations, showing real-time progress.
- Display colored success/error messages for each operation.
- Save detailed results for each model-strategy combination to a JSON file in the `results/` directory.
- Calculate and display evaluation metrics (e.g., accuracy, F1 score, estimated cost).
- **Save an aggregated summary of all metrics** to `results/all_runs_summary_metrics.csv`.
- **Prompt to generate plots (if evaluations were run):** If model evaluations were executed, it will ask if you want to generate plots. If yes, it will run `src.utils.generate_plots_only` to create all charts.
- Present a formatted summary table of results in the console.

### Interactive UI Features

The program now includes several UI enhancements:

1. **Loading Animations**: Displayed during long-running operations like:
   - Data loading
   - Model selection
   - LLM processing (with progress updates)
   - Evaluation
   - Chart generation
   - Results saving

2. **Progress Indicators**: Shows real-time progress during LLM processing:
   ```
   Processing with GPT-4 [15/50] ⠋
   ```

3. **Colored Output**:
   - ✓ Success messages in green
   - ✗ Error messages in red
   - ℹ Info messages in blue
   - ⚠ Warning messages in yellow

4. **Real-time Processing**: View the progress as the selected LLMs process the questions:

![Real-time LLM processing progress](img/exec_llm.png)

5. **Formatted Results Summary**: Displays a clear table of results at the end of processing:

![Formatted results summary table](img/llm_results.png)

6.  **Plot Generation & Post-Processing**: The main pipeline (`src/main.py`) offers options to automatically update summaries and generate plots. However, for more granular control or specific updates, you can run the following utility scripts manually:

    *   **a. Update Summary CSV (Optional, usually handled by `src/main.py`)**:
        *   **Command**: `python -m src.utils.update_summary_from_json`
        *   **Purpose**: This script reads all raw JSON output files from the `results/json/[strategy_folder]/` directories (e.g., `results/json/default/`, `results/json/cotn3/`, etc.) and creates/updates the master `results/all_runs_summary_metrics.csv` file. 
        *   **When to Run**: Typically, `src/main.py` handles this if you choose to generate plots or if it runs evaluations. Run this manually if you have added or modified raw JSON files in `results/json/` outside of a standard `src/main.py` execution and need to refresh the summary CSV before generating plots or advanced analysis.

    *   **b. Generate Advanced Analysis Data (Optional)**:
        *   **Command**: `python -m src.utils.generate_advanced_analysis`
        *   **Purpose**: This script takes the `results/all_runs_summary_metrics.csv` as input and calculates additional derived metrics, such as efficiency scores (e.g., Accuracy per Dollar, Tokens per Second) and aggregated statistics by model type or strategy. The outputs are saved as new CSV files in the `results/advanced_analysis/` directory.
        *   **When to Run**: Run this script after you have an up-to-date `all_runs_summary_metrics.csv` and before you run `generate_plots_only.py` if you want to include plots based on these derived/advanced metrics.

    *   **c. Generate All Plots**:
        *   **Command**: `python -m src.utils.generate_plots_only`
        *   **Purpose**: This is the primary script for generating all visual outputs for the project.
        *   **When to Run**: Execute this after `all_runs_summary_metrics.csv` is current, and (if desired) after `generate_advanced_analysis.py` has been run. `src/main.py` can also trigger this script if you opt to generate plots at the end of an evaluation run or when using existing JSONs.
        *   **Reads From**:
            *   `results/all_runs_summary_metrics.csv` (for most comparative and trade-off plots).
            *   Raw JSON files in `results/json/[strategy_folder]/` (specifically for generating confusion matrices).
            *   CSVs in `results/advanced_analysis/` (for plots visualizing derived and aggregated metrics).
        *   **Output**: All charts are saved in organized subdirectories within `results/CSV_PLOTS/` (e.g., `comparative_performance/`, `trade_off_analysis/`, `model_type_analysis/`, `derived_metrics_analysis/`, `confusion_matrices/`). Interactive HTML versions are saved alongside static PNG images.

    *   **d. Perform In-depth Analysis on Summary Metrics**:
        *   **Command**: `python -m src.evaluations.analyze_summary_metrics`
        *   **Purpose**: This script loads the `results/all_runs_summary_metrics.csv` file and performs a deeper analysis. It generates specific tables (e.g., top models by accuracy, cost, speed; performance by model type or strategy) and saves these tables as individual CSV files in the `results/IJCAI_RESULTS/` directory. It also generates corresponding plots for these analyses, saved in `results/analysis_charts/`.
        *   **When to Run**: Run this script after `all_runs_summary_metrics.csv` is up-to-date to get detailed tabular and visual summaries of the overall benchmark performance.
        *   **Reads From**: `results/all_runs_summary_metrics.csv`.
        *   **Output**: 
            *   CSV files containing analysis tables in `results/IJCAI_RESULTS/`.
            *   Plot images (PNG) in `results/analysis_charts/`.

    Key outputs from the plotting script include:
    *   **Aggregated Metrics Summary (`results/all_runs_summary_metrics.csv`)**: A CSV file containing key metrics for every model-strategy combination executed.
    *   **Confusion Matrices (`results/CSV_PLOTS/confusion_matrices/`)**: Generated from raw JSON outputs in `results/json/`.
    *   **Various Performance Charts (in `results/CSV_PLOTS/` subdirectories)**: Bar charts, scatter plots, Pareto frontiers, bubble plots, etc., visualizing accuracy, latency, cost, and efficiency metrics.

## Configuration

-   **Supported LLM Providers and Models:** This project is designed to work with a variety of LLM providers. Support is integrated for:
    - OpenAI (e.g., GPT-4o, GPT-4.1 series,o3-mini, o4-mini, o3-pro). For `o3-pro` models (e.g., `o3-pro-2025-06-10`), the project utilizes the new `responses.create` endpoint, supporting the `reasoning` parameter (e.g., `{\"effort\": \"high\"}`) and distinct token types (input, completion, reasoning) for cost and performance analysis.
    - Google Gemini (e.g., Gemini 2.5 Pro, Gemini 2.5 Flash with `thinking_budget`). Requires `google-genai>=1.10.0` (version `1.15.0` confirmed working) for `ThinkingConfig` support.
    - Anthropic (e.g., Claude 3.7 Sonnet, Claude 3.5 Sonnet & Haiku)
    - Groq (e.g., Llama 4 Maverick/Scout, Llama 3.3 70B, Llama 3.1 8B, with `reasoning_effort` for Grok models)
    - Writer.com (e.g., Palmyra-fin)
    - xAI (e.g., Grok-3)
    Specific model IDs, versions, and their parameters are defined within the Python files in the `src/configs/` directory (e.g., `default_config.py`, `cot_config.py`).

-   **Model Selection & Parameters:** Edit the configuration files within the `src/configs/` directory (e.g., `default_config.py`). These files list the available models (`config_id`), their corresponding API identifiers (`model_id`), types (`type` which maps to the correct API client in `llm_clients.py`), and strategy-specific parameters. You can add, remove, or modify entries here to control which models are available for selection and how they behave.
    - For example, a model configuration entry might look like this:
      ```python
      {
          "config_id": "gemini-2.5-flash",
          "type": "gemini",
          "model_id": "gemini-2.5-flash-preview-04-17",
          "parameters": {
              "temperature": 0.7,
              "top_p": 0.95,
              "top_k": 64,
              "thinking_budget": 24576
          }
      },
      ```
    - Note that some models support unique parameters that significantly affect their behavior and cost, such as `thinking_budget` for certain Gemini models (e.g., Gemini 2.5 Flash) or `reasoning_effort` for Groq models (e.g., `grok-3-mini-beta`). Ensure these are configured appropriately in the `parameters` section of the model's configuration.
    - Groq models (`grok-3-mini-beta` and `grok-3-mini-fast-beta`) now have configurations for `high` and `low` `reasoning_effort` respectively.
-   **API Keys & File Paths:** Global settings like API key environment variable names and default data/results paths can be adjusted in `src/config.py` if needed, though using the `.env` file is recommended for keys.
-   **Prompt Templates:** Modify or add prompt templates in the `src/prompts/` directory (e.g., `default.py`, `cot.py`) to change how questions are presented to the LLMs for different strategies.

## Directory Structure

```
CFA_MCQ_REPRODUCER/
├── data/                     # Input data files (e.g., questions, existing results for parsing)
│   ├── cotn3/
│   ├── cotn5/
│   ├── default/
│   └── sd/
├── results/
│   ├── analysis_charts/      # Charts from analyze_summary_metrics.py
│   ├── IJCAI_RESULTS/        # CSV tables from analyze_summary_metrics.py
│   ├── CSV_PLOTS/            # CENTRAL DIRECTORY FOR ALL GENERATED PLOTS, CATEGORIZED
│   │   ├── comparative_performance/ # General model/strategy comparison plots
│   │   ├── confusion_matrices/      # Confusion matrices for each run
│   │   ├── derived_metrics_analysis/ # Plots from advanced_analysis CSVs (efficiency, etc.)
│   │   ├── model_type_analysis/     # Plots comparing reasoning vs. non-reasoning models
│   │   └── trade_off_analysis/      # Pareto frontiers, bubble charts for cost/latency/accuracy
│   ├── advanced_analysis/    # CSV files with derived metrics and aggregations
│   ├── outputs/                # Raw outputs from each model run (JSONs, logs) - USED FOR CONFUSION MATRIX PLOTS
│   └── all_runs_summary_metrics.csv # Aggregated metrics from all runs
├── src/
│   ├── __init__.py
│   ├── config.py          # API keys, file paths, model configs, global settings
│   ├── llm_clients.py     # Functions for interacting with LLM APIs
│   ├── evaluations/       # Directory for performance evaluation modules
│   │   ├── classification.py # Classification metrics (accuracy, F1, etc.)
│   │   ├── resource_metrics.py # Resource usage (tokens, latency)
│   │   ├── cost_evaluation.py  # Cost estimation for various LLM providers
│   │   ├── analyze_summary_metrics.py # Script to perform deeper analysis on aggregated results and generate specific charts/tables
│   │   └── __init__.py
│   ├── plotting.py        # Chart generation functions
│   ├── prompts/           # Directory for LLM prompt templates
│   │   └── default.py     # Default prompt templates
│   │   └── cot.py         # Chain of Thought prompt templates
│   ├── utils/             # Utility functions
│   │   ├── ui_utils.py    # UI utilities (loading animations, colored output)
│   │   ├── prompt_utils.py # Prompt generation and parsing utilities
│   │   └── __init__.py    # (Likely, or add if not present)
│   └── main.py            # Main script to run the pipeline
├── .env                   # Local environment variables (API keys). Not version controlled.
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

### 3. Generate All Plots

After running evaluations and, if necessary, updating the summary CSV:

```bash
python -m src.utils.generate_plots_only
```

This script will:
- Read the `results/all_runs_summary_metrics.csv` for most plots.
- Read individual run JSONs from `results/outputs/` to generate confusion matrices.
- Save all generated plots into categorized subdirectories under `results/CSV_PLOTS/`.

### (Optional) Update Summary CSV from JSON Outputs
