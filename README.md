# CFA MCQ Question Reproducer

This project processes CFA MCQ questions using various Large Language Models (LLMs)
and evaluates their performance. It features an interactive UI with loading animations
and progress indicators to provide real-time feedback during processing.

## Project Structure

```
CFA_MCQ_REPRODUCER/
├── data/
│   └── updated_data.json  # Input MCQ data (questions, correct answers)
├── results/
│   ├── comparison_charts/ # Output charts comparing model performance
│   └── *.json             # Raw JSON outputs for each model run
├── src/
│   ├── __init__.py
│   ├── config.py          # API keys, file paths, model configs, global settings
│   ├── llm_clients.py     # Functions for interacting with LLM APIs
│   ├── evaluations/       # Directory for performance evaluation modules
│   │   ├── classification.py # Classification metrics (accuracy, F1, etc.)
│   │   ├── resource_metrics.py # Resource usage (tokens, latency)
│   │   ├── cost_evaluation.py  # Cost estimation for various LLM providers
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
    AWS_ACCESS_KEY_ID="your_aws_access_key_id"
    AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
    AWS_REGION="your_aws_region" # e.g., us-east-1
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
- **Prompt you to choose a run mode:**
  - **Full Evaluation:** Runs all available models with the Default, Self-Consistency CoT (N=3), and Self-Consistency CoT (N=5) strategies automatically.
  - **Custom Run:** Allows you to interactively select specific models and a single strategy to run.
- Process each question with the selected configurations, showing real-time progress.
- Display colored success/error messages for each operation.
- Save detailed results for each model-strategy combination to a JSON file in the `results/` directory.
- Calculate and display evaluation metrics (e.g., accuracy, F1 score, estimated cost).
- **Save an aggregated summary of all metrics** to `results/all_runs_summary_metrics.csv`.
- Generate a comprehensive suite of comparison charts in `results/comparison_charts/`.
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

6.  **Performance Visualization**: Generates and saves a suite of comparison charts in `results/comparison_charts/` for comprehensive analysis of model and strategy performance. Interactive HTML versions are also saved alongside static PNG images. Key outputs include:
    *   **Aggregated Metrics Summary (`results/all_runs_summary_metrics.csv`)**: A CSV file containing key metrics for every model-strategy combination executed. Columns include: `run_id`, `model_id_full`, `base_model_id`, `strategy_name`, `strategy_type`, `strategy_param`, `display_name`, `accuracy`, `f1_score`, `avg_time_per_question`, `total_run_time`, `total_output_tokens`, `total_cost`.
    *   **Model-Strategy Accuracy Comparison (`model_strategy_comparison_accuracy.png`/`.html`)**: Bar chart visualizing the accuracy scores for each model-strategy combination. Higher bars denote better performance.
    *   **Average Processing Time per Question (`model_strategy_comparison_response_time.png`/`.html`)**: Bar chart showing the average time (in seconds) each model-strategy took per question. Lower bars indicate faster processing.
    *   **Total Output Tokens Generated (`model_strategy_comparison_output_tokens.png`/`.html`)**: Bar chart illustrating the total output tokens generated by each configuration. Useful for assessing verbosity and potential costs.
    *   **Combined Performance: Accuracy & Avg. Question Time (`model_strategy_combined_metrics.png`/`.html`)**: Dual-axis chart presenting accuracy (bars) and average processing time (line) together for evaluating speed vs. accuracy trade-offs.
    *   **Accuracy/F1 Comparison: Default vs. SC-CoT (N=3) (`comparison_<metric>_by_strategy.png`/`.html`)**: Grouped bar chart directly comparing the performance (Accuracy and F1 Score) of the Default strategy against the Self-Consistency CoT (N=3) strategy for each base model.
    *   **Self-Consistency CoT Comparison: N=3 vs. N=5 (`comparison_sc_<metric>_n3_vs_n5.png`/`.html`)**: Grouped bar chart comparing the performance (Accuracy and F1 Score) between N=3 and N=5 samples for the Self-Consistency CoT strategy across models.
    *   **Accuracy/F1 vs. Time Trade-off (`tradeoff_<metric>_vs_time.png`/`.html`)**: Scatter plot visualizing the relationship between performance (Accuracy or F1 Score) on the Y-axis and Average Time per Question on the X-axis. Points are colored by model and shaped by strategy type, helping identify efficient configurations.
    *   **Total Run Time Comparison by Strategy (`comparison_total_time_by_strategy.png`/`.html`)**: Grouped bar chart showing the *total* time taken to process all questions for each model, grouped by strategy. This provides a view of the overall execution duration for each configuration.
    *   **Default Strategy - Accuracy Comparison (`default_strategy_accuracy_comparison.png`/`.html`)**: Bar chart comparing the accuracy of all tested models when using only the 'default' prompt strategy. Allows for clear model-to-model comparison under the baseline strategy.
    *   **Default Strategy - F1 Score Comparison (`default_strategy_f1_score_comparison.png`/`.html`)**: Bar chart comparing the F1-score of all tested models when using only the 'default' prompt strategy.
    *   **Default Strategy - Latency Comparison (`default_strategy_average_latency_ms_comparison.png`/`.html`)**: Bar chart comparing the average processing time (latency in milliseconds) of all tested models when using only the 'default' prompt strategy.

## Configuration

-   **Model Selection & Parameters:** Edit the configuration files within the `src/configs/` directory (e.g., `src/configs/default_config.py`). These files list the available models (`config_id`), their corresponding API identifiers (`model_id`), types (`type`), and strategy-specific parameters (like `temperature`, `max_tokens`). You can add, remove, or modify entries here to control which models are available for selection and how they behave.
-   **API Keys & File Paths:** Global settings like API key environment variable names and default data/results paths can be adjusted in `src/config.py` if needed, though using the `.env` file is recommended for keys.
-   **Prompt Templates:** Modify or add prompt templates in the `src/prompts/` directory (e.g., `default.py`, `cot.py`) to change how questions are presented to the LLMs for different strategies.