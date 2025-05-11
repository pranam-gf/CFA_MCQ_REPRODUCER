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
│   ├── evaluation.py      # Performance evaluation metrics and functions
│   ├── plotting.py        # Chart generation functions
│   ├── prompts/           # Directory for LLM prompt templates
│   │   └── default.py     # Default prompt templates
│   │   └── cot.py         # Chain of Thought prompt templates
│   ├── ui_utils.py        # UI utilities for loading animations and colored output
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
    ```
    The script will load these variables. Alternatively, you can set them as system environment variables.

5.  **Prepare Input Data:**
    Place your `updated_data.json` file (containing the MCQ questions, answers, and explanations)
    into the `CFA_MCQ_REPRODUCER/data/` directory.

## Running the Pipeline

Navigate to the `CFA_MCQ_REPRODUCER` directory in your terminal.
Run the main script as a module:

```bash
python -m src.main
```

This will:
- Load data from `data/updated_data.json` with a loading animation.
- Present an interactive model selection interface.
- Process each question with the selected LLMs, showing real-time progress.
- Display colored success/error messages for each operation.
- Save detailed results for each model to a JSON file in the `results/` directory.
- Calculate and display evaluation metrics (e.g., accuracy).
- Generate comparison charts in `results/comparison_charts/`.
- Present a formatted summary table of results.

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

6.  **Performance Visualization**: Generates and saves a suite of comparison charts in `results/comparison_charts/` for comprehensive analysis of model and strategy performance. Interactive HTML versions are also saved alongside static PNG images. Key charts include:
    *   **Model-Strategy Accuracy Comparison (`model_strategy_comparison_accuracy.png`/`.html`)**: A bar chart visualizing the accuracy scores for each model-strategy combination. Higher bars denote better performance in answering questions correctly, aiding in the identification of top-performing setups.
    *   **Average Processing Time per Question (`model_strategy_comparison_response_time.png`/`.html`)**: This bar chart shows the average time (in seconds) each model-strategy took per question. Lower bars indicate faster processing, which is important for efficiency.
    *   **Total Output Tokens Generated (`model_strategy_comparison_output_tokens.png`/`.html`)**: This bar chart illustrates the total output tokens generated by each configuration. This helps in assessing model verbosity and potential operational costs related to token usage.
    *   **Combined Performance: Accuracy & Avg. Question Time (`model_strategy_combined_metrics.png`/`.html`)**: A dual-axis chart presenting accuracy (bars) and average processing time (line) together. This allows for quick evaluation of the balance between speed and accuracy for different model-strategy pairings.

## Configuration

-   **Model Selection:** Edit `