# CFA MCQ Question Reproducer

This project processes CFA MCQ questions using various Large Language Models (LLMs)
and evaluates their performance.

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
    ```
    The script will load these variables. Alternatively, you can set them as system environment variables.

5.  **Prepare Input Data:**
    Place your `updated_data.json` file (containing the MCQ questions, answers, and explanations)
    into the `CFA_MCQ_REPRODUCER/data/` directory.

## Running the Pipeline

Navigate to the `CFA_MCQ_REPRODUCER` directory in your terminal.
Run the main script as a module:

```bash
pip install python-dotenv openai Pillow reportlab scikit-learn numpy boto3 google-generativeai matplotlib seaborn kaleido writerai
python -m src.main
```

This will:
- Load data from `data/updated_data.json`.
- Iterate through the models defined in `src/config.py`.
- Query each LLM for answers.
- Save detailed results for each model to a JSON file in the `results/` directory.
- Calculate evaluation metrics (e.g., accuracy).
- Generate comparison charts in `results/comparison_charts/`.

## Configuration

-   **Model Selection:** Edit `ALL_MODEL_CONFIGS` in `src/config.py` to add, remove, or modify LLM configurations.
-   **Paths:** File and directory paths are also defined in `src/config.py` and are relative to the project root.

## TODO / Next Steps for Porting Code

-   Move `ALL_MODEL_CONFIGS` list from `llm_cfa_reproduce.py` to `CFA_MCQ_REPRODUCER/src/config.py`.
-   Port the LLM interaction logic (`generate_prompt`, `get_llm_response`, `process_questions_with_llm`) into `CFA_MCQ_REPRODUCER/src/llm_clients.py`.
-   Port evaluation functions (`compute_cosine_similarity`, `evaluate_similarity`, `evaluate_classification`) to `CFA_MCQ_REPRODUCER/src/evaluation.py`.
-   Port plotting logic (the entire `# 4. Model Comparison Charts` section) to `CFA_MCQ_REPRODUCER/src/plotting.py`.
-   Update `CFA_MCQ_REPRODUCER/src/main.py` to call these ported functions correctly, passing necessary data and configurations.
-   Ensure all imports are correctly resolved within the new module structure (using relative imports like `from . import config` or absolute imports if the package structure allows, like `from src.config import ...` if running from project root and `src` is added to `PYTHONPATH` or running as module `python -m src.main`). 