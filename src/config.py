"""
Configuration settings for the CFA MCQ Reproducer project.

This file centralizes API keys, file paths, model configurations,
and global operational flags.
"""
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    logging.info(f".env file loaded from {ENV_PATH}")
else:
    logging.info(f".env file not found at {ENV_PATH}. Relying on system environment variables.")

def get_credential(key: str, default: str | None = None) -> str | None:
    """Safely retrieves a credential."""
    val = os.environ.get(key)
    if val:
        logging.debug(f"Loaded {key} from environment variable.")
        return val
    if default:
        logging.warning(f"{key} not found, using default value.")
        return default
    logging.warning(f"{key} not found and no default provided.")
    return None

MISTRAL_API_KEY = get_credential("MISTRAL_API_KEY") 
OPENAI_API_KEY = get_credential("OPENAI_API_KEY")
GEMINI_API_KEY = get_credential("GEMINI_API_KEY")
XAI_API_KEY = get_credential("XAI_API_KEY") 
WRITER_API_KEY = get_credential("WRITER_API_KEY")
AWS_ACCESS_KEY_ID = get_credential("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_credential("AWS_SECRET_ACCESS_KEY")
AWS_REGION = get_credential("AWS_REGION", "us-east-1")
GROQ_API_KEY = get_credential("GROQ_API_KEY")




PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "comparison_charts"


DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

FILLED_JSON_PATH = DATA_DIR / "updated_data.json" 
# ALL_MODEL_CONFIGS = [ ... ] # This list will be removed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), 
        
        
    ]
)
logger = logging.getLogger(__name__)

logger.info("Configuration loaded.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found. OpenAI models will fail.")
# Update these checks if they rely on ALL_MODEL_CONFIGS being defined here
# For now, assuming these checks might be better placed in main.py when configs are loaded.
# if (any(m.get('type') == 'bedrock' for m in ALL_MODEL_CONFIGS) or \
#     any(m.get('type') == 'sagemaker' for m in ALL_MODEL_CONFIGS)) and \
#    (not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY):
#     logger.warning("AWS credentials not found. Bedrock/SageMaker models will fail.")
# if any(m.get('type') == 'gemini' for m in ALL_MODEL_CONFIGS) and not GEMINI_API_KEY:
#     logger.warning("GEMINI_API_KEY not found. Gemini models will fail.")
# if any(m.get('type') == 'xai' for m in ALL_MODEL_CONFIGS) and not XAI_API_KEY:
#     logger.warning("XAI_API_KEY not found. Grok models will fail.")
# if any(m.get('type') == 'writer' for m in ALL_MODEL_CONFIGS) and not WRITER_API_KEY: 
#     logger.warning("WRITER_API_KEY not found. Writer models will fail.")
# if any(m.get('type') == 'groq' for m in ALL_MODEL_CONFIGS) and not GROQ_API_KEY:
#     logger.warning("GROQ_API_KEY not found. Groq models will fail.") 