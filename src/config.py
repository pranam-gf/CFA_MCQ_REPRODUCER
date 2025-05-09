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
ALL_MODEL_CONFIGS = [
    
     {
    "config_id": "claude-3.7-sonnet",
    "type": "bedrock",
    "model_id": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "use_inference_profile": True,
    "inference_profile_arn": "arn:aws:bedrock:us-east-1:000518066116:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens": 512,
        "anthropic_version": "bedrock-2023-05-31"
    }
    },
    {
        "config_id": "claude-3.5-sonnet",
        "type": "bedrock",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 200,
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    {
        "config_id": "claude-3.5-haiku",
        "type": "bedrock",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 512,
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    {
        "config_id": "mistral-large",
        "type": "bedrock",
        "model_id": "mistral.mistral-large-2402-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 200
        }
    },
    {
        "config_id": "llama3-70b",
        "type": "bedrock",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_gen_len": 200
        }
    },
    {
        "config_id": "llama3.1-8b",
        "type": "bedrock",
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_gen_len": 200
        }
    },
    
    {
        "config_id": "gpt-4o",
        "type": "openai",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
    },
    
    {
        "config_id": "gpt-4.1",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-mini",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-nano",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }
    },
    
    {
        "config_id": "grok-3",
        "type": "xai",
        "model_id": "grok-3-latest",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200
        }
    },
    
    {
    "config_id": "gemini-2.5-pro",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-exp-03-25",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 2048
    }
    },
    {
    "config_id": "gemini-2.0-flash",
    "type": "gemini",
    "model_id": "gemini-2.0-flash",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 768
    }
    },
        {
        "config_id": "palmyra-fin",
        "type": "writer",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 200
        }
    },
    {
        "config_id": "deepseek-via-groq",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.9
        }
    }
]


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
if (any(m.get('type') == 'bedrock' for m in ALL_MODEL_CONFIGS) or \
    any(m.get('type') == 'sagemaker' for m in ALL_MODEL_CONFIGS)) and \
   (not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY):
    logger.warning("AWS credentials not found. Bedrock/SageMaker models will fail.")
if any(m.get('type') == 'gemini' for m in ALL_MODEL_CONFIGS) and not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found. Gemini models will fail.")
if any(m.get('type') == 'xai' for m in ALL_MODEL_CONFIGS) and not XAI_API_KEY:
    logger.warning("XAI_API_KEY not found. Grok models will fail.")
if any(m.get('type') == 'writer' for m in ALL_MODEL_CONFIGS) and not WRITER_API_KEY: 
    logger.warning("WRITER_API_KEY not found. Writer models will fail.")
if any(m.get('type') == 'groq' for m in ALL_MODEL_CONFIGS) and not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found. Groq models will fail.") 