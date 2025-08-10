"""
Model pricing constants for various LLM providers.
All pricing is normalized to cost per 1 million tokens.

This file centralizes all pricing information used for cost calculations in the evaluation process,
making it easier to update prices and ensuring consistency across the codebase.
"""
import logging
from typing import Optional, Dict
logger = logging.getLogger(__name__)

OPENAI_PRICING = {
    "gpt-4o": {
        "prompt_tokens_cost_per_million": 2.50,
        "completion_tokens_cost_per_million": 10.00,
    },
    "o3": {
        "prompt_tokens_cost_per_million": 1.10,
        "completion_tokens_cost_per_million": 4.40,
    },
    "o3-mini-2025-01-31": {
        "prompt_tokens_cost_per_million": 1.10,
        "completion_tokens_cost_per_million": 4.40,
    },
    "o4-mini": {
        "prompt_tokens_cost_per_million": 1.10,
        "completion_tokens_cost_per_million": 4.40,
    },
    "o4-mini-2025-04-16": {
        "prompt_tokens_cost_per_million": 1.10,
        "completion_tokens_cost_per_million": 4.40,
    },
    "gpt-4.1-2025-04-14": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 8.00,
    },
    "gpt-4.1-mini-2025-04-14": {
        "prompt_tokens_cost_per_million": 0.40,
        "completion_tokens_cost_per_million": 1.60,
    },
    "gpt-4.1-nano-2025-04-14": {
        "prompt_tokens_cost_per_million": 0.10,
        "completion_tokens_cost_per_million": 0.40,
    },
    "o3-pro-2025-06-10": {
        "prompt_tokens_cost_per_million": 20.00,
        "completion_tokens_cost_per_million": 80.00,
        "reasoning_tokens_cost_per_million": 2.00,
    },
    "gpt-5-2025-08-07": {
        "prompt_tokens_cost_per_million": 1.25,
        "completion_tokens_cost_per_million": 10.00,
    },
    "gpt-5-mini-2025-08-07": {
        "prompt_tokens_cost_per_million": 0.25,
        "completion_tokens_cost_per_million": 2.00,
    },
    "gpt-5-nano-2025-08-07": {
        "prompt_tokens_cost_per_million": 0.05,
        "completion_tokens_cost_per_million": 0.40,
    }
}

ANTHROPIC_PRICING = {
    "claude-3-7-sonnet-20250219": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "claude-sonnet-4-20250514": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "claude-3-5-sonnet-20241022": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "claude-3-5-haiku-20241022": {
        "prompt_tokens_cost_per_million": 0.80,
        "completion_tokens_cost_per_million": 4.00,
    },
    "claude-3-opus-20240229": {
        "prompt_tokens_cost_per_million": 15.00,
        "completion_tokens_cost_per_million": 75.00,
    },
    "claude-opus-4-20250514": {
        "prompt_tokens_cost_per_million": 15.00,
        "completion_tokens_cost_per_million": 75.00,
    },
    "claude-3-haiku-20240307": {
        "prompt_tokens_cost_per_million": 0.25,
        "completion_tokens_cost_per_million": 1.25,
    },
    "claude-opus-4-1-20250805": {
        "prompt_tokens_cost_per_million": 15.00,
        "completion_tokens_cost_per_million": 75.00,
    }
}

GEMINI_PRICING = {
    "gemini-2.5-pro-preview-05-06": {
        "prompt_tokens_cost_per_million": 1.25,
        "completion_tokens_cost_per_million": 10.00,
    },
    "gemini-2.5-flash-preview-04-17": {
        "prompt_tokens_cost_per_million": 0.15,
        "completion_tokens_cost_per_million": 0.60,
        "completion_tokens_cost_per_million_thinking": 3.50
    }
}

GROQ_PRICING = {
    "llama-3.1-8b-instant": {
        "prompt_tokens_cost_per_million": 0.05,
        "completion_tokens_cost_per_million": 0.08,
    },
    "llama-3.3-70b-versatile": {
        "prompt_tokens_cost_per_million": 0.59,
        "completion_tokens_cost_per_million": 0.79,
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "prompt_tokens_cost_per_million": 0.20,
        "completion_tokens_cost_per_million": 0.60,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "prompt_tokens_cost_per_million": 0.11,
        "completion_tokens_cost_per_million": 0.34,
    },
    "deepseek-r1-distill-llama-70b": {
        "prompt_tokens_cost_per_million": 0.75,
        "completion_tokens_cost_per_million": 0.99,
    },
        "openai/gpt-oss-20b": {
        "prompt_tokens_cost_per_million": 0.10,
        "completion_tokens_cost_per_million": 0.50,
    },
        "openai/gpt-oss-120b": {
        "prompt_tokens_cost_per_million": 0.15,
        "completion_tokens_cost_per_million": 0.75,
    },
    "moonshotai/kimi-k2-instruct": {
        "prompt_tokens_cost_per_million": 1.00,
        "completion_tokens_cost_per_million": 3.00,
    },
    "qwen/qwen3-32b": {
        "prompt_tokens_cost_per_million": 0.29,
        "completion_tokens_cost_per_million": 0.59,
    },
}

MISTRAL_PRICING = {
    "mistral-large-latest": {
        "prompt_tokens_cost_per_million": 2.00,
        "completion_tokens_cost_per_million": 6.00,
    },
    "codestral-latest": {
        "prompt_tokens_cost_per_million": 0.3,
        "completion_tokens_cost_per_million": 0.9,
    }
}

WRITER_PRICING = {
    "palmyra-fin": {
        "prompt_tokens_cost_per_million": 5.00,
        "completion_tokens_cost_per_million": 12.00,
    }
}

GROK_PRICING = {
    "grok-3-latest": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
        "context_window": 131072
    },
    "grok-3-mini-beta": {
        "prompt_tokens_cost_per_million": 0.30,
        "completion_tokens_cost_per_million": 0.50,
        "context_window": 131072
    },
    "grok-3-mini-fast-beta": {
        "prompt_tokens_cost_per_million": 0.60,
        "completion_tokens_cost_per_million": 4.00,
        "context_window": 131072
    },
    "grok-4-0709": {
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
        "context_window": 256000
    }
}       

def get_pricing(model_type: str, model_id: str):
    """
    Unified function to get pricing for any model type.
    
    Args:
        model_type: The type of model (e.g., "openai", "anthropic")
        model_id: The specific model ID
        
    Returns:
        Dictionary containing pricing information or None if not found
    """
    pricing_functions = {
        "openai": get_openai_pricing,
        "anthropic": get_anthropic_pricing,
        "gemini": get_gemini_pricing,
        "groq": get_groq_pricing,
        "mistral_official": get_mistral_pricing,
        "writer": get_writer_pricing,
        "xai": get_grok_pricing
    }
    
    lookup_function = pricing_functions.get(model_type)
    if not lookup_function:
        logger.warning(f"No pricing lookup function found for model type: '{model_type}'")
        return None  
    return lookup_function(model_id)

def get_openai_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Retrieves the pricing for a given OpenAI model ID.
    Handles variations like preview versions by stripping common suffixes.
    """
    pricing = OPENAI_PRICING.get(model_id)

    if pricing:
        input_cost_per_million = pricing.get("prompt_tokens_cost_per_million")
        output_cost_per_million = pricing.get("completion_tokens_cost_per_million")
        reasoning_cost_per_million = pricing.get("reasoning_tokens_cost_per_million")

        if input_cost_per_million is not None and output_cost_per_million is not None:
            result = {
                "input": input_cost_per_million / 1_000_000,
                "output": output_cost_per_million / 1_000_000
            }
            if reasoning_cost_per_million is not None:
                result["reasoning"] = reasoning_cost_per_million / 1_000_000
            return result
        else:
            logger.warning(f"Found pricing entry for OpenAI model ID: '{model_id}', but keys for input/output costs are missing or inconsistent. Cost will be $0.")
            return None

    if model_id == "o4-mini-2025-04-16" and "o4-mini" in OPENAI_PRICING:
        pricing_fallback = OPENAI_PRICING["o4-mini"]
        input_cost = pricing_fallback.get("prompt_tokens_cost_per_million")
        output_cost = pricing_fallback.get("completion_tokens_cost_per_million")
        if input_cost is not None and output_cost is not None:
            return {"input": input_cost / 1_000_000, "output": output_cost / 1_000_000}

    if model_id == "o3-mini-2025-01-31" and "o3-mini" in OPENAI_PRICING:
        pricing_fallback = OPENAI_PRICING["o3-mini"]
        input_cost = pricing_fallback.get("prompt_tokens_cost_per_million")
        output_cost = pricing_fallback.get("completion_tokens_cost_per_million")
        if input_cost is not None and output_cost is not None:
            return {"input": input_cost / 1_000_000, "output": output_cost / 1_000_000}

    logger.warning(f"Pricing not found for OpenAI model ID: '{model_id}'. Cost will be $0.")
    return None

def get_anthropic_pricing(model_id: str):
    """Returns the pricing for Anthropic models."""
    pricing = ANTHROPIC_PRICING.get(model_id)
    
    if not pricing:
        if "claude-3.7-sonnet" in model_id:
            pricing = ANTHROPIC_PRICING.get("claude-3-7-sonnet-20250219")
        elif "claude-3.5-sonnet" in model_id: 
            pricing = ANTHROPIC_PRICING.get("claude-3-5-sonnet-20241022")
        elif "claude-3.5-haiku" in model_id:
            pricing = ANTHROPIC_PRICING.get("claude-3.5-haiku-20241022")
        elif "claude-3-opus" in model_id: 
            pricing = ANTHROPIC_PRICING.get("claude-3-opus-20240229")
        elif "claude-3-haiku" in model_id: 
            pricing = ANTHROPIC_PRICING.get("claude-3-haiku-20240307")

    if not pricing:
        logger.warning(f"Pricing not found for Anthropic model ID: '{model_id}'")
    return pricing

def get_gemini_pricing(model_id: str):
    """Returns the pricing for Gemini models."""
    return GEMINI_PRICING.get(model_id)

def get_groq_pricing(model_id: str):
    """Returns pricing information for Groq models."""
    pricing = GROQ_PRICING.get(model_id)
    return pricing

def get_mistral_pricing(model_id: str):
    """Returns the pricing for Mistral models."""
    pricing = MISTRAL_PRICING.get(model_id)
    
    if not pricing:
        logger.warning(f"Pricing not found for Mistral model ID: '{model_id}'")
    return pricing

def get_writer_pricing(model_id: str):
    """Returns the pricing for Writer (Palmyra) models."""
    pricing = WRITER_PRICING.get(model_id)
    if not pricing:
        logger.warning(f"Pricing not found for Writer model ID: '{model_id}'")
    return pricing

def get_grok_pricing(model_id: str):
    """Returns the pricing for Grok/Xai models."""
    pricing = GROK_PRICING.get(model_id)
    if not pricing:
        if "grok-3-mini-fast" in model_id:
            pricing = GROK_PRICING.get("grok-3-mini-fast-beta")
        elif "grok-3-mini" in model_id:
            pricing = GROK_PRICING.get("grok-3-mini-beta")
        elif "grok-3-fast" in model_id:
            pricing = GROK_PRICING.get("grok-3-latest")
            
    if not pricing:
        logger.warning(f"Pricing not found for Grok/Xai model ID: '{model_id}'")
    return pricing 