import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)







OPENAI_PRICING_PER_TOKEN = {
    
    "o1": {"input": 15.00 / 1_000_000, "output": 60.00 / 1_000_000},         
    "o1-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},      
    "o3": {"input": 10.00 / 1_000_000, "output": 40.00 / 1_000_000},         
    "o3-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},      
    "o4-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},      

    
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},      

    
    "gpt-4.1": {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},      
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000}, 
    "gpt-4.1-nano": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000}, 
}

def get_openai_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """Looks up the pricing for a given OpenAI model ID."""
    pricing = OPENAI_PRICING_PER_TOKEN.get(model_id)
    if not pricing:
        
        
        parts = model_id.split('-')
        base_model_candidate = parts[0] + '-' + parts[1] if len(parts) > 1 else model_id 
        if len(parts) > 2 and (parts[1] == '3.5' or parts[1] == '4'): 
             base_model_candidate += '-' + parts[2]

        if base_model_candidate != model_id:
             pricing = OPENAI_PRICING_PER_TOKEN.get(base_model_candidate)

        if not pricing and model_id.startswith('gpt-4.1-'): 
             pricing = OPENAI_PRICING_PER_TOKEN.get('gpt-4.1')
        
        if not pricing:
            
            
            if model_id.startswith('ft:gpt-3.5-turbo'): 
                
                
                pricing = OPENAI_PRICING_PER_TOKEN.get("gpt-3.5-turbo-0125")
                logger.warning(f"Using placeholder pricing for fine-tuned model '{model_id}'. Verify actual costs.")
            
    if not pricing:
        logger.warning(f"Pricing not found for OpenAI model ID: '{model_id}'. Cost calculation will be skipped for this model.")
        
    return pricing


def get_gemini_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Returns the pricing for Gemini models per 1 million tokens.
    Source: https://ai.google.dev/gemini-api/docs/pricing (Accessed on date of commit)
    """
    
    pricing_data = {
        "gemini-2.5-pro-exp-03-25": {  
            "prompt_tokens_cost_per_million": 1.25,
            "completion_tokens_cost_per_million": 10.00,
        },
        "gemini-2.5-flash-preview-04-17": {  
            "prompt_tokens_cost_per_million": 0.15,
            "completion_tokens_cost_per_million": 0.60,
            "completion_tokens_cost_per_million_thinking": 3.50 
        }
    }
    return pricing_data.get(model_id)


def get_groq_pricing(model_id: str) -> Optional[Dict[str, float | str]]:
    """
    Returns pricing information for Groq models per 1 million tokens.
    Source: https://groq.com/pricing/ (Accessed on date of commit)
    """
    pricing_data = {
        "llama-3.1-8b-instant": {
            "prompt_tokens_cost_per_million": 0.05,
            "completion_tokens_cost_per_million": 0.08,
            "notes": "Prices from groq.com/pricing."
        },
        "llama-3.3-70b-versatile": {
            "prompt_tokens_cost_per_million": 0.59,
            "completion_tokens_cost_per_million": 0.79,
            "notes": "Prices from groq.com/pricing."
        },
        "meta-llama/llama-4-maverick-17b-128e-instruct": {
            "prompt_tokens_cost_per_million": 0.20,
            "completion_tokens_cost_per_million": 0.60,
            "notes": "Prices from groq.com/pricing."
        },
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "prompt_tokens_cost_per_million": 0.11,
            "completion_tokens_cost_per_million": 0.34,
            "notes": "Prices from groq.com/pricing."
        },
        "meta-llama/Llama-Guard-4-12B": {
            "prompt_tokens_cost_per_million": 0.20,
            "completion_tokens_cost_per_million": 0.20,
            "notes": "Prices from groq.com/pricing."
        }
    }
    return pricing_data.get(model_id)
ANTHROPIC_PRICING_PER_MILLION_TOKENS = {
    "claude-3-7-sonnet-20250219": {  
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "claude-3-5-sonnet-20241022": {  
        "prompt_tokens_cost_per_million": 3.00,
        "completion_tokens_cost_per_million": 15.00,
    },
    "claude-3.5-haiku-20241022": {  
        "prompt_tokens_cost_per_million": 0.80,
        "completion_tokens_cost_per_million": 4.00,
    }
}

MISTRAL_OFFICIAL_PRICING_PER_MILLION_TOKENS = {
    "mistral-large-latest": { 
        "prompt_tokens_cost_per_million": 2.00, 
        "completion_tokens_cost_per_million": 6.00,
    },
    "codestral-latest": { 
        "prompt_tokens_cost_per_million": 0.3, 
        "completion_tokens_cost_per_million": 0.9, 
    }
}

WRITER_PALMYRA_PRICING_PER_MILLION_TOKENS = {
    "palmyra-fin-default": {
        "prompt_tokens_cost_per_million": 5.00,
        "completion_tokens_cost_per_million": 12.00,
    }
}


def get_anthropic_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Returns the pricing for Anthropic models per 1 million tokens.
    Source: https://docs.anthropic.com/en/docs/about-claude/pricing
    """
    pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get(model_id)
    
    if not pricing:
        if "claude-3.7-sonnet" in model_id:
            pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get("claude-3-7-sonnet-20250219")
        elif "claude-3.5-sonnet" in model_id: 
            pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get("claude-3-5-sonnet-20241022")
        elif "claude-3.5-haiku" in model_id:
            pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get("claude-3-5-haiku-20241022")
        elif "claude-3-opus" in model_id: 
            pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get("claude-3-opus-20240229")
        elif "claude-3-haiku" in model_id: 
            pricing = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get("claude-3-haiku-20240307")

    if not pricing:
        logger.warning(f"Pricing not found for Anthropic model ID: '{model_id}'. Cost calculation will be skipped for this model.")
    return pricing

def get_mistral_official_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Returns the pricing for official Mistral API models per 1 million tokens.
    """
    pricing = MISTRAL_OFFICIAL_PRICING_PER_MILLION_TOKENS.get(model_id)
    if not pricing:
        logger.warning(f"Pricing not found for Mistral (Official API) model ID: '{model_id}'. Cost calculation will be skipped.")
    return pricing

def get_writer_palmyra_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Returns the pricing for Writer (Palmyra) models per 1 million tokens.
    Source: https://writer.com/llms/palmyra-fin/
    """
    pricing = WRITER_PALMYRA_PRICING_PER_MILLION_TOKENS.get(model_id)
    if not pricing:
        
        
        logger.warning(f"Pricing not found for Writer (Palmyra) model ID: '{model_id}'. Cost calculation will be skipped.")
    return pricing




GROK_PRICING_PER_MILLION_TOKENS = {    
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
    }
}

def get_grok_pricing(model_id: str) -> Optional[Dict[str, float]]:
    """
    Returns the pricing for Grok/Xai models per 1 million tokens.
    """
    pricing = GROK_PRICING_PER_MILLION_TOKENS.get(model_id)
    if not pricing:
        if "grok-3-mini-fast" in model_id:
            pricing = GROK_PRICING_PER_MILLION_TOKENS.get("grok-3-mini-fast-beta")
        elif "grok-3-mini" in model_id:
            pricing = GROK_PRICING_PER_MILLION_TOKENS.get("grok-3-mini-beta")
        elif "grok-3-fast" in model_id:
            pricing = GROK_PRICING_PER_MILLION_TOKENS.get("grok-3-latest")
            
    if not pricing:
        logger.warning(f"Pricing not found for Grok/Xai model ID: '{model_id}'. Cost calculation will be skipped.")
    return pricing

def calculate_model_cost(results_data: List[Dict[str, Any]], model_config_item: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the estimated cost for API calls based on token usage for various model providers.

    Args:
        results_data: A list of dictionaries, where each dictionary contains
                      results from a single LLM call, potentially including
                      'input_tokens' and 'output_tokens'.
        model_config_item: The configuration dictionary for the model run,
                           used to check model type and ID.

    Returns:
        A dictionary containing the total estimated cost, e.g., {'total_cost': 0.123}.
        Returns {'total_cost': 0.0} if the model type is not supported, pricing is missing,
        or token data is unavailable.
    """
    total_cost = 0.0
    model_type = model_config_item.get("type")
    config_id_used = model_config_item.get("config_id", model_config_item.get("model_id"))

    if not config_id_used:
         logger.error("Cannot determine model ID from model_config_item for cost calculation.")
         return {"total_cost": 0.0}
    pricing = None
    price_per_input_token = 0.0
    price_per_output_token = 0.0

    if model_type == "openai":
        pricing = get_openai_pricing(config_id_used)
        if pricing:
            price_per_input_token = pricing.get("input", 0.0)
            price_per_output_token = pricing.get("output", 0.0)

    elif model_type == "groq":
        pricing = get_groq_pricing(config_id_used)
        if pricing:
            
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    elif model_type == "gemini":
        pricing = get_gemini_pricing(config_id_used)
        if pricing:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    elif model_type == "anthropic":
        pricing = get_anthropic_pricing(config_id_used)
        if pricing:
            
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    elif model_type == "mistral_official":
        pricing = get_mistral_official_pricing(config_id_used)
        if pricing:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    elif model_type == "writer": 
        pricing = get_writer_palmyra_pricing(config_id_used)
        if pricing:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    elif model_type == "xai": 
        model_id = model_config_item.get("model_id")
        pricing = get_grok_pricing(model_id)
        if pricing:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
    else:
        logger.warning(f"Cost calculation not supported for model type: '{model_type}'. Cost will be zero.")
        return {"total_cost": 0.0}

    if not pricing:
        logger.warning(f"Skipping cost calculation for run using config ID '{config_id_used}' (type: {model_type}) as pricing is not defined.")
        return {"total_cost": 0.0}

    if price_per_input_token == 0.0 and price_per_output_token == 0.0:
        logger.warning(f"Both input and output token prices are zero for model '{config_id_used}' (type: {model_type}). Cost will be zero.")
        return {"total_cost": 0.0}
    
    num_items_missing_tokens = 0
    for item in results_data:
        input_tokens = item.get('input_tokens')
        output_tokens = item.get('output_tokens')

        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            item_cost = (input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)
            total_cost += item_cost
        else:
            num_items_missing_tokens += 1
            

    if num_items_missing_tokens > 0 and len(results_data) > 0 : 
        logger.warning(f"Cost calculation for {config_id_used} (type: {model_type}) might be incomplete. Missing token data for {num_items_missing_tokens}/{len(results_data)} items.")

    logger.info(f"Estimated cost for {model_type} model {config_id_used}: ${total_cost:.6f}")
    return {"total_cost": total_cost} 