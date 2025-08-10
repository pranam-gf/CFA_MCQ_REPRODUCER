import logging
from typing import List, Dict, Any, Optional
from src.configs.model_pricing import (
    get_pricing,
    get_openai_pricing,
    get_anthropic_pricing,
    get_gemini_pricing,
    get_groq_pricing,
    get_mistral_pricing as get_mistral_official_pricing,
    get_writer_pricing as get_writer_palmyra_pricing,
    get_grok_pricing
)

logger = logging.getLogger(__name__)

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
    model_id = model_config_item.get("model_id", "")
    config_id_used = model_config_item.get("config_id", model_id)

    if not model_id:
         logger.error("Missing model_id from model_config_item for cost calculation.")
         return {"total_cost": 0.0}
         
    pricing = None
    price_per_input_token = 0.0
    price_per_output_token = 0.0
    price_per_reasoning_token = 0.0

    pricing = get_pricing(model_type, model_id)

    if pricing:
        # Handle both old format (direct keys) and new format (input/output keys)
        if "input" in pricing and "output" in pricing:
            price_per_input_token = pricing.get("input", 0.0)
            price_per_output_token = pricing.get("output", 0.0)
        else:
            price_per_input_token = pricing.get("prompt_tokens_cost_per_million", 0.0) / 1_000_000
            price_per_output_token = pricing.get("completion_tokens_cost_per_million", 0.0) / 1_000_000
        
        if model_type == "openai" and model_id.startswith("o3-pro") and "reasoning" in pricing:
            price_per_reasoning_token = pricing.get("reasoning", 0.0)
        elif "reasoning_tokens_cost_per_million" in pricing:
             price_per_reasoning_token = pricing.get("reasoning_tokens_cost_per_million", 0.0) / 1_000_000
    else:
        logger.warning(f"Skipping cost calculation for run using model ID '{model_id}' (type: {model_type}) as pricing is not defined.")
        return {"total_cost": 0.0}

    if price_per_input_token == 0.0 and price_per_output_token == 0.0 and price_per_reasoning_token == 0.0:
        logger.warning(f"All input, output, and reasoning token prices are effectively zero for model '{model_id}' (type: {model_type}) after lookup. Cost will be zero.")
        return {"total_cost": 0.0}
    
    num_items_missing_tokens = 0
    for item in results_data:
        input_tokens = item.get('input_tokens')
        output_tokens = item.get('output_tokens')
        reasoning_tokens = item.get('reasoning_tokens')

        item_cost = 0.0
        has_input_output_tokens = isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float))
        
        if has_input_output_tokens:
            item_cost = (input_tokens * price_per_input_token) + (output_tokens * price_per_output_token)
            if model_type == "openai" and model_id.startswith("o3-pro") and isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0 and price_per_reasoning_token > 0:
                item_cost += (reasoning_tokens * price_per_reasoning_token)
                logger.debug(f"Added reasoning cost for {model_id}: {reasoning_tokens} tokens * {price_per_reasoning_token} price/token")
            total_cost += item_cost
        else:
            num_items_missing_tokens += 1
            

    if num_items_missing_tokens > 0 and len(results_data) > 0 : 
        logger.warning(f"Cost calculation for {model_id} (type: {model_type}) might be incomplete. Missing token data for {num_items_missing_tokens}/{len(results_data)} items.")

    logger.info(f"Estimated cost for {model_type} model {model_id}: ${total_cost:.6f}")
    return {"total_cost": total_cost}