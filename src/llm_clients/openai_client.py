"""
OpenAI client for handling GPT models including reasoning models.
"""
import time
import logging
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from .. import config

logger = logging.getLogger(__name__)

def get_openai_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle OpenAI API calls for all OpenAI models including reasoning models.
    
    Args:
        prompt: The prompt string or list of messages
        model_config: Model configuration dictionary
        is_json_response_expected: Whether to expect JSON response format
        
    Returns:
        Dict with response data or error information
    """
    model_id = model_config.get("model_id")
    config_id = model_config.get("config_id", model_id)
    parameters = model_config.get("parameters", {}).copy()
    start_time = time.time()
    
    if not config.OPENAI_API_KEY:
        logger.error(f"Missing OpenAI API key for model {config_id}.")
        return {"error_message": "Missing OpenAI API key", "response_time": 0}
        
    if is_json_response_expected and parameters.get("response_format", {}).get("type") == "json_object":
        if "json" not in prompt.lower():
            logger.warning(f"OpenAI model {config_id} called with response_format=json_object, but 'json' not found in prompt.")
    elif not is_json_response_expected:
        parameters.pop('response_format', None)
    
    logger.info(f"Calling OpenAI API for {config_id} (JSON Expected: {is_json_response_expected})")
    
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    response_text = ""
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    
    try:
        
        if model_id == "o3-pro" or model_id.startswith("o3-pro-"):
            logger.info(f"Using /v1/responses endpoint for {model_id}")
            
            
            if isinstance(prompt, str):
                input_messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                input_messages = prompt
            else:
                error_msg = f"Invalid prompt format for {model_id} (responses endpoint expects list of messages or string)"
                return {"error_message": error_msg, "response_time": time.time() - start_time}  
            responses_params = {k: v for k, v in parameters.items() if k not in ["messages", "prompt"]}
            reasoning_config = responses_params.pop("reasoning", {"effort": "medium"})
            unsupported_params = ["response_format", "tools", "tool_choice"]
            for param_key in unsupported_params:
                if param_key in responses_params:
                    logger.warning(f"Parameter '{param_key}' not supported for {model_id} (responses endpoint)")
                    del responses_params[param_key]
            
            api_response = openai_client.responses.create(
                model=model_id,
                input=input_messages,
                reasoning=reasoning_config,
                **responses_params
            )     
            if api_response.choices and api_response.choices[0].message:
                response_text = api_response.choices[0].message.content.strip() if api_response.choices[0].message.content else ""
            else:
                logger.warning(f"No valid choices or message content found for {config_id}")
                response_text = ""
            
            if hasattr(api_response, 'usage') and api_response.usage:
                input_tokens = api_response.usage.input_tokens or 0
                output_tokens = api_response.usage.completion_tokens or 0
                reasoning_tokens = api_response.usage.reasoning_tokens or 0  
        else:
            
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = prompt
            else:
                error_msg = f"Invalid prompt format for OpenAI chat model {model_id}"
                return {"error_message": error_msg, "response_time": time.time() - start_time}
            chat_params = parameters.copy()
            chat_params.pop("reasoning", None)
            
            api_response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                **chat_params
            )
            
            if api_response.choices and api_response.choices[0].message:
                response_text = api_response.choices[0].message.content.strip() if api_response.choices[0].message.content else ""
            else:
                response_text = ""
            
            if hasattr(api_response, 'usage') and api_response.usage:
                input_tokens = api_response.usage.prompt_tokens or 0
                output_tokens = api_response.usage.completion_tokens or 0
        
        elapsed_time = time.time() - start_time
        logger.info(f"OpenAI API call for {config_id} completed in {elapsed_time:.2f}s")
        
        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "response_time": elapsed_time,
            "error_message": None,
            "model_config_id": config_id
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"OpenAI API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": response_text,
            "details": {"type": type(e).__name__, "unexpected": True}
        }