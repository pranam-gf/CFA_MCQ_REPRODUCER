"""
Groq client for handling Groq-hosted models.
"""
import time
import logging
from openai import OpenAI
from .. import config

logger = logging.getLogger(__name__)


def get_groq_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle Groq API calls.
    
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
    
    if not config.GROQ_API_KEY:
        logger.error(f"Missing Groq API key for model {config_id}")
        return {"error_message": "Missing Groq API key", "response_time": 0}
    
    logger.info(f"Calling Groq API for {config_id} (JSON Expected: {is_json_response_expected})")
    
    groq_client = OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    
    if is_json_response_expected and parameters.get("response_format", {}).get("type") == "json_object":
        if "json" not in prompt.lower():
            logger.warning(f"Groq model {config_id} called with response_format=json_object, but 'json' not found in prompt")
    else:
        parameters.pop('response_format', None)
     
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        error_msg = f"Invalid prompt format for Groq model {model_id}"
        return {"error_message": error_msg, "response_time": time.time() - start_time}
    
    try:
        logger.debug(f"Groq API call for {config_id} with parameters: {parameters}")
        
        api_response = groq_client.chat.completions.create(
            model=model_id,
            messages=messages,
            **parameters
        )
        
        response_text = ""
        if api_response.choices and api_response.choices[0].message:
            response_text = api_response.choices[0].message.content.strip() if api_response.choices[0].message.content else ""
        
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(api_response, 'usage') and api_response.usage:
            input_tokens = api_response.usage.prompt_tokens or 0
            output_tokens = api_response.usage.completion_tokens or 0
            logger.info(f"Groq usage for {config_id}: Input={input_tokens}, Output={output_tokens}")
        else:
            logger.warning(f"No usage data available from Groq API for {config_id}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Groq API call for {config_id} completed in {elapsed_time:.2f}s. Response length: {len(response_text)}")
        
        if not response_text:
            logger.warning(f"Empty response from Groq API for {config_id}")
        
        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "response_time": elapsed_time,
            "error_message": None,
            "model_config_id": config_id
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Groq API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": "",
            "details": {"type": type(e).__name__, "unexpected": True}
        }