"""
xAI client for handling Grok models.
"""
import time
import logging
from openai import OpenAI
from .. import config

logger = logging.getLogger(__name__)


def get_xai_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle xAI API calls for Grok models.
    
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
    
    if not config.XAI_API_KEY:
        logger.error(f"Missing xAI API key for model {config_id}")
        return {"error_message": "Missing xAI API key", "response_time": 0}
    
    
    parameters.pop('response_format', None)
    
    logger.info(f"Calling xAI API for {config_id}")
    
    xai_client = OpenAI(
        api_key=config.XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )
    
    
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        error_msg = f"Invalid prompt format for xAI model {model_id}"
        return {"error_message": error_msg, "response_time": time.time() - start_time}
    
    try:
        logger.info(f"xAI API call for {config_id} with messages: {messages[:1]}... and parameters: {parameters}")
        
        api_response = xai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            **parameters
        )
        
        logger.info(f"xAI raw API response for {config_id}: {api_response}")
        
        response_text = ""
        if api_response.choices:
            logger.info(f"xAI response has {len(api_response.choices)} choices for {config_id}")
            if api_response.choices[0].message:
                raw_content = api_response.choices[0].message.content
                logger.info(f"xAI raw content for {config_id}: '{raw_content}' (type: {type(raw_content)})")
                response_text = raw_content.strip() if raw_content else ""
                if not response_text:
                    logger.warning(f"xAI returned empty content for {config_id}. Raw content was: {repr(raw_content)}")
            else:
                logger.warning(f"xAI response choice has no message for {config_id}")
        else:
            logger.warning(f"xAI response has no choices for {config_id}")
        
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        
        if hasattr(api_response, 'usage') and api_response.usage:
            input_tokens = api_response.usage.prompt_tokens or 0
            base_completion_tokens = api_response.usage.completion_tokens or 0
            if (hasattr(api_response.usage, 'completion_tokens_details') and
                api_response.usage.completion_tokens_details and
                hasattr(api_response.usage.completion_tokens_details, 'reasoning_tokens')):
                reasoning_tokens = api_response.usage.completion_tokens_details.reasoning_tokens or 0
            
            output_tokens = base_completion_tokens + reasoning_tokens
            
            logger.info(f"xAI usage for {config_id}: Input={input_tokens}, Base={base_completion_tokens}, Reasoning={reasoning_tokens}, Total Output={output_tokens}")
        else:
            logger.warning(f"No usage data available from xAI API for {config_id}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"xAI API call for {config_id} completed in {elapsed_time:.2f}s. Response length: {len(response_text)}")
        
        if not response_text:
            logger.error(f"xAI API returned empty response for {config_id} despite successful API call. Tokens: Input={input_tokens}, Output={output_tokens}")
        
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
        logger.error(f"xAI API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": "",
            "details": {"type": type(e).__name__, "unexpected": True}
        }