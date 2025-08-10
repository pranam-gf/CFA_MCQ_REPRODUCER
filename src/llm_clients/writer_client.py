"""
Writer client for handling Writer models.
"""
import time
import logging
import writerai
from writerai import Writer
from .. import config
logger = logging.getLogger(__name__)

def _estimate_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int | None:
    """Estimates token count for a given text using tiktoken."""
    if not text:
        return 0
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Could not estimate tokens using tiktoken: {e}")
        return None

def get_writer_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle Writer API calls.
    
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
    
    if not config.WRITER_API_KEY:
        logger.error(f"Missing Writer API key for model {config_id}")
        return {"error_message": "Missing Writer API key", "response_time": 0}
    
    parameters.pop('response_format', None)
    logger.info(f"Calling Writer API for {config_id}")
    writer_client = Writer(api_key=config.WRITER_API_KEY)
    
    
    if isinstance(prompt, list):
        
        prompt_text = " ".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict) and msg.get("content")])
    else:
        prompt_text = prompt
    try:
        api_response = writer_client.chat.chat(
            messages=[{"role": "user", "content": prompt_text}],
            model=model_id,
            **parameters
        )
        
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        
        if api_response.choices and api_response.choices[0].message:
            response_text = api_response.choices[0].message.content.strip() if api_response.choices[0].message.content else ""
        
        if hasattr(api_response, 'usage') and api_response.usage:
            input_tokens = getattr(api_response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(api_response.usage, 'completion_tokens', 0) or 0
            logger.info(f"Writer usage for {config_id}: Input={input_tokens}, Output={output_tokens}")
        else:
            logger.warning(f"No usage data from Writer API for {config_id}, estimating tokens")
            input_tokens = _estimate_tokens_tiktoken(prompt_text) or 0
            output_tokens = _estimate_tokens_tiktoken(response_text) or 0
            logger.info(f"Writer estimated tokens for {config_id}: Input={input_tokens}, Output={output_tokens}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Writer API call for {config_id} completed in {elapsed_time:.2f}s. Response length: {len(response_text)}")
        
        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": 0,
            "response_time": elapsed_time,
            "error_message": None,
            "model_config_id": config_id
        }
        
    except writerai.APIError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Writer API Error for {config_id} after {elapsed_time:.2f}s: {e}")
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": "",
            "details": {"type": "WriterAPIError"}
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Writer API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
        return {
            "error_message": str(e),
            "response_time": elapsed_time,
            "response_text": "",
            "details": {"type": type(e).__name__, "unexpected": True}
        }