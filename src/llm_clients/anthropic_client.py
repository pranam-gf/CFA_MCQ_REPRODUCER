"""
Anthropic client for handling Claude models.
"""
import time
import logging
import os
import httpx
import anthropic
from .. import config

logger = logging.getLogger(__name__)


def get_anthropic_response(prompt: str | list[dict[str, str]], model_config: dict, is_json_response_expected: bool = False) -> dict:
    """
    Handle Anthropic API calls for Claude models with error recovery for streaming interruptions.
    
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
    
    if not config.ANTHROPIC_API_KEY:
        logger.error(f"Missing Anthropic API key for model {config_id}.")
        return {"error_message": "Missing Anthropic API key", "response_time": 0}    
    parameters.pop('response_format', None)    
    thinking_config = None
    if 'thinking' in parameters:
        thinking_config = parameters.pop('thinking')
        logger.info(f"Extended thinking enabled for {config_id} with budget: {thinking_config.get('budget_tokens', 'default')}")
    
    logger.info(f"Calling Anthropic API for {config_id}")
    
    anthropic_client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY,
        timeout=httpx.Timeout(timeout=60.0 * 60.0)
    )    
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        messages = prompt
    else:
        error_msg = f"Invalid prompt format for Anthropic model {model_id}"
        return {"error_message": error_msg, "response_time": time.time() - start_time}
    max_retries = 3
    retry_count = 0
    response_text = ""
    thinking_text = ""
    input_tokens = 0
    output_tokens = 0
    
    while retry_count <= max_retries:
        try:
            api_params = {
                "model": model_id,
                "messages": messages,
                **parameters
            }
            if thinking_config:
                api_params["thinking"] = thinking_config
            response_text = ""
            thinking_text = ""
            input_tokens = 0
            output_tokens = 0
            with anthropic_client.messages.stream(**api_params) as stream:
                for event in stream:
                    if event.type == "message_start":
                        if event.message.usage and hasattr(event.message.usage, 'input_tokens'):
                            input_tokens = event.message.usage.input_tokens
                            logger.info(f"Anthropic stream: input_tokens from message_start: {input_tokens}")
                            
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            response_text += event.delta.text
                        elif event.delta.type == "thinking_delta":
                            if hasattr(event.delta, 'thinking'):
                                thinking_text += event.delta.thinking
                            elif hasattr(event.delta, 'content'):
                                thinking_text += event.delta.content
                            else:
                                logger.debug(f"Unknown thinking_delta structure: {event.delta}")
                        elif event.delta.type == "signature_delta":
                            logger.debug(f"Received signature_delta for thinking content verification")
                            
                    elif event.type == "message_delta":
                        if event.usage and hasattr(event.usage, 'output_tokens'):
                            output_tokens = event.usage.output_tokens
                            logger.debug(f"Anthropic stream: cumulative output_tokens from message_delta: {output_tokens}")
                            
                    elif event.type == "message_stop":
                        logger.info(f"Anthropic stream: message_stop event received")
                        final_message = stream.get_final_message()
                        if final_message and final_message.usage:
                            if input_tokens == 0 and hasattr(final_message.usage, 'input_tokens'):
                                input_tokens = final_message.usage.input_tokens
                            if hasattr(final_message.usage, 'output_tokens'):
                                output_tokens = final_message.usage.output_tokens
                                
                    elif event.type == "error":
                        logger.error(f"Anthropic stream error for {config_id}: {event.error}")
                        error_details = {
                            "type": event.error.get("type", "stream_error"),
                            "message": event.error.get("message", "Unknown stream error")
                        }
                        return {
                            "error_message": f"Anthropic Stream Error: {event.error.get('type', 'unknown')}",
                            "response_time": time.time() - start_time,
                            "details": error_details,
                            "response_text": response_text,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }
                break  
                
        except (httpx.RemoteProtocolError, anthropic.APIConnectionError) as e:
            retry_count += 1
            logger.warning(f"Network error for {config_id} (attempt {retry_count}/{max_retries + 1}): {str(e)}")
            if retry_count > max_retries:
                elapsed_time = time.time() - start_time
                logger.error(f"Max retries exceeded for {config_id} after {elapsed_time:.2f}s")
                return {
                    "error_message": f"Network error after {max_retries} retries: {str(e)}",
                    "response_time": elapsed_time,
                    "details": {"type": type(e).__name__, "retries_attempted": retry_count - 1},
                    "response_text": response_text
                }
            
            wait_time = min(2 ** (retry_count - 1), 10)  
            logger.info(f"Waiting {wait_time}s before retry {retry_count} for {config_id}")
            time.sleep(wait_time)
            continue
            
        except anthropic.APIStatusError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Anthropic API Status Error for {config_id} after {elapsed_time:.2f}s: {e.status_code} - {e.message}")
            error_details = {"type": "APIStatusError", "message": str(e.message), "status_code": e.status_code}
            return {
                 "error_message": f"Anthropic API Status Error: {e.status_code} - {e.message}",
                 "response_time": elapsed_time,
                 "details": error_details,
                 "response_text": response_text
             }
            
        except anthropic.RateLimitError as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Anthropic Rate Limit Error for {config_id} after {elapsed_time:.2f}s: {e}")
            return {
                 "error_message": f"Anthropic Rate Limit Error: {str(e)}",
                 "response_time": elapsed_time,
                 "details": {"type": "RateLimitError"},
                 "response_text": response_text
             }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Anthropic API Error for {config_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
            return {
                 "error_message": str(e),
                 "response_time": elapsed_time,
                 "response_text": response_text,
                 "details": {"type": type(e).__name__, "unexpected": True}
             }
    
    
    response_text = response_text.strip()
    elapsed_time = time.time() - start_time
    
    if input_tokens == 0 or output_tokens == 0:
        logger.warning(f"Anthropic stream for {config_id}: Token counts missing. Input: {input_tokens}, Output: {output_tokens}")
    
    logger.info(f"Anthropic API call for {config_id} completed in {elapsed_time:.2f}s. Response length: {len(response_text)}")
    
    result = {
        "response_text": response_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "response_time": elapsed_time,
        "error_message": None,
        "model_config_id": config_id
    }        
    if thinking_text:
        result["thinking_text"] = thinking_text
        logger.info(f"Extended thinking captured for {config_id}, thinking length: {len(thinking_text)}")
    
    return result