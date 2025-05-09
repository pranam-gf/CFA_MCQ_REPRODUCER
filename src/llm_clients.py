"""
Functions for interacting with various Language Model APIs.
"""
import time
import json
import logging
import re
import boto3
from openai import OpenAI
import google.generativeai as genai
from writerai import Writer


from . import config

logger = logging.getLogger(__name__)

def generate_prompt(entry: dict, model_type: str | None = None) -> str:
    """Generates the prompt for the LLM."""
    vignette = entry.get('vignette', 'Vignette not available.')
    question_full_text = entry.get('question', 'Question not available.')

    if model_type == "gemini":
        return f"""Given the following context (vignette) and a multiple-choice question (which includes the question stem and all options), please analyze the information and select the best answer.

Respond with ONLY the single uppercase letter of your chosen option (e.g., A, B, C). Do not include any other text, explanation, or punctuation.

Context (Vignette):
{vignette}

Question and Options:
{question_full_text}

Your chosen option letter:"""
    else:
        prompt_instruction = "Respond with ONLY the single letter corresponding to the correct answer (e.g., A, B, or C). Do not include any other text, explanation, or formatting."
        return f"""
Consider the following vignette:
{vignette}

Based on the vignette, answer the following question:
{question_full_text}

{prompt_instruction}
"""

def get_llm_response(prompt: str, model_config: dict) -> dict | None:
    """
    Sends a prompt to the specified LLM API and parses the single letter response.
    Also measures response time and attempts to extract token counts.

    Args:
        prompt: The prompt string to send to the LLM.
        model_config: Dictionary containing model type, ID, parameters.

    Returns:
        A dictionary containing {'response_json': {'answer': <letter>, 'explanation': ''},
                                'response_time', 'input_tokens', 'output_tokens'},
        or None if the API call failed or response parsing failed.
    """
    model_type = model_config.get("type")
    model_id = model_config.get("model_id")
    parameters = model_config.get("parameters", {}).copy() 
    config_id = model_config.get("config_id", model_id) 

    
    parameters.pop('response_format', None)
    
    

    logger.info(f"Sending prompt to {model_type} model: {config_id} (requesting single letter answer)")
    start_time = time.time()
    input_tokens = None
    output_tokens = None
    response_text_for_error = "N/A"
    elapsed_time = 0

    try:
        if model_type == "bedrock":
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                logger.error(f"Missing AWS credentials for Bedrock model {config_id}.")
                return None
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            accept = 'application/json' 
            contentType = 'application/json' 

            if "anthropic" in model_id:
                messages = [{"role": "user", "content": prompt}]
                
                
                body = json.dumps({
                    "messages": messages,
                    "anthropic_version": parameters.get("anthropic_version", "bedrock-2023-05-31"),
                    **{k: v for k, v in parameters.items() if k not in ["anthropic_version"]}
                })
                model_identifier = model_config.get("inference_profile_arn") if model_config.get("use_inference_profile", False) else model_id
                api_response = bedrock_client.invoke_model(
                    body=body, modelId=model_identifier, accept=accept, contentType=contentType
                )
                response_body = json.loads(api_response.get('body').read())
                response_text_for_error = response_body.get('content', [{}])[0].get('text', '')
                input_tokens = response_body.get('usage', {}).get('input_tokens')
                output_tokens = response_body.get('usage', {}).get('output_tokens')
            elif "mistral" in model_id or "meta" in model_id: 
                body = json.dumps({"prompt": prompt, **parameters})
                api_response = bedrock_client.invoke_model(
                    body=body, modelId=model_id, accept=accept, contentType=contentType
                )
                response_body = json.loads(api_response.get('body').read())
                if "mistral" in model_id:
                    response_text_for_error = response_body.get('outputs', [{}])[0].get('text', '')
                    if 'usage' in response_body:
                        input_tokens = response_body['usage'].get('input_tokens')
                        output_tokens = response_body['usage'].get('output_tokens')
                elif "meta" in model_id: 
                    response_text_for_error = response_body.get('generation', '')
                    if hasattr(api_response, 'get') and api_response.get('ResponseMetadata'):
                        headers = api_response.get('ResponseMetadata', {}).get('HTTPHeaders', {})
                        input_tokens = int(headers.get('x-amzn-bedrock-input-token-count', 0))
                        output_tokens = int(headers.get('x-amzn-bedrock-output-token-count', 0))
                    elif 'usage' in response_body:
                         input_tokens = response_body['usage'].get('input_tokens')
                         output_tokens = response_body['usage'].get('output_tokens')
                if input_tokens is None:
                    logger.warning(f"Token count not found for {config_id}, estimating.")
                    input_tokens = len(prompt.split()) * 1.3
                    output_tokens = len(response_text_for_error.split()) * 1.3
            else:
                logger.error(f"Unsupported Bedrock model provider for {config_id}")
                return None

        elif model_type == "openai":
            if not config.OPENAI_API_KEY:
                logger.error(f"Missing OpenAI API key for model {config_id}.")
                return None
            openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            api_response = openai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters 
            )
            response_text_for_error = api_response.choices[0].message.content.strip()
            if api_response.usage:
                input_tokens = api_response.usage.prompt_tokens
                output_tokens = api_response.usage.completion_tokens

        elif model_type == "xai":
            if not config.XAI_API_KEY:
                logger.error(f"Missing xAI API key for model {config_id}.")
                return None
            xai_client = OpenAI(
                api_key=config.XAI_API_KEY,
                base_url="https://api.x.ai/v1",
            )
            api_response = xai_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **parameters
            )
            response_text_for_error = api_response.choices[0].message.content.strip()
            if hasattr(api_response, 'usage') and api_response.usage:
                input_tokens = api_response.usage.prompt_tokens
                output_tokens = api_response.usage.completion_tokens

        elif model_type == "gemini":
            if not config.GEMINI_API_KEY:
                logger.error(f"Missing Gemini API key for model {config_id}.")
                return None
            genai.configure(api_key=config.GEMINI_API_KEY)
            
            gemini_model_instance = genai.GenerativeModel(model_id)
            
            logger.debug(f"Gemini prompt for {config_id}:\n{prompt}")
            api_response = gemini_model_instance.generate_content(
                prompt
            )
            logger.debug(f"Gemini raw api_response object for {config_id}: {api_response}")
            if hasattr(api_response, 'prompt_feedback'):
                logger.info(f"Gemini prompt_feedback for {config_id}: {api_response.prompt_feedback}")
            
            if hasattr(api_response, 'candidates') and api_response.candidates:
                logger.info(f"Gemini candidates object for {config_id}: {api_response.candidates}")
                for i, cand in enumerate(api_response.candidates):
                    finish_reason = getattr(cand, 'finish_reason', 'N/A')
                    logger.info(f"Candidate {i} finish_reason: {finish_reason}")
                    if hasattr(cand, 'content') and cand.content and hasattr(cand.content, 'parts'):
                        logger.info(f"Candidate {i} content parts: {cand.content.parts}")
                    else:
                        logger.info(f"Candidate {i} has no content parts or content structure is different.")
            else:
                logger.info(f"No candidates found in Gemini response for {config_id}")


            response_text_for_error = "" 
            if hasattr(api_response, 'text') and api_response.text: 
                response_text_for_error = api_response.text.strip()
            elif hasattr(api_response, 'candidates') and api_response.candidates:
                try:
                    candidate = api_response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        if hasattr(candidate.content.parts[0], 'text'):
                             response_text_for_error = candidate.content.parts[0].text.strip()
                        else:
                            logger.warning(f"Gemini candidate {config_id} content part has no text attribute: {candidate.content.parts[0]}")
                    
                    if hasattr(candidate, 'finish_reason'):
                         logger.info(f"Gemini candidate finish_reason for {config_id}: {candidate.finish_reason}")
                except Exception as e_parse_candidate:
                    logger.warning(f"Error parsing Gemini candidate for {config_id}: {e_parse_candidate}")
            
            if not response_text_for_error:
                logger.warning(f"Gemini response text is empty for {config_id} after checking .text and .candidates[0].content.parts[0].text.")
                
                if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback.block_reason:
                    logger.error(f"Gemini content blocked for {config_id}. Reason: {api_response.prompt_feedback.block_reason} - Details: {api_response.prompt_feedback.safety_ratings}")
                elif hasattr(api_response, 'candidates') and api_response.candidates:
                    for i, cand in enumerate(api_response.candidates):
                        finish_reason = getattr(cand, 'finish_reason', 'N/A')
                        if finish_reason != 'STOP' and finish_reason != 'MAX_TOKENS':
                             logger.warning(f"Gemini candidate {i} for {config_id} had finish_reason: {finish_reason}, and no text was extracted.")

            logger.info(f"Gemini raw response text for {config_id}: {response_text_for_error[:200]}...")

            if hasattr(api_response, 'usage_metadata') and api_response.usage_metadata:
                input_tokens = api_response.usage_metadata.prompt_token_count
                output_tokens = api_response.usage_metadata.candidates_token_count
            elif input_tokens is None:
                logger.warning(f"Gemini token count not found for {config_id}, estimating.")
                input_tokens = len(prompt.split()) * 1.3 
                output_tokens = len(response_text_for_error.split()) * 1.3

        elif model_type == "writer":
            if not config.WRITER_API_KEY:
                logger.error(f"Missing Writer API key for model {config_id}.")
                return None
            writer_client = Writer(api_key=config.WRITER_API_KEY)
            writer_params = {k: v for k, v in parameters.items() if k in ["temperature", "max_tokens"]}
            api_response = writer_client.chat.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                **writer_params
            )
            response_text_for_error = api_response.choices[0].message.content.strip()
            if hasattr(api_response, 'usage') and api_response.usage:
                input_tokens = api_response.usage.prompt_tokens
                output_tokens = api_response.usage.completion_tokens
            else:
                logger.warning(f"Writer token count not found for {config_id}, estimating.")
                input_tokens = len(prompt.split()) * 1.3
                output_tokens = len(response_text_for_error.split()) * 1.3
                
        elif model_type == "sagemaker":
            if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
                logger.error(f"Missing AWS credentials for SageMaker model {config_id}.")
                return None
            sagemaker_client = boto3.client(
                service_name='sagemaker-runtime',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            body_payload = {"inputs": prompt, "parameters": parameters}
            
            
            body = json.dumps(body_payload)

            endpoint_name = model_id.split('/')[-1] 
            api_response = sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=body
            )
            response_body = json.loads(api_response['Body'].read().decode())
            
            
            
            
            if isinstance(response_body, list) and len(response_body) > 0 and isinstance(response_body[0], dict):
                if 'generated_text' in response_body[0]:
                    response_text_for_error = response_body[0]['generated_text']
                
                else: 
                    response_text_for_error = str(response_body[0]) 
            elif isinstance(response_body, dict):
                
                for key in ['generated_text', 'text', 'answer', 'completion', 'generation']:
                    if key in response_body:
                        response_text_for_error = response_body[key]
                        break
                else: 
                    response_text_for_error = str(response_body)
            else:
                response_text_for_error = str(response_body) 
            
            logger.warning(f"SageMaker token count not directly available for {config_id}, estimating.")
            input_tokens = len(prompt.split()) * 1.3 
            output_tokens = len(response_text_for_error.split()) * 1.3
        else:
            logger.error(f"Unsupported model type: {model_type} for model {config_id}")
            return None

        elapsed_time = time.time() - start_time
        logger.info(f"Response from {config_id} received in {elapsed_time:.2f}s. Raw: '{response_text_for_error[:100]}...'")

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"API call to {config_id} failed after {elapsed_time:.2f}s: {e}. Raw response snippet: {response_text_for_error[:100]}...")
        return None

    
    
    cleaned_answer = "X"

    if model_type == "gemini": 
        pass 
    elif response_text_for_error: 
        
        
        match = re.search(r"^(?:[^a-zA-Z0-9]*?(?:Answer(?: is)?)?:?\s*)?([A-Z])", response_text_for_error.strip(), re.IGNORECASE)
        if match:
            cleaned_answer = match.group(1).upper()
            logger.info(f"Extracted answer '{cleaned_answer}' for {config_id} from '{response_text_for_error[:50]}...'")
        else:
            logger.error(f"Could not extract single letter answer for {config_id} from '{response_text_for_error[:50]}...'. Marking as X.")
    else:
        logger.error(f"Empty or no usable response text received for {config_id}. Marking as X.")

    
    
    response_json = {
        "answer": cleaned_answer,
        "explanation": ""  
    }
    
    logger.debug(f"Final parsed response for {config_id}: {response_json}")

    return {
        "response_json": response_json,
        "response_time": elapsed_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens 
                                      
                                      
                                      
                                      
    }

def process_questions_with_llm(data: list[dict], model_config_item: dict) -> list[dict]:
    """
    Processes each question entry using the specified LLM, expecting only a single letter answer.
    """
    results = []
    total_questions = len(data)
    config_id = model_config_item.get("config_id", model_config_item.get("model_id"))
    model_type = model_config_item.get("type")

    for i, entry in enumerate(data):
        logger.info(f"Processing question {i+1}/{total_questions} with model {config_id}...")
        prompt = generate_prompt(entry, model_type=model_type)
        llm_data = get_llm_response(prompt, model_config_item)

        llm_answer = ""
        is_correct = None
        response_time = None
        current_input_tokens = None
        current_output_tokens = None
        answer_length = 0
        

        if llm_data and llm_data.get('response_json'):
            parsed_response = llm_data['response_json']
            response_time = llm_data['response_time']
            current_input_tokens = llm_data.get('input_tokens')
            current_output_tokens = llm_data.get('output_tokens')

            llm_answer = parsed_response.get("answer", "").strip().upper()
            answer_length = len(llm_answer)
            
            
            if "error_message" in llm_data:
                logger.warning(f"Q {i+1} ({config_id}): Problem with LLM response: {llm_data['error_message']}")
                

            correct_answer_str = str(entry.get('correctAnswer', '')).strip().upper()
            if not correct_answer_str or "PLACEHOLDER" in correct_answer_str:
                logger.warning(f"Q {i+1} ({config_id}): Correct answer missing/placeholder. Cannot evaluate correctness.")
                is_correct = None
            elif llm_answer == "X":
                is_correct = False 
                logger.info(f"Q {i+1} ({config_id}): LLM response parsed as 'X' (see previous error for details). Marked as incorrect.")
            elif not llm_answer:
                is_correct = False
                logger.warning(f"Q {i+1} ({config_id}): LLM provided an empty or non-standard invalid answer ('{llm_answer}'). Marked as incorrect.")
            else:
                is_correct = (llm_answer == correct_answer_str)
            
            logger.info(f"Q {i+1} ({config_id}): LLM Ans: '{llm_answer}', Correct: '{correct_answer_str}', Match: {is_correct}, Time: {response_time:.2f}s")

        else:
            if llm_data:
                response_time = llm_data.get('response_time', 0)
                logger.error(f"Q {i+1} ({config_id}): Failed to parse LLM response or extract answer (internal error in get_llm_response) after {response_time:.2f}s.")
            else:
                logger.error(f"Q {i+1} ({config_id}): Failed to get any valid LLM response (API call likely failed).")
            llm_answer = "ERROR" 
            is_correct = False 

        updated_entry = entry.copy()
        updated_entry.update({
            'LLM_answer': llm_answer,
            'is_correct': is_correct,
            'response_time': response_time,
            'input_tokens': current_input_tokens,
            'output_tokens': current_output_tokens,
            'answer_length': answer_length,
            
        })
        results.append(updated_entry)
    return results 