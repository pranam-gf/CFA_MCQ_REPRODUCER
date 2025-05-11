"""
Default model configurations for single-letter answer strategies.
"""

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
        "max_tokens": 20,
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
            "max_tokens": 20,
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
            "max_tokens": 20,
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
            "max_tokens": 20
        }
    },
    {
        "config_id": "llama3-70b",
        "type": "bedrock",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_gen_len": 20
        }
    },
    {
        "config_id": "llama3.1-8b",
        "type": "bedrock",
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_gen_len": 20
        }
    },
    
    {
        "config_id": "gpt-4o",
        "type": "openai",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    
    {
        "config_id": "gpt-o3",
        "type": "openai",
        "model_id": "o3", 
        "parameters": {
            "temperature": 0.1,
            "max_completion_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-o4-mini",
        "type": "openai",
        "model_id": "o4-mini", 
        "parameters": {
            "temperature": 0.1,
            "max_completion_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-mini",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "gpt-4.1-nano",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20,
            "response_format": {"type": "json_object"}
        }
    },
    
    {
        "config_id": "grok-3",
        "type": "xai",
        "model_id": "grok-3-latest",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20
        }
    },
    
    {
    "config_id": "gemini-2.5-pro",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-exp-03-25",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 20
    }
    },
    {
    "config_id": "gemini-2.0-flash",
    "type": "gemini",
    "model_id": "gemini-2.0-flash",
    "parameters": {
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 20
    }
    },
        {
        "config_id": "palmyra-fin",
        "type": "writer",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 20
        }
    },
    {
        "config_id": "deepseek-r1-distill-llama-70b-via-groq",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b",
        "parameters": {
            "temperature": 0.6,
            "max_tokens": 20,
            "top_p": 0.9
        }
    }
] 