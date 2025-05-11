"""
Model configurations tailored for Chain-of-Thought (CoT) prompting strategies.
"""

ALL_MODEL_CONFIGS_COT = [    
    {
        "config_id": "gemini-2.5-pro-cot",
        "type": "gemini",
        "model_id": "gemini-2.5-pro-exp-03-25",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048
        }
    },
    {
        "config_id": "gemini-2.0-flash-cot",
        "type": "gemini",
        "model_id": "gemini-2.0-flash",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048 
        }
    },
    {
        "config_id": "claude-3.7-sonnet-cot",
        "type": "bedrock",
        "model_id": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "use_inference_profile": True,
        "inference_profile_arn": "arn:aws:bedrock:us-east-1:000518066116:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    {
        "config_id": "claude-3.5-sonnet-cot",
        "type": "bedrock",
        "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    {
        "config_id": "claude-3.5-haiku-cot",
        "type": "bedrock",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 2048,
            "anthropic_version": "bedrock-2023-05-31"
        }
    },
    {
        "config_id": "mistral-large-cot",
        "type": "bedrock",
        "model_id": "mistral.mistral-large-2402-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 2048
        }
    },
    {
        "config_id": "llama3-70b-cot",
        "type": "bedrock",
        "model_id": "meta.llama3-70b-instruct-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 2048 
        }
    },
    {
        "config_id": "llama3.1-8b-cot",
        "type": "bedrock",
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 2048
        }
    },

    
    {
        "config_id": "gpt-4o-cot",
        "type": "openai",
        "model_id": "gpt-4o",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
            
        }
    },
    {
        "config_id": "gpt-o3-cot",
        "type": "openai",
        "model_id": "o3", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_completion_tokens": 8000 
            
        }
    },
    {
        "config_id": "gpt-o4-mini-cot",
        "type": "openai",
        "model_id": "o4-mini", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_completion_tokens": 8000 
            
        }
    },
    {
        "config_id": "gpt-4.1-cot",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
            
        }
    },
    {
        "config_id": "gpt-4.1-mini-cot",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
            
        }
    },
    {
        "config_id": "gpt-4.1-nano-cot",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
            
        }
    },

    
    {
        "config_id": "grok-3-cot",
        "type": "xai",
        "model_id": "grok-3-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
        }
    },

    
    {
        "config_id": "palmyra-fin-cot",
        "type": "writer",
        "model_id": "palmyra-fin",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 2048
        }
    },

    
    {
        "config_id": "deepseek-r1-distill-llama-70b-via-groq-cot",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5, 
            "max_tokens": 4000, 
            "top_p": 0.9
        }
    }
] 