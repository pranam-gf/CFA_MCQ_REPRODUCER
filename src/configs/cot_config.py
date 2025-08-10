"""
Model configurations tailored for Chain-of-Thought (CoT) prompting strategies.
"""

ALL_MODEL_CONFIGS_COT = [    
    {
        "config_id": "gemini-2.5-pro-cot",
        "type": "gemini",
        "model_id": "gemini-2.5-pro-preview-05-06",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536
        }
    },
    {
        "config_id": "gemini-2.5-flash-cot",
        "type": "gemini",
        "model_id": "gemini-2.5-flash-preview-04-17",
        "prompt_strategy_type": "COHERENT_CFA_COT", 
        "parameters": {
            "temperature": 0.5, 
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 65536,
            "thinking_budget": 24576
        }
    },
    {
        "config_id": "claude-3.7-sonnet-cot",
        "type": "anthropic",
        "model_id": "claude-3-7-sonnet-20250219",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "claude-opus-4-cot",
        "type": "anthropic",
        "model_id": "claude-opus-4-20250514",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 32000
        }
    },
    {
        "config_id": "claude-opus-4.1-cot",
        "type": "anthropic",
        "model_id": "claude-opus-4-1-20250805",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32000
     
        }
    },
    {
        "config_id": "claude-opus-4.1-thinking",
        "type": "anthropic",
        "model_id": "claude-opus-4-1-20250805",
        "parameters": {
            "temperature": 1,
            "max_tokens": 32000,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 30000
            }
        }
    },
    {
        "config_id": "claude-sonnet-4-cot",
        "type": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "claude-3.5-sonnet-cot",
        "type": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku-cot",
        "type": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "mistral-large-official-cot",
        "type": "mistral_official",
        "model_id": "mistral-large-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 4096 
        }
    },
    {
        "config_id": "codestral-latest-official",
        "type": "mistral_official",
        "model_id": "codestral-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4o-cot",
        "type": "openai",
        "model_id": "gpt-4o",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
            
        }
    },
    {
        "config_id": "o3-mini",
        "type": "openai",
        "model_id": "o3-mini-2025-01-31",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "o4-mini-cot",
        "model_id": "o4-mini-2025-04-16",
        "type": "openai",
        "name": "O4 Mini (CoT)",
        "temperature": 0.7,
    },
    {
        "config_id": "o3-pro-cot",
        "model_id": "o3-pro-2025-06-10",
        "type": "openai",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning": {"effort": "high"}
        }
    },
    {
        "config_id": "gpt-4.1-cot",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-mini-cot",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
            
        }
    },
    {
        "config_id": "gpt-4.1-nano-cot",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "grok-3-mini-beta-high-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high"
        }
    },
    {
        "config_id": "grok-3-mini-beta-low-effort",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low"
        }
    },
    {
        "config_id": "grok-3-cot",
        "type": "xai",
        "model_id": "grok-3-latest",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },
    {
        "config_id": "grok-4-0709-cot",
        "type": "xai",
        "model_id": "grok-4-0709",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0
        }
    },

    
    {
        "config_id": "palmyra-fin-default",
        "type": "writer",
        "model_id": "palmyra-fin",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.5
        }
    },
    {
        "config_id": "deepseek-r1-cot",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b", 
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-maverick",
        "type": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama-4-scout",
        "type": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-llama3.3-70b-cot",
        "type": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant-cot",
        "type": "groq",
        "model_id": "llama-3.1-8b-instant",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "groq-gpt-oss-20b-cot",
        "type": "groq",
        "model_id": "openai/gpt-oss-20b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature":0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "groq-gpt-oss-120b-cot",
        "type": "groq",
        "model_id": "openai/gpt-oss-120b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "groq-kimi-k2-cot",
        "type": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1
        }
    },
    {
        "config_id": "groq-qwen3-32b-cot",
        "type": "groq",
        "model_id": "qwen/qwen3-32b",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "gpt-5-cot",
        "type": "openai",
        "model_id": "gpt-5-2025-08-07",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning": {"effort": "high"}
            
        }
    },
    {
        "config_id": "gpt-5-mini-cot",
        "type": "openai",
        "model_id": "gpt-5-mini-2025-08-07",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning": {"effort": "high"}
        }
    },
    {
        "config_id": "gpt-5-nano-cot",
        "type": "openai",
        "model_id": "gpt-5-nano-2025-08-07",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning": {"effort": "high"}
        }
    },
    {
        "config_id": "o1-mini-cot",
        "type": "openai",
        "model_id": "o1-mini-2024-09-12",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning":{"effort": "high"},
            "max_tokens": 65536
        }
    },
    {
        "config_id": "o3-cot",
        "type": "openai",
        "model_id": "o3-2025-04-16",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning":{"effort": "high"},
            "max_tokens": 100000
        }
    },
    {
        "config_id": "o3-pro-cot",
        "type": "openai",
        "model_id": "o3-pro-2025-06-10",
        "prompt_strategy_type": "COHERENT_CFA_COT",
        "parameters": {
            "reasoning": {"effort": "high"},
            "max_tokens": 100000
        }
    }
]