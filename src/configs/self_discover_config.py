"""
Model configurations specifically for the Self-Discover strategy.
Initially mirrors default configs, can be tuned later.
"""

SELF_DISCOVER_CONFIGS = [
    
    {
    "config_id": "claude-3.7-sonnet-self-discover",
    "type": "anthropic",
    "model_id": "claude-3-7-sonnet-20250219",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens": 64000
    }
    },
    {
        "config_id": "claude-opus-4-self-discover",
        "type": "anthropic",
        "model_id": "claude-opus-4-20250514",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 32000
        }
    },
    {
        "config_id": "claude-opus-4.1-self-discover",
        "type": "anthropic",
        "model_id": "claude-opus-4-1-20250805",
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
                "budget_tokens": 31000
            }
        }
    },
    {
        "config_id": "claude-sonnet-4-self-discover",
        "type": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 64000
        }
    },
    {
        "config_id": "claude-3.5-sonnet-self-discover",
        "type": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "claude-3.5-haiku-self-discover",
        "type": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.999,
            "top_k": 250,
            "max_tokens": 8192
        }
    },
    {
        "config_id": "mistral-large-official-self-discover",
        "type": "mistral_official",
        "model_id": "mistral-large-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 4096
        }
    },
    {
        "config_id": "codestral-latest-self-discover",
        "type": "mistral_official",
        "model_id": "codestral-latest",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768 
        }
    },
    {
        "config_id": "gpt-4o-self-discover",
        "type": "openai",
        "model_id": "gpt-4o",
        "parameters": {
            "temperature": 0.1
        }
    },
    
    {
        "config_id": "o3-mini-self-discover",
        "type": "openai",
        "model_id": "o3-mini-2025-01-31",
        "parameters": {
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "o4-mini-self-discover",
        "type": "openai",
        "model_id": "o4-mini-2025-04-16",
        "parameters": {
            "response_format": {"type": "json_object"}
        }
    },
    {
        "config_id": "o3-pro-self-discover",
        "model_id": "o3-pro-2025-06-10",
        "type": "openai",
        "parameters": {
            "temperature": 0.1,
            "reasoning": {"effort": "high"}
        }
    },
    {
        "config_id": "gpt-4.1-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-mini-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "gpt-4.1-nano-self-discover",
        "type": "openai",
        "model_id": "gpt-4.1-nano-2025-04-14",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 32768
        }
    },
    
    {
        "config_id": "grok-3-mini-beta-high-effort-self-discover",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "high"
        }
    },
    {
        "config_id": "grok-3-mini-beta-low-effort-self-discover",
        "type": "xai",
        "model_id": "grok-3-mini-beta",
        "parameters": {
            "temperature": 0.1,
            "reasoning_effort": "low"
        }
    },
    {
        "config_id": "grok-3-self-discover",
        "type": "xai",
        "model_id": "grok-3-latest",
        "parameters": {
            "temperature": 0.1,
            "max_tokens": 16384
        }
    },
    {
        "config_id": "grok-4-0709-self-discover",
        "type": "xai",
        "model_id": "grok-4-0709",
        "parameters": {
            "temperature": 0
        }
    },
    
    {
    "config_id": "gemini-2.5-pro-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-pro-preview-05-06",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536,
        "thinking_budget": 24576
    }
    },
    {
    "config_id": "gemini-2.5-flash-self-discover",
    "type": "gemini",
    "model_id": "gemini-2.5-flash-preview-04-17",
    "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 65536,
        "thinking_budget": 24576
    }
    },
    {
        "config_id": "palmyra-fin-default",
        "type": "writer",
        "model_id": "palmyra-fin",
        "parameters": {
            "temperature": 0.1
        }
    },
    {
        "config_id": "deepseek-r1-self-discover",
        "type": "groq",
        "model_id": "deepseek-r1-distill-llama-70b", 
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
        "config_id": "groq-llama3.3-70b-self-discover",
        "type": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-llama3.1-8b-instant-self-discover",
        "type": "groq",
        "model_id": "llama-3.1-8b-instant",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 32768
        }
    },
    {
        "config_id": "groq-gpt-oss-20b-self-discover",
        "type": "groq",
        "model_id": "openai/gpt-oss-20b",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "groq-gpt-oss-120b-self-discover",
        "type": "groq",
        "model_id": "openai/gpt-oss-120b",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "groq-kimi-k2-self-discover",
        "type": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "groq-qwen3-32b-self-discover",
        "type": "groq",
        "model_id": "qwen/qwen3-32b",
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    },
    {
        "config_id": "gpt-5-self-discover",
        "type": "openai",
        "model_id": "gpt-5-2025-08-07",
        "parameters": {
            "reasoning": {"effort": "high"}
        }
    },
    {
        "config_id": "gpt-5-mini-self-discover",
        "type": "openai",
        "model_id": "gpt-5-mini-2025-08-07",
        "parameters": {
            "reasoning": {"effort": "high"}

        }
    },
    {
        "config_id": "gpt-5-nano-self-discover",
        "type": "openai",
        "model_id": "gpt-5-nano-2025-08-07",
        "parameters": {
            "reasoning": {"effort": "high"}

        }
    },
    {
        "config_id": "o1-mini-self-discover",
        "type": "openai",
        "model_id": "o1-mini-2024-09-12",
        "parameters": {
            "reasoning": {"effort": "high"},
            "max_tokens": 65536
        }
    },
    {
        "config_id": "o3-self-discover",
        "type": "openai",
        "model_id": "o3-2025-04-16",
        "parameters": {
            "reasoning": {"effort": "high"},
            "max_tokens": 100000

        }
    },
    {
        "config_id": "o3-pro-self-discover",
        "type": "openai",
        "model_id": "o3-pro-2025-06-10",
        "parameters": {
            "reasoning": {"effort": "high"},
            "max_tokens": 100000

        }
    }
]