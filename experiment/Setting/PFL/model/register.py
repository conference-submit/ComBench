"""
Model Registration Module
"""

import logging
import os
from typing import Dict, List

logger = logging.getLogger("compile_error_fixer")

# Supported model list
SUPPORTED_MODELS = {
    # OpenAI Models
    "gpt-5": {
        "provider": "openai",
        "description": "OpenAI GPT-5 Model",
        "max_tokens": 128000,
        "max_input_tokens": 272000
    },
    "gpt-4.1": {
        "provider": "openai",
        "description": "OpenAI GPT-4.1 Model",
        "max_tokens": 32768,
        "max_input_tokens": 1000000
    },
    "gpt-4o": {
        "provider": "openai",
        "description": "OpenAI GPT-4o Model",
        "max_tokens": 16384,
        "max_input_tokens": 128000
    },
    # Anthropic Models
    "claude-opus-4": {
        "provider": "anthropic",
        "description": "Anthropic Claude 4 Opus Model",
        "max_tokens": 8192,
        "max_input_tokens": 200000
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "description": "Anthropic Claude 4 Sonnet Model",
        "max_tokens": 16384,
        "max_input_tokens": 128000
    },
    "claude-3-7-sonnet": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.7 Sonnet Model",
        "max_tokens": 16384,
        "max_input_tokens": 128000
    },
    "claude-3-5-sonnet-v2": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Sonnet v2 Model",
        "max_tokens": 8192,
        "max_input_tokens": 200000
    },
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Haiku Model",
        "max_tokens": 4096,
        "max_input_tokens": 200000
    },
    
    # Google Models
    "gemini-2.5-pro": {
        "provider": "google",
        "description": "Google Gemini 2.5 Pro Model",
        "max_tokens": 64000,
        "max_input_tokens": 500000
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "description": "Google Gemini 1.5 Pro Model",
        "max_tokens": 8192,
        "max_input_tokens": 1000000
    },
    "gemini-1.0-pro": {
        "provider": "google",
        "description": "Google Gemini 1.0 Pro Model",
        "max_tokens": 4096,
        "max_input_tokens": 32768
    },
    
    # DeepSeek Models
    "deepseek-v3": {
        "provider": "deepseek",
        "description": "DeepSeek V3 Model",
        "max_tokens": 8192,
        "max_input_tokens": 128000
    },
    
    # Qwen Models
    "qwen3-235b-a22b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode, 22B activated parameters",
        "max_tokens": 8192,
        "max_input_tokens": 55808,  # 64000 - 8192
        "model_id": "8842"
    },
    "qwen3-30b-a3b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode, only 3B activated parameters, suitable for high-throughput scenarios",
        "max_tokens": 8192,
        "max_input_tokens": 32768,  # 40960 - 8192
        "model_id": "8905"
    },
    "qwen3-32b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode",
        "max_tokens": 8192,
        "max_input_tokens": 32768,  # 40960 - 8192
        "model_id": "8907"
    },
    "qwen3-14b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode",
        "max_tokens": 8192,
        "max_input_tokens": 32768,  # 40960 - 8192
        "model_id": "8913"
    },
    "qwen3-8b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode",
        "max_tokens": 8192,
        "max_input_tokens": 32768,  # 40960 - 8192
        "model_id": "8908"
    },
    "qwen3-4b": {
        "provider": "qwen",
        "description": "(Reasoning Model) Alibaba's Qwen 3rd Generation Model, supports reasoning-non-reasoning hybrid mode",
        "max_tokens": 8192,
        "max_input_tokens": 32768,  # 40960 - 8192
        "model_id": "8906"
    },
    
    # Kimi Models
    "Kimi-K2": {
        "provider": "kimi",
        "description": "Kimi K2 Model",
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "model_id": "10799"
    }
}

# Available model list
available_models = []


def register_all_models() -> List[str]:
    """Register all supported models"""
    global available_models
    available_models = []
    
    for model_name, model_info in SUPPORTED_MODELS.items():
        available_models.append(model_name)
        logger.debug(f"Registered model: {model_name} ({model_info['description']})")
    
    # If no available models, use mock model
    if not available_models:
        logger.warning("No available models found, will use mock model")
        available_models.append("mock-model")
    
    logger.info(f"Registered {len(available_models)} available models: {', '.join(available_models)}")
    return available_models


def get_available_models() -> List[str]:
    """Get the list of available models"""
    if not available_models:
        register_all_models()
    return available_models


def get_model_info(model_name: str) -> Dict:
    """Get model information"""
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    return {
        "provider": "unknown",
        "description": "Unknown model",
        "max_tokens": 4096
    }