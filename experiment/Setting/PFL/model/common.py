"""
Model Common Interface Module
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

# Read model information from registry (including default max_tokens)
from model.register import get_model_info

logger = logging.getLogger("compile_error_fixer")

# Simple configuration system, replacing app.config
DEFAULT_CONFIG = {
    "models": {
        "default": "gpt-4o"
    },
    "model_parameters": {
        "temperature": 0.2
    }
}

# Global variables
_current_model = None


def set_model(model_name: str):
    """Set the currently used model"""
    global _current_model
    _current_model = model_name
    logger.info(f"Model set: {model_name}")


def get_current_model() -> str:
    """Get the currently used model"""
    global _current_model
    if not _current_model:
        # If no model is set, use the default model
        _current_model = DEFAULT_CONFIG["models"]["default"]
        logger.info(f"Using default model: {_current_model}")
    return _current_model


def _get_default_max_tokens(model_name: str) -> int:
    """Get default max_tokens from model registry information"""
    try:
        info = get_model_info(model_name)
        value = int(info.get("max_tokens", 4096))
        return value
    except Exception:
        return 4096


def call_llm(messages: List[Dict], model: Optional[str] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    """
    Main interface for calling language models
    
    Users need to implement corresponding calling functions based on model type:
    - call_openai() for OpenAI models (gpt-*)
    - call_anthropic() for Anthropic models (claude-*)
    - call_gemini() for Google models (gemini-*)
    - call_deepseek() for DeepSeek models (deepseek-*)
    - call_qwen() for Qwen models (qwen*)
    - call_kimi() for Kimi models (Kimi-*)
    - call_mock_model() for testing
    
    Args:
        messages: Message list, format: [{"role": "user", "content": "..."}]
        model: Model name, if None, use currently set model
        temperature: Temperature parameter, if None, use default value
        max_tokens: Maximum token count, if None, use model default value
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When the corresponding model calling function is not implemented
    """
    try:
        model_name = model or get_current_model()
        temp = temperature if temperature is not None else DEFAULT_CONFIG["model_parameters"]["temperature"]
        effective_max_tokens = max_tokens if max_tokens is not None else _get_default_max_tokens(model_name)
        logger.info(f"call_llm using parameters: model={model_name}, temperature={temp}, max_tokens={effective_max_tokens}")
        print(f"call_llm using parameters: model={model_name}, temperature={temp}, max_tokens={effective_max_tokens}")
        
        # Check model name prefix to determine which API to use
        if model_name.startswith("mock-"):
            return call_mock_model(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("gpt-"):
            return call_openai(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("claude-"):
            return call_anthropic(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("gemini-"):
            return call_gemini(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("deepseek-"):
            return call_deepseek(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("qwen"):
            return call_qwen(messages, model_name, temp, effective_max_tokens)
        elif model_name.startswith("Kimi-"):
            return call_kimi(messages, model_name, temp, effective_max_tokens)
        else:
            logger.warning(f"Unknown model type: {model_name}, falling back to Mock model")
            return call_mock_model(messages, model_name, temp, effective_max_tokens)
    
    except Exception as e:
        logger.error(f"Error occurred when calling LLM: {e}")
        raise


def call_openai(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call OpenAI API
    
    Users need to implement this function to connect to OpenAI API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_openai() function needs to be implemented by the user. Please implement this function according to OpenAI API documentation.\n"
        "Reference: https://platform.openai.com/docs/api-reference/chat"
    )


def call_anthropic(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call Anthropic Claude API
    
    Users need to implement this function to connect to Anthropic API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_anthropic() function needs to be implemented by the user. Please implement this function according to Anthropic API documentation.\n"
        "Reference: https://docs.anthropic.com/claude/reference/getting-started-with-the-api"
    )


def call_gemini(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call Google Gemini API
    
    Users need to implement this function to connect to Google Gemini API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_gemini() function needs to be implemented by the user. Please implement this function according to Google Gemini API documentation.\n"
        "Reference: https://ai.google.dev/docs"
    )


def call_deepseek(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call DeepSeek API
    
    Users need to implement this function to connect to DeepSeek API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_deepseek() function needs to be implemented by the user. Please implement this function according to DeepSeek API documentation.\n"
        "Reference: https://platform.deepseek.com/api-docs"
    )


def call_qwen(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call Qwen API
    
    Users need to implement this function to connect to Qwen API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_qwen() function needs to be implemented by the user. Please implement this function according to Qwen API documentation.\n"
        "Reference: https://help.aliyun.com/zh/dashscope/"
    )


def call_kimi(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call Kimi API
    
    Users need to implement this function to connect to Kimi API
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Text content returned by the model
        
    Raises:
        NotImplementedError: When function is not implemented
    """
    raise NotImplementedError(
        "call_kimi() function needs to be implemented by the user. Please implement this function according to Kimi API documentation.\n"
        "Reference: https://platform.moonshot.cn/docs"
    )


def call_mock_model(messages: List[Dict], model: str, temperature: float, max_tokens: Optional[int] = None) -> str:
    """
    Call Mock model for testing
    
    Args:
        messages: Message list
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum token count
        
    Returns:
        str: Mock response content
    """
    import time
    
    # Simulate processing time
    time.sleep(0.1)
    
    # Return mock response
    return "This is a mock response for testing. Please implement the corresponding LLM calling functions to get real model responses."