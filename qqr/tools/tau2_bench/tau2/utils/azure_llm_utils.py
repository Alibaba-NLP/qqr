"""
Azure OpenAI utilities for tau2-bench

This module provides support for both Azure OpenAI and standard OpenAI APIs.
Configuration is controlled via environment variables:

- USE_AZURE_OPENAI: Set to "true" to use Azure OpenAI (default: "false", uses standard OpenAI)
- AZURE_OPENAI_MODELS: Comma-separated list of models to route through Azure (only when USE_AZURE_OPENAI=true)
- OPENAI_API_KEY: Standard OpenAI API key (used when USE_AZURE_OPENAI=false)
- For Azure: Uses Azure CLI credential or can be configured with AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
"""
import os
import json
import time
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from loguru import logger

# API Provider Configuration
# Set USE_AZURE_OPENAI=true to use Azure OpenAI, otherwise uses standard OpenAI
USE_AZURE_OPENAI = os.environ.get("USE_AZURE_OPENAI", "false").lower() == "true"

# Define models that need to use Azure endpoints (only relevant when USE_AZURE_OPENAI=true)
# Can be customized via AZURE_OPENAI_MODELS environment variable (comma-separated)
DEFAULT_AZURE_MODELS = []
_azure_models_env = os.environ.get("AZURE_OPENAI_MODELS", "")
AZURE_MODELS = [m.strip() for m in _azure_models_env.split(",") if m.strip()] if _azure_models_env else DEFAULT_AZURE_MODELS

# Log the current API configuration at startup
if USE_AZURE_OPENAI:
    logger.info(f"API Provider: Azure OpenAI (Models: {AZURE_MODELS})")
else:
    logger.info("API Provider: Standard OpenAI")

# Lazy import Azure dependencies only when needed
_azure_client_module = None
def _get_azure_dependencies():
    """Lazily import Azure dependencies to avoid import errors when not using Azure"""
    global _azure_client_module
    if _azure_client_module is None:
        try:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import AzureOpenAI
            _azure_client_module = {
                "AzureCliCredential": AzureCliCredential,
                "get_bearer_token_provider": get_bearer_token_provider,
                "AzureOpenAI": AzureOpenAI
            }
        except ImportError as e:
            raise ImportError(
                f"Azure OpenAI dependencies not installed. "
                f"Please install them with: pip install azure-identity openai\n"
                f"Original error: {e}"
            )
    return _azure_client_module

# Logging configuration
LLM_LOG_DIR = os.environ.get("LLM_LOG_DIR", "logs/llm_requests")
LLM_LOG_ENABLED = os.environ.get("LLM_LOG_ENABLED", "true").lower() == "true"

def setup_llm_logging():
    """Set up LLM request log directory"""
    if LLM_LOG_ENABLED:
        log_dir = Path(LLM_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LLM request logging enabled. Logs will be saved to: {log_dir}")
    else:
        logger.info("LLM request logging disabled")

# Initialize logging
setup_llm_logging()

def save_llm_interaction(model: str, request_data: dict, response_data: dict, 
                        cost: float = None, error: str = None):
    """
    Save complete LLM interaction records
    
    Args:
        model: Model name
        request_data: Request data
        response_data: Response data  
        cost: Request cost
        error: Error message (if any)
    """
    if not LLM_LOG_ENABLED:
        return
        
    try:
        timestamp = datetime.now().isoformat()
        interaction_id = f"{timestamp}_{hash(str(request_data))}"
        
        log_entry = {
            "interaction_id": interaction_id,
            "timestamp": timestamp,
            "model": model,
            "request": request_data,
            "response": response_data,
            "cost": cost,
            "error": error,
            "usage": response_data.get("usage") if response_data else None
        }
        
        # Save to date directory
        log_dir = Path(LLM_LOG_DIR)
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_safe = model.replace("/", "_").replace(":", "_")
        daily_log_dir = log_dir / date_str / model_safe
        daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save by model files
        log_file = daily_log_dir / f"{datetime.now().isoformat()}_{model_safe}_requests.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
        logger.debug(f"Saved LLM interaction to: {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to save LLM interaction log: {e}")

def log_request_summary(model: str, messages: List[dict], tools: Optional[List[dict]], **kwargs):
    """Log request summary information"""
    if not LLM_LOG_ENABLED:
        return
        
    try:
        message_count = len(messages)
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        tool_count = len(tools) if tools else 0
        
        logger.info(f"LLM Request - Model: {model}, Messages: {message_count}, "
                   f"Chars: {total_chars}, Tools: {tool_count}")
    except Exception as e:
        logger.error(f"Failed to log request summary: {e}")

# Azure OpenAI configuration from environment variables
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

def should_use_azure(model: str) -> bool:
    """Check if model should use Azure OpenAI endpoint
    
    Returns True only when:
    1. USE_AZURE_OPENAI environment variable is set to "true"
    2. AND the model is in the AZURE_MODELS list
    """
    if not USE_AZURE_OPENAI:
        return False
    return model in AZURE_MODELS

def get_azure_config(model: str) -> dict:
    """Get Azure configuration from environment variables"""
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set")
    if not AZURE_OPENAI_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT environment variable is not set")
    
    return {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_API_VERSION,
        "deployment": AZURE_OPENAI_DEPLOYMENT
    }

def get_azure_client(model: str):
    """Get Azure OpenAI client"""
    azure_deps = _get_azure_dependencies()
    AzureOpenAI = azure_deps["AzureOpenAI"]
    AzureCliCredential = azure_deps["AzureCliCredential"]
    get_bearer_token_provider = azure_deps["get_bearer_token_provider"]
    
    config = get_azure_config(model)
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    
    return AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=config["endpoint"],
        api_version=config["api_version"]
    )

def get_deployment_name(model: str) -> str:
    """Get Azure deployment name for the model"""
    config = get_azure_config(model)
    return config["deployment"]

class AzureResponseWrapper:
    """
    Wrap Azure OpenAI response to be compatible with litellm's ModelResponse format
    """
    def __init__(self, azure_response, original_model: str):
        self.azure_response = azure_response
        self.original_model = original_model
        self._model = original_model  # Keep original model name for cost calculation
        
        # Copy key attributes to current object
        self.id = azure_response.id
        self.object = azure_response.object
        self.created = azure_response.created
        self.choices = azure_response.choices
        self.usage = azure_response.usage
        self.system_fingerprint = getattr(azure_response, 'system_fingerprint', None)
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        
    def get(self, key, default=None):
        if key == "model":
            return self._model
        elif key == "usage":
            return self.usage
        return getattr(self, key, default)
    
    def __getattr__(self, name):
        """Proxy other attribute access to original response"""
        if name == "model":
            return self._model
        return getattr(self.azure_response, name)

def azure_completion(model: str, messages: List[dict], tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, **kwargs):
    """
    Make completion call using Azure OpenAI
    """
    request_start_time = time.time()
    request_data = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "kwargs": kwargs
    }
    
    # Log request summary
    log_request_summary(model, messages, tools, **kwargs)
    
    try:
        client = get_azure_client(model)
        deployment_name = get_deployment_name(model)
        config = get_azure_config(model)
        
        # Build call parameters
        call_params = {
            "model": deployment_name,
            "messages": messages,
        }
        
        # Filter unsupported parameters and convert parameter names
        supported_params = [
            "temperature", "max_completion_tokens", "max_tokens", "top_p", "frequency_penalty", 
            "presence_penalty", "stream", "stop", "seed"
        ]
        
        for param in supported_params:
            if param in kwargs:
                # For some Azure models, need to convert max_tokens to max_completion_tokens
                if param == "max_tokens" and "max_tokens" in kwargs:
                    call_params["max_completion_tokens"] = kwargs["max_tokens"]
                else:
                    call_params[param] = kwargs[param]
        
        # Add tools related parameters
        if tools:
            call_params["tools"] = tools
        if tool_choice:
            call_params["tool_choice"] = tool_choice
            
        logger.info(f"Using Azure OpenAI for model {model} with deployment {deployment_name} at {config['endpoint']}")
        
        # Call Azure OpenAI
        response = client.chat.completions.create(**call_params)
        
        # Calculate response time
        response_time = time.time() - request_start_time
        
        # Wrap response to be compatible with litellm format
        wrapped_response = AzureResponseWrapper(response, model)
        
        # Prepare response data for logging
        response_data = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": model,
            "usage": response.usage.model_dump() if response.usage else None,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        **({"tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in choice.message.tool_calls
                        ]} if choice.message.tool_calls else {})
                    },
                    "finish_reason": choice.finish_reason
                } for choice in response.choices
            ],
            "system_fingerprint": getattr(response, 'system_fingerprint', None),
            "response_time_seconds": response_time
        }
        
        cost = 0.0
        if response.usage:
            pass
            

        save_llm_interaction(
            model=model,
            request_data=request_data,
            response_data=response_data,
            cost=cost
        )
        
        logger.info(f"Azure OpenAI request completed in {response_time:.2f}s. "
                   f"Usage: {response.usage.model_dump() if response.usage else 'N/A'}")
        
        return wrapped_response
        
    except Exception as e:
        response_time = time.time() - request_start_time
        error_msg = str(e)
        
        logger.error(f"Azure OpenAI call failed after {response_time:.2f}s: {error_msg}")
        

        save_llm_interaction(
            model=model,
            request_data=request_data,
            response_data=None,
            cost=0.0,
            error=error_msg
        )
        
        raise e


def get_log_stats(days: int = 1, model: str = None) -> dict:

    if not LLM_LOG_ENABLED:
        return {"error": "Logging is disabled"}
        
    stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_cost": 0.0,
        "models": {},
        "date_range": []
    }
    
    try:
        log_dir = Path(LLM_LOG_DIR)
        
        # Get dates from recent days
        from datetime import datetime, timedelta
        dates = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        
        stats["date_range"] = dates
        
        for date_str in dates:
            daily_log_dir = log_dir / date_str
            if not daily_log_dir.exists():
                continue
                
            for log_file in daily_log_dir.glob("*.jsonl"):
                file_model = log_file.stem.replace("_requests", "").replace("_", "/")
                
                if model and model != file_model:
                    continue
                
                if file_model not in stats["models"]:
                    stats["models"][file_model] = {
                        "requests": 0,
                        "successful": 0,
                        "failed": 0,
                        "cost": 0.0,
                        "total_tokens": 0
                    }
                
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            stats["total_requests"] += 1
                            stats["models"][file_model]["requests"] += 1
                            
                            if entry.get("error"):
                                stats["failed_requests"] += 1
                                stats["models"][file_model]["failed"] += 1
                            else:
                                stats["successful_requests"] += 1
                                stats["models"][file_model]["successful"] += 1
                                
                            if entry.get("cost"):
                                stats["total_cost"] += entry["cost"]
                                stats["models"][file_model]["cost"] += entry["cost"]
                                
                            if entry.get("usage"):
                                usage = entry["usage"]
                                total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                                stats["models"][file_model]["total_tokens"] += total_tokens
                                
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        stats["error"] = str(e)
        
    return stats

def export_logs_to_csv(output_file: str, days: int = 1, model: str = None):

    if not LLM_LOG_ENABLED:
        logger.error("Logging is disabled")
        return
        
    try:
        import csv
        from datetime import datetime, timedelta
        
        log_dir = Path(LLM_LOG_DIR)
        
        # Get dates from recent days
        dates = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            dates.append(date.strftime("%Y-%m-%d"))
        
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "timestamp", "model", "interaction_id", "request_messages_count",
                "response_content_length", "total_tokens", "prompt_tokens", 
                "completion_tokens", "cost", "response_time", "error"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for date_str in dates:
                daily_log_dir = log_dir / date_str
                if not daily_log_dir.exists():
                    continue
                    
                for log_file in daily_log_dir.glob("*.jsonl"):
                    file_model = log_file.stem.replace("_requests", "").replace("_", "/")
                    
                    if model and model != file_model:
                        continue
                    
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                
                                usage = entry.get("usage", {})
                                response = entry.get("response", {})
                                request = entry.get("request", {})
                                
                                row = {
                                    "timestamp": entry.get("timestamp"),
                                    "model": entry.get("model"),
                                    "interaction_id": entry.get("interaction_id"),
                                    "request_messages_count": len(request.get("messages", [])),
                                    "response_content_length": len(str(response)) if response else 0,
                                    "total_tokens": usage.get("total_tokens", 0),
                                    "prompt_tokens": usage.get("prompt_tokens", 0),
                                    "completion_tokens": usage.get("completion_tokens", 0),
                                    "cost": entry.get("cost", 0),
                                    "response_time": response.get("response_time_seconds", 0),
                                    "error": entry.get("error", "")
                                }
                                
                                writer.writerow(row)
                                
                            except json.JSONDecodeError:
                                continue
                                
        logger.info(f"Logs exported to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export logs to CSV: {e}")

def cleanup_old_logs(days_to_keep: int = 30):
    if not LLM_LOG_ENABLED:
        return
        
    try:
        from datetime import datetime, timedelta
        
        log_dir = Path(LLM_LOG_DIR)
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = 0
        for date_dir in log_dir.iterdir():
            if date_dir.is_dir():
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        import shutil
                        shutil.rmtree(date_dir)
                        deleted_count += 1
                        logger.info(f"Deleted old log directory: {date_dir}")
                except ValueError:

                    continue
                    
        logger.info(f"Cleanup completed. Deleted {deleted_count} old log directories.")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")
