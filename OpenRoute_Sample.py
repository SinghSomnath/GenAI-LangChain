import requests
import json
import time
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str, app_name: str = "ModelRouter", app_url: str = ""):
        """
        Initialize OpenRouter client with API key and app details
        
        Args:
            api_key: Your OpenRouter API key
            app_name: Your application name
            app_url: Your application URL (optional)
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,
            "X-Title": app_name
        }
        
        # Define model routing order (primary to fallback)
        self.model_routing_order = [
            "openai/gpt-5-codex",
            "openai/gpt-4",
            "openai/gpt-4o-audio-preview",
            "openai/gpt-5-chat",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "meta-llama/llama-2-70b-chat",
            "mistralai/mixtral-8x7b-instruct"
        ]
        
        # CHANGE: Added popular_models attribute that was missing
        self.popular_models = [
            "openai/gpt-4-turbo",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-sonnet",
            "anthropic/claude-3-haiku",
            "google/gemini-pro",
            "meta-llama/llama-2-70b-chat",
            "mistralai/mixtral-8x7b-instruct",
            "openai/gpt-4o",
            "anthropic/claude-3-opus"
        ]
    
    def make_request(self, model: str, messages: List[Dict], **kwargs) -> Optional[Dict]:
        # """
        # Make a request to a specific model via OpenRouter
        
        # Args:
        #     model: Model identifier
        #     messages: List of message dictionaries
        #     **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        # Returns:
        #     Response dictionary or None if failed
        # """
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        try:
            logger.info(f"Attempting request to model: {model}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"Success with model: {model}")
                return response.json()
            else:
                logger.warning(f"Failed with model {model}: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout with model: {model}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception with model {model}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with model {model}: {str(e)}")
            return None
    
    def route_with_fallback(self, 
                           messages: List[Dict], 
                           custom_model_order: Optional[List[str]] = None,
                           **kwargs) -> Dict:
        # """
        # Route request through models with fallback logic
        
        # Args:
        #     messages: List of message dictionaries
        #     custom_model_order: Optional custom model routing order
        #     **kwargs: Additional parameters for the API call
            
        # Returns:
        #     Response dictionary with success status and data
        # """
        model_order = custom_model_order or self.model_routing_order
        
        for i, model in enumerate(model_order):
            response = self.make_request(model, messages, **kwargs)
            
            if response:
                return {
                    "success": True,
                    "model_used": model,
                    "attempt_number": i + 1,
                    "response": response
                }
            
            # Add delay between attempts to avoid hitting rate limits
            if i < len(model_order) - 1:
                time.sleep(1)
        
        return {
            "success": False,
            "model_used": None,
            "attempt_number": len(model_order),
            "error": "All models failed to respond"
        }
    
    def get_available_models(self) -> List[Dict]:
        """
        Get list of available models from OpenRouter
        
        Returns:
            List of available models
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []
    
    def get_top_popular_models(self, limit: int = 10) -> List[Dict]:
        """
        Get top popular models from the available models list
        
        Args:
            limit: Number of top models to return (default: 5)
            
        Returns:
            List of top popular models with their details
        """
        all_models = self.get_available_models()
        
        if not all_models:
            logger.warning("No models available from API")
            return []
        
        # Create a mapping of model IDs for quick lookup
        model_map = {model.get("id"): model for model in all_models}
        
        # Filter popular models that are available
        popular_available = []
        for popular_model in self.popular_models:
            if popular_model in model_map:
                popular_available.append(model_map[popular_model])
                if len(popular_available) >= limit:
                    break
        
        # If we don't have enough popular models, fill with first available models
        if len(popular_available) < limit:
            remaining_needed = limit - len(popular_available)
            popular_ids = {model.get("id") for model in popular_available}
            
            for model in all_models:
                if model.get("id") not in popular_ids:
                    popular_available.append(model)
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
        
        logger.info(f"Found {len(popular_available)} popular models")
        return popular_available

def main():
    """
    Example usage of the OpenRouter client with model routing
    """
    
    # Initialize client (replace with your actual API key)
    api_key = "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"
    client = OpenRouterClient(
        api_key=api_key,
        app_name="MyApp",
        app_url="https://myapp.com"
    )
    
    # Example conversation
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ]
    
    # Make request with fallback routing
    result = client.route_with_fallback(
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

####### Parameter Details ####### 
# Temperature (0.7):

# Range: 0.0 to 2.0
# 0.0 = deterministic, focused responses
# 0.7 = balanced creativity and coherence (good default)
# 2.0 = highly random, creative responses
# Max Tokens (500):

# Limits the response length to 500 tokens
# 1 token ≈ 0.75 words in English
# 500 tokens ≈ 375 words approximately
    
    if result["success"]:
        print(f"✅ Success with {result['model_used']} (attempt #{result['attempt_number']})")
        response_content = result["response"]["choices"][0]["message"]["content"]
        print(f"\nResponse:\n{response_content}")
        
        # Print usage info if available
        if "usage" in result["response"]:
            usage = result["response"]["usage"]
            print(f"\nTokens used: {usage.get('total_tokens', 'N/A')}")
            
    else:
        print(f"❌ All models failed after {result['attempt_number']} attempts")
        print(f"Error: {result['error']}")

# Example with custom model prioritization
def example_custom_routing():
    """
    Example with custom model routing order
    """
    api_key = "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"
    client = OpenRouterClient(api_key=api_key)
    
    # Custom model order prioritizing Anthropic models
    custom_order = [
        "openai/gpt-4-turb",
        "anthropic/claude-3-sonnet", 
        "anthropic/claude-3-haiku",
        "openai/gpt-3.5-turbo"
    ]
    
    messages = [
        {"role": "user", "content": "Write a short poem about AI"}
    ]
    
    result = client.route_with_fallback(
        messages=messages,
        custom_model_order=custom_order,
        temperature=0.9,
        max_tokens=200
    )
    
    if result["success"]:
        print(f"Poem generated by {result['model_used']}:")
        print(result["response"]["choices"][0]["message"]["content"])

# Example of checking available models
def list_available_models():
    """
    Example of listing available models
    """
    api_key = "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"
    client = OpenRouterClient(api_key=api_key)
    
    models = client.get_available_models()
    
    
    
    print("Available OpenAI models:")
    
    # openai_models = [m for m in models if "openai" in m.get("id", "").lower()]
    # for model in openai_models[:5]:  # Show first 5
    #     # print(f"  - {model.get('id')}: {model.get('name', 'N/A')}")
    #     print(f"  - {model.get('id')}")
    #     print("Available OpenAI models:")
    
    top_models = client.get_top_popular_models(limit=10)
    
    print("Top 5 Popular Models:")
    for i, model in enumerate(top_models, 1):
        model_id = model.get('id', 'Unknown')
        model_name = model.get('name', 'N/A')
        # print(f"  {i}. {model_id}: {model_name}")
        print(f"\"{model_name}\": \"{model_id}\",")

if __name__ == "__main__":
    # Run main example
    # main()
    
    # print("\n" + "="*50 + "\n")
    
    # # Run custom routing example
    # example_custom_routing()
    
    # print("\n" + "="*50 + "\n")
    
    # List available models
    list_available_models()