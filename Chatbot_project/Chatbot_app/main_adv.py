from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import tiktoken
import langchain_helper


# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

# Global LLM instance
llm = None

def get_llm(model="gapgpt-qwen-3.5", temperature=0.7, max_tokens=500):
    """Get or create LLM instance with given parameters"""
    global llm
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to simple estimation
        return len(text) // 4

def get_llm_response(user_prompt, system_prompt=None, temperature=0.7, max_tokens=500):
    """
    Get response from LLM based on user prompt.
    
    Args:
        user_prompt (str): The user's question or prompt
        system_prompt (str): Optional system instruction for the AI
        temperature (float): Creativity level (0.0 to 1.0)
        max_tokens (int): Maximum tokens in response
    
    Returns:
        str: The LLM's response
    """
    
    # Get LLM instance
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    
    # Create prompt template with messages
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append(("system", system_prompt))
    
    # Add user prompt
    messages.append(("user", "{user_prompt}"))
    
    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(messages)
    
    # Create the chain
    chain = prompt_template | llm | StrOutputParser()
    
    # Invoke the chain
    response = chain.invoke({"user_prompt": user_prompt})
    
    return response

def get_structured_response(user_prompt, system_prompt=None, temperature=0.7, max_tokens=500):
    """
    Get structured response with metadata.
    
    Returns:
        dict: Contains response, token counts, and metadata
    """
    
    response = get_llm_response(user_prompt, system_prompt, temperature, max_tokens)
    
    # Count tokens
    input_tokens = count_tokens(user_prompt)
    output_tokens = count_tokens(response)
    
    return {
        "response": response,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }

if __name__ == "__main__":
    # Test the helper
    test_response = get_llm_response("What is the capital of France?")
    print("Test Response:", test_response)