from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

# Available models
AVAILABLE_MODELS = {
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "Qwen 3.5": "gapgpt-qwen-3.5",
}

# Default model
DEFAULT_MODEL = "Qwen 3.5"

# Initialize the LLM with default model
llm = None

def get_llm_instance(model_name=DEFAULT_MODEL, temperature=0.7, max_tokens=500):
    """
    Get or create an LLM instance with the specified model.
    
    Args:
        model_name (str): The name of the model to use
        temperature (float): Creativity level (0.0 to 1.0)
        max_tokens (int): Maximum tokens in response
    
    Returns:
        ChatOpenAI: The LLM instance
    """
    # Get the actual model identifier
    model_identifier = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS[DEFAULT_MODEL])
    
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_identifier,
        temperature=temperature,
        max_tokens=max_tokens
    )

# Memory store - stores conversation history per session
memory_store = {}

def get_session_history(session_id: str):
    """Get or create conversation history for a session"""
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]

def get_llm_response(
    user_prompt, 
    system_prompt=None, 
    temperature=0.7, 
    session_id="default", 
    max_history=5,
    model_name=DEFAULT_MODEL
):
    """
    Get response from LLM based on user prompt with memory.
    
    Args:
        user_prompt (str): The user's question or prompt
        system_prompt (str): Optional system instruction for the AI
        temperature (float): Creativity level (0.0 to 1.0)
        session_id (str): Unique session identifier for memory
        max_history (int): Maximum number of messages to remember (default: 5)
        model_name (str): The model to use (e.g., "GPT-4 Turbo", "Qwen 3.5")
    
    Returns:
        str: The LLM's response
    """
    
    # Get the LLM instance with the specified model
    llm_instance = get_llm_instance(model_name, temperature)
    
    # Get conversation history for this session
    history = get_session_history(session_id)
    
    # Keep only last max_history messages (each message is a dict with role and content)
    if len(history) > max_history:
        history = history[-max_history:]
        memory_store[session_id] = history
    
    # Build messages for the prompt
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append(("system", system_prompt))
    else:
        messages.append(("system", "You are a helpful AI assistant. Be friendly and provide detailed, accurate responses."))
    
    # Add conversation history as placeholder
    if history:
        messages.append(MessagesPlaceholder(variable_name="chat_history"))
    
    # Add current user prompt
    messages.append(("user", "{user_prompt}"))
    
    # Create the chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(messages)
    
    # Create the chain
    chain = prompt_template | llm_instance | StrOutputParser()
    
    # Prepare input data
    input_data = {"user_prompt": user_prompt}
    
    # Add chat history if available
    if history:
        # Convert history to LangChain message format
        chat_history = []
        for msg in history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        input_data["chat_history"] = chat_history
    
    # Invoke the chain
    response = chain.invoke(input_data)
    
    # Save to memory (user message and assistant response)
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": response})
    
    # Keep only last max_history messages
    if len(history) > max_history:
        memory_store[session_id] = history[-max_history:]
    
    return response

def clear_memory(session_id="default"):
    """Clear conversation memory for a session"""
    if session_id in memory_store:
        memory_store[session_id] = []
        return True
    return False

def get_memory_status(session_id="default"):
    """Get current memory status for a session"""
    history = get_session_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "messages": history
    }


def get_structured_response(
    user_prompt, 
    system_prompt=None, 
    temperature=0.7, 
    session_id="default", 
    max_history=5,
    model_name=DEFAULT_MODEL
):
    """
    Get structured response with additional metadata.
    
    Returns:
        dict: Contains response, tokens, and metadata
    """
    
    response = get_llm_response(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        session_id=session_id,
        max_history=max_history,
        model_name=model_name
    )
    
    return {
        "response": response,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "session_id": session_id,
        "memory_size": len(get_session_history(session_id)),
        "model": model_name
    }

if __name__ == "__main__":
    # Test the helper with different models
    print("Testing Chatbot with Model Selection...")
    
    # Test with Qwen
    print("\n--- Testing Qwen 3.5 ---")
    response1 = get_llm_response(
        "What is the capital of France?", 
        model_name="Qwen 3.5"
    )
    print(f"Response: {response1}")
    
    # Test with GPT-3.5
    print("\n--- Testing GPT-3.5 Turbo ---")
    response2 = get_llm_response(
        "What is the capital of France?", 
        model_name="GPT-3.5 Turbo"
    )
    print(f"Response: {response2}")
    
    # Test memory with model
    print("\n--- Testing Memory with Model ---")
    response3 = get_llm_response(
        "My name is John.", 
        model_name="Qwen 3.5",
        session_id="test_session"
    )
    print(f"Response: {response3}")
    
    response4 = get_llm_response(
        "What is my name?", 
        model_name="Qwen 3.5",
        session_id="test_session"
    )
    print(f"Response: {response4}")