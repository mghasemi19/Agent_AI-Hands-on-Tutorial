import streamlit as st
import langchain_helper
import time
import uuid

# Page configuration
st.set_page_config(
    page_title="LLM Chat Interface with Memory",
    page_icon="",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stSelectbox > div > div > select {
        font-size: 16px;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff4b4b;
    }
    .user-box {
        background-color: #e8f0fe;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4285f4;
    }
    .stAlert {
        margin-top: 10px;
    }
    .memory-indicator {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .memory-indicator.off {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .model-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .model-badge.gpt4 {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .model-badge.gpt35 {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .model-badge.qwen {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_time" not in st.session_state:
    st.session_state.response_time = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0
if "max_history" not in st.session_state:
    st.session_state.max_history = 5
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Qwen 3.5"

# Title and description
st.title("LLM Chat Interface with Memory")
st.markdown("""
    <p style='font-size: 18px; color: #666;'>
        Ask any question about <b>particle physics</b> and get responses from different LLM models.
        The chatbot remembers the last messages in the conversation!
    </p>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Selection")
    
    # Get available models from helper
    available_models = list(langchain_helper.AVAILABLE_MODELS.keys())
    
    selected_model = st.selectbox(
        "Choose a Model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        help="Select the LLM model to use for responses"
    )
    
    # Update session state
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
    
    # Show model info
    model_info = {
        "GPT-4 Turbo": {
            "icon": "🚀",
            "desc": "Most capable, best for complex tasks",
            "class": "gpt4"
        },
        "GPT-3.5 Turbo": {
            "icon": "⚡",
            "desc": "Fast, cost-effective, good for most tasks",
            "class": "gpt35"
        },
        "Qwen 3.5": {
            "icon": "🐉",
            "desc": "Custom model, optimized for specific tasks",
            "class": "qwen"
        }
    }
    
    if selected_model in model_info:
        info = model_info[selected_model]
        st.info(f"{info['icon']} **{selected_model}**\n\n{info['desc']}")
    
    st.divider()
    
    # Memory settings
    st.subheader("🧠 Memory Settings")
    
    max_history = st.slider(
        "Messages to Remember",
        min_value=1,
        max_value=20,
        value=st.session_state.max_history,
        step=1,
        help="Number of recent messages the chatbot will remember"
    )
    
    if max_history != st.session_state.max_history:
        st.session_state.max_history = max_history
    
    # Session info
    st.info(f"🔑 Session ID: `{st.session_state.session_id[:8]}...`")
    
    # Memory status
    memory_status = langchain_helper.get_memory_status(st.session_state.session_id)
    if memory_status["message_count"] > 0:
        st.markdown(f"""
            <div class="memory-indicator">
                ✅ Memory Active<br>
                <span style="font-size: 0.9em; color: #666;">
                    Storing <strong>{memory_status['message_count']}</strong> messages 
                    (max: {st.session_state.max_history})
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="memory-indicator off">
                ❌ No Memory<br>
                <span style="font-size: 0.9em; color: #666;">
                    Start a conversation to build memory
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    # Show recent messages from memory
    if memory_status["message_count"] > 0:
        with st.expander("📝 Recent Memory", expanded=False):
            for msg in memory_status["messages"][-6:]:
                role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
                content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                st.text(f"{role}: {content}")
    
    # Clear memory button
    if st.button("🗑️ Clear Memory", use_container_width=True):
        langchain_helper.clear_memory(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.success("✅ Memory cleared!")
        st.rerun()
    
    # New session button
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.conversation_count = 0
        st.success("✅ New session started!")
        st.rerun()
    
    st.divider()
    
    # Temperature slider
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more deterministic"
    )
    
    # System prompt input
    st.subheader("💬 System Prompt (Optional)")
    system_prompt = st.text_area(
        "Instructions for the AI",
        placeholder="E.g., You are a helpful assistant that speaks like a pirate.",
        height=100,
        help="Set the AI's role or behavior"
    )
    
    # Quick preset prompts
    st.subheader("⚡ Quick Presets")
    presets = {
        "Default Assistant": "",
        "Pirate Mode": "You are a helpful assistant that speaks like a pirate. Use pirate vocabulary and be enthusiastic!",
        "Formal Mode": "You are a formal, professional assistant. Use proper language and be concise.",
        "Creative Writer": "You are a creative writer. Write engaging, descriptive, and imaginative responses.",
        "Teacher": "You are a patient teacher. Explain concepts clearly and provide examples.",
        "Tech Expert": "You are a technical expert. Provide detailed, accurate technical explanations."
    }
    
    selected_preset = st.selectbox("Choose a preset", list(presets.keys()))
    if selected_preset != "Default Assistant" and st.button("Apply Preset"):
        system_prompt = presets[selected_preset]
        st.rerun()
    
    st.divider()
    st.caption("🚀 Powered by LangChain + OpenAI")

# Display chat history
st.subheader("💬 Chat History")

# Model indicator in chat header
if st.session_state.messages:
    memory_status = langchain_helper.get_memory_status(st.session_state.session_id)
    
    # Show model and memory info
    model_display = selected_model
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"🧠 Remembering last {min(len(st.session_state.messages), st.session_state.max_history)} messages")
    with col2:
        st.caption(f"🤖 Model: {model_display}")

# Container for chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="user-box">
                    <strong>🧑 You:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Add model badge to assistant messages
            model_badge = ""
            if "model" in message:
                model_name = message["model"]
                model_class = "gpt4" if "GPT-4" in model_name else "gpt35" if "GPT-3.5" in model_name else "qwen"
                model_badge = f'<span class="model-badge {model_class}">{model_name}</span>'
            
            st.markdown(f"""
                <div class="response-box">
                    <strong>🤖 Assistant</strong> {model_badge}<br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)

# Input area at the bottom
st.divider()

# Create columns for input and button
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Type your message here...",
        placeholder=f"Ask me anything! Using {selected_model}",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("🚀 Send", type="primary", use_container_width=True)

# Process the input when send button is clicked or Enter is pressed
if send_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.conversation_count += 1
    
    # Get response from LLM with memory
    with st.spinner(f"🤔 Thinking with {selected_model}..."):
        start_time = time.time()
        
        try:
            response = langchain_helper.get_llm_response(
                user_prompt=user_input,
                system_prompt=system_prompt if system_prompt else None,
                temperature=temperature,
                session_id=st.session_state.session_id,
                max_history=st.session_state.max_history,
                model_name=selected_model
            )
            
            end_time = time.time()
            st.session_state.response_time = end_time - start_time
            
            # Add assistant response to chat history with model info
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "model": selected_model
            })
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error with {selected_model}: {str(e)}",
                "model": selected_model
            })
    
    # Rerun to update the display
    st.rerun()

# Clear input after sending (using session state trick)
if send_button:
    st.session_state.user_input = ""

# Footer with response stats
if st.session_state.messages:
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💬 Total Messages", len(st.session_state.messages))
    with col2:
        st.metric("🔄 Turns", st.session_state.conversation_count)
    with col3:
        st.metric("🌡️ Temperature", temperature)
    with col4:
        memory_status = langchain_helper.get_memory_status(st.session_state.session_id)
        st.metric("🧠 Memory Size", memory_status["message_count"])
    with col5:
        st.metric("🤖 Model", selected_model[:10] + "..." if len(selected_model) > 10 else selected_model)

# Example prompts in the footer
st.divider()
st.caption("💡 Try asking: 'What is the meaning of life?' | 'Explain quantum computing' | 'Write a poem about coding'")
st.caption(f"🔄 Current model: **{selected_model}** • Switch models anytime in the sidebar")