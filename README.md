# Agent AI Frameworks Tutorial

This repository includes tutorials on Agent AI frameworks. 

## Chatbot Folder
In the `chatbot/` folder, an LLM is fine-tuned with particle physics papers (CERN publications, arXiv preprints, and high-energy physics research). The model is hosted on a UI that allows clients to chat, ask questions about physics concepts, and query research findings.

**Tech Stack:** Fine-tuned LLaMA/Mistral model, RAG with vector database, FastAPI backend, Streamlit/React frontend.

![Chat Interface](https://github.com/user-attachments/assets/3610a14f-1c79-47ef-bfeb-18e046cbb2df)

## Setup

```bash
git clone https://github.com/yourusername/agent-ai-tutorials.git
cd agent-ai-tutorials/chatbot
pip install -r requirements.txt
python app.py
```

## Agent with MCP
In the `Agentic_AI_tutorial/` folder, The agent connects an LLM to multiple MCP servers, enabling capabilities such as searching arXiv research papers, retrieving paper information, accessing local files, and fetching external resources. MCP provides a standardized interface between the LLM and these tools, making the architecture modular and easy to extend with new services.

<img width="527" height="851" alt="Screenshot 2026-08-07 082442" src="https://github.com/user-attachments/assets/24e035e7-6683-481f-8525-57f5e9a5e77d" />

