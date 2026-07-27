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
