# MCP Research Assistant — Web Interface

## 1. Put these files in one folder

- `new_mcp_chatbot.py`
- `research_server.py`
- `server_config.json`
- `web_app.py`
- `index.html`
- `requirements.txt`

## 2. Create `.env`

```env
OPENAI_API_KEY=your_api_key
# Optional for OpenAI-compatible providers:
BASE_URL=https://your-provider.example/v1
MODEL_NAME=gpt-5.1
```

## 3. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The filesystem MCP server also requires Node.js because the configuration starts it with `npx`.

## 4. Run the web application

```bash
uvicorn web_app:app --reload
```

Open `http://127.0.0.1:8000`.

## Notes

- Conversation history is stored in memory and resets when the server restarts.
- MCP processes are opened once during application startup and closed cleanly during shutdown.
- For production, add authentication, persistent storage, request limits, and HTTPS.
