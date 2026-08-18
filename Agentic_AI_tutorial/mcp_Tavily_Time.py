import os
import sys
import asyncio
import subprocess
import json

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
tavily_api_key = os.getenv("TAVILY_API_KEY")

print(f"🔑 OpenAI API Key: {'✅ Loaded' if api_key else '❌ Missing'}")
print(f"🔑 Tavily API Key: {'✅ Loaded' if tavily_api_key else '❌ Missing'}")
print(f"🔗 Base URL: {base_url or 'Default'}")

# ============================================================
# 2. Find npx
# ============================================================

def find_npx():
    possible_paths = [
        r"C:\Program Files\nodejs\npx.cmd",
        r"C:\Program Files\nodejs\npx.exe",
        r"C:\Program Files (x86)\nodejs\npx.cmd",
        r"C:\Program Files (x86)\nodejs\npx.exe",
        os.path.expanduser(r"~\AppData\Roaming\npm\npx.cmd"),
        os.path.expanduser(r"~\AppData\Roaming\npm\npx.exe"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try from PATH
    try:
        result = subprocess.run(
            ["where", "npx"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            path = result.stdout.strip().split('\n')[0]
            if path:
                return path
    except:
        pass
    
    return None

NPX_PATH = find_npx()
print(f"📦 npx path: {NPX_PATH if NPX_PATH else '❌ Not found'}")

# ============================================================
# 3. Alternative: Use Tavily Python Package Directly
# ============================================================

TAVILY_TOOLS = []

if tavily_api_key:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=tavily_api_key)
        print("✅ Tavily Python client initialized")
        
        @tool
        def tavily_search(query: str) -> str:
            """Search the web for current information on any topic."""
            try:
                result = tavily_client.search(query, max_results=5)
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def tavily_extract(urls: str) -> str:
            """Extract content from URLs. Input: comma-separated URLs."""
            try:
                url_list = [u.strip() for u in urls.split(",")]
                result = tavily_client.extract(urls=url_list)
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Error: {str(e)}"
        
        TAVILY_TOOLS = [tavily_search, tavily_extract]
        print(f"✅ Added {len(TAVILY_TOOLS)} Tavily tools")
        
    except ImportError:
        print("⚠️  tavily-python not installed. Run: pip install tavily-python")
    except Exception as e:
        print(f"⚠️  Tavily init error: {e}")

# ============================================================
# 4. Safe wrapper for MCP tools
# ============================================================

async def get_mcp_tools_safe():
    """Safely get MCP tools with error handling"""
    
    mcp_config = {}
    
    # Use remote time server (no local installation needed)
    mcp_config["time"] = {
        "transport": "streamable_http",
        "url": "https://mcp.time.mcpcentral.io"
    }
    print("✅ Time MCP server configured (remote)")
    
    # Only add Tavily MCP if npx is available and we're not using Python package
    if tavily_api_key and NPX_PATH and not TAVILY_TOOLS:
        mcp_config["tavily"] = {
            "command": NPX_PATH,
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": tavily_api_key},
            "transport": "stdio",
        }
        print("✅ Tavily MCP server configured")
    elif TAVILY_TOOLS:
        print("ℹ️  Using Tavily Python tools instead of MCP")
    else:
        print("ℹ️  Tavily tools not available")
    
    if not mcp_config:
        return []
    
    try:
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()
        return tools
    except Exception as e:
        print(f"⚠️  MCP tools error: {e}")
        return []

# ============================================================
# 5. Main async function
# ============================================================

async def main():
    # ========================================================
    # 6. Get tools from MCP
    # ========================================================
    
    mcp_tools = await get_mcp_tools_safe()
    
    # Combine MCP tools + Tavily Python tools
    all_tools = list(mcp_tools) + TAVILY_TOOLS
    
    print("\n🔧 Discovered tools:")
    print("=" * 60)
    
    for tool in all_tools:
        print(f"\n📌 Tool: {tool.name}")
        print(f"   Description: {tool.description}")
    
    if not all_tools:
        print("❌ No tools available!")
        return
    
    # ========================================================
    # 7. Create LLM
    # ========================================================
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key,
        base_url=base_url
    )
    print("\n✅ OpenAI LLM initialized")

    ollama = init_chat_model("llama3.1:8b ", model_provider="ollama", temperature=0)
    
    # ========================================================
    # 8. Create Agent
    # ========================================================
    
    agent = create_agent(
        #model=llm,
        model=ollama,
        tools=all_tools,
        system_prompt="""
        You are a helpful AI assistant with access to tools.
        
        Available tools include:
        - Time tools: Get current time in different locations
        - Tavily tools: Search the web for current information
        
        Use tools when needed. Do not guess information.
        """
    )
    print("✅ Agent created successfully!")
    
    # ========================================================
    # 9. Test Time
    # ========================================================
    
    print("\n\n" + "="*60)
    print("===== TEST 1: TIME TOOL =====")
    print("="*60)
    
    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "What time is it in Tokyo right now?"}
                ]
            }
        )
        print(f"\n🤖 {result['messages'][-1].content}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # ========================================================
    # 10. Test Search
    # ========================================================
    
    if TAVILY_TOOLS:
        print("\n\n" + "="*60)
        print("===== TEST 2: SEARCH =====")
        print("="*60)
        
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Search for the latest developments in quantum computing."
                        }
                    ]
                }
            )
            print(f"\n🤖 {result['messages'][-1].content}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("\n\n" + "="*60)
        print("===== TEST 2: SEARCH (SKIPPED) =====")
        print("="*60)
        print("Tavily tools not available. Run: pip install tavily-python")
    
    # ========================================================
    # 11. Test Combined
    # ========================================================
    
    if TAVILY_TOOLS:
        print("\n\n" + "="*60)
        print("===== TEST 3: COMBINED =====")
        print("="*60)
        
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": """
                            Search for the latest AI news.
                            Find where the event happened.
                            Then tell me the current time in that location.
                            """
                        }
                    ]
                }
            )
            print(f"\n🤖 {result['messages'][-1].content}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("\n\n" + "="*60)
        print("===== TEST 3: COMBINED (SKIPPED) =====")
        print("="*60)

# ============================================================
# 12. Start
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")