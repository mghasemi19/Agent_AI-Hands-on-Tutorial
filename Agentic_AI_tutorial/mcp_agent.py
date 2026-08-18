import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

_ = load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["./utils/math_server.py"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # MCP handshake
            await session.initialize()

            # Discover MCP tools and convert them
            # into LangChain tools
            tools = await load_mcp_tools(session)

            print("Available tools:")
            for tool in tools:
                print("-", tool.name)

            model = ChatOpenAI(
                model="gpt-5.2",
                temperature=0, api_key=api_key, base_url=base_url
            )

            agent = create_agent(
                model=model,
                tools=tools
            )

            result = await agent.ainvoke(
                {
                    "messages": [
                        {   
                            "role": "system",
                            "content": "Do not use your own knowledge about addition and multiplication. "
                                       "First tell me which tool you used and then the answer. At the beggining"
                                       "also write the user prompt."
                        },                        
                        {   
                            "role": "user",
                            "content": "What is (3 + 5) multiplied by 12?"
                        }
                    ]
                }
            )
            
            print(result["messages"][-1].content)


asyncio.run(main())