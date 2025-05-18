"""
Smart Travel System Implementation - Based on Amap MCP + SSE + langchain_mcp_adapters
"""
from langchain.agents import create_openai_functions_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv(override=True)

os.environ['GAODE_MAP_KEY'] = ''
#os.environ["OPENAI_API_KEY"] = '.'
#os.environ["OPENAI_API_KEY"] = 'sk-proj--Axy0c--'
os.environ["OPENAI_API_KEY"] = 'sk-'
os.environ['LANGCHAIN_TRACING_V2'] = "true"   
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"   
os.environ['LANGCHAIN_API_KEY'] = ""   
os.environ['LANGCHAIN_PROJECT'] = "pr-silver-bank-1"

# Initialize language model
llm_ds = init_chat_model(
    "gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url='https://api.zhizengzeng.com/v1/',
    #base_url='https://api.openai.com/v1/',
    #base_url='https://open.bigmodel.cn/api/paas/v4',
    model_provider="openai"
)


async def run_agent(query):
    """
    Implement smart travel assistant, process user queries and provide answers
    
    Parameters:
        query: User query string, for example "How to get from Beijing to Xi'an? And help me plan famous tourist attractions along the way"
        
    Returns:
        Dictionary containing query results
    """
    # Get Amap API key
    gaode_map_key = os.getenv("GAODE_MAP_KEY")
    
    # Create MCP client connecting to Amap API - using dictionary configuration
    client = MultiServerMCPClient(
        {
            "search": {
                "url": f"https://mcp.amap.com/sse?key={gaode_map_key}",
                "transport": "sse",
            }
        }
    )
    
    try:
        # Get MCP tools
        mcptools = await client.get_tools()
        
        # Create tools description string
        tools_description = "Available tools:\n"
        for tool in mcptools:
            tools_description += f"- {tool.name}: {tool.description}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful intelligent assistant. Please carefully analyze user questions and use the provided tools to answer questions."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=llm_ds,
            tools=mcptools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=mcptools,
            verbose=True,
            max_iterations=5,
            return_intermediate_steps=True,  # Return intermediate steps for debugging
            handle_parsing_errors=True
        )
        
        # Run agent to process user query
        agent_response = await agent_executor.ainvoke({
            "input": query
        })
        
        # Return formatted response
        return {
            "status": "success",
            "result": agent_response.get("output", ""),
            "steps": len(agent_response.get("intermediate_steps", [])),
        }
    
    except Exception as e:
        # Capture and return error information
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # Ensure resources are properly closed
        #await client.aclose()
        pass


async def call_agent_with_specific_context(query, context=None):
    """
    Call agent with specific context
    
    Parameters:
        query: User query
        context: Additional context (optional)
    
    Returns:
        Agent response
    """
    # Combine query and context
    full_query = query
    if context:
        full_query = f"Context information: {context}\n\nUser query: {query}"
    
    # Call agent
    return await run_agent(full_query)


async def main():
    """Run example query"""
    query = "How to get from Beijing to Xi'an? And help me plan famous tourist attractions along the way"
    
    print(f"Processing query: {query}")
    result = await run_agent(query)
    
    if result["status"] == "success":
        print(f"Query result: {result['result']}")
        print(f"Number of processing steps: {result['steps']}")
    else:
        print(f"Processing error: {result.get('error', 'Unknown error')}")


# Example using LangGraph StateGraph (not enabled, for reference only)
async def run_with_state_graph():
    """
    Run agent using LangGraph StateGraph (example code, not enabled)
    """
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition
    
    # Get Amap API key
    gaode_map_key = os.getenv("GAODE_MAP_KEY")
    
    # Create MCP client
    client = MultiServerMCPClient(
        {
            "search": {
                "url": f"https://mcp.amap.com/sse?key={gaode_map_key}",
                "transport": "sse",
            }
        }
    )
    
    # Get tools
    tools = await client.get_tools()
    
    def call_model(state: MessagesState):
        response = llm_ds.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}
    
    # Build state graph
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    
    # Run query
    response = await graph.ainvoke({"messages": "How to get from Beijing to Xi'an? And help me plan famous tourist attractions along the way"})
    
    # Clean up resources
    await client.aclose()
    
    return response


if __name__ == "__main__":
    # Run example query
    print("Smart Travel Assistant Example - Based on Amap MCP + SSE + langchain_mcp_adapters")
    asyncio.run(main())
