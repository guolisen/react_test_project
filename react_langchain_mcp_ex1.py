"""
Travel Planning Assistant Implementation - Based on Gaode Maps MCP + SSE + langchain_mcp_adapters + LangGraph
Using LangGraph StateGraph and Gaode Maps tools to plan travel itineraries
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import os
import asyncio
from dotenv import load_dotenv

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
model = init_chat_model(
    "gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url='https://api.zhizengzeng.com/v1/',
    #base_url='https://api.openai.com/v1/',
    #base_url='https://open.bigmodel.cn/api/paas/v4',
    model_provider="openai"
)


async def create_travel_planner_graph():
    """
    Create a travel planning StateGraph, using Gaode Maps API tools to process user queries
    
    Returns:
        StateGraph instance and MCP client
    """
    
    # Get Gaode Maps API key
    gaode_map_key = os.getenv("GAODE_MAP_KEY")
    
    # Create MCP client connecting to Gaode Maps API - using dictionary configuration
    client = MultiServerMCPClient(
        {
            "search": {
                "url": f"https://mcp.amap.com/sse?key={gaode_map_key}",
                "transport": "sse",
            }
        }
    )
    
    # Get MCP tools
    tools = await client.get_tools()
    
    # Define function to call the model
    def call_model(state: MessagesState):
        """Call language model with bound tools"""
        # Add system prompt for the model, guiding how to plan travel itineraries
        if state["messages"] and state["messages"][0] != "system":
            system_message = {
                "role": "system",
                "content": """You are a professional travel planning assistant, skilled at using Gaode Maps tools to create personalized travel itineraries.
                Your goal is to understand the user's travel needs and use the provided Gaode Maps tools to plan the best routes, find attractions, and arrange itineraries.
                
                When planning travel itineraries, please follow these steps:
                1. Analyze the user's travel needs, including departure location, destination, time constraints, and preferences
                2. Determine which Gaode Maps tools are needed to obtain the required information
                3. First plan the main routes and transportation methods
                4. Find popular attractions, food options, and accommodation choices along the route and at the destination
                5. Optimize the itinerary arrangement based on travel time and user preferences
                6. Provide detailed daily schedules, including activities, transportation, and time estimates
                7. Ensure the itinerary is reasonable, feasible, and contains sufficient details

                Please note the following points:
                - If the user does not specify particular attractions, recommend popular or distinctive attractions
                - Consider transportation time and time needed to visit attractions, avoiding overly tight schedules
                - Take into account weather, season, and local cultural factors
                - Provide a clear itinerary structure, explaining the transportation method for each segment
                """
            }
            state["messages"] = [system_message] + state["messages"]
        
        # Call the model and bind tools
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": state["messages"] + [response]}
    
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
    
    return graph, client


async def plan_travel_itinerary(query):
    """
    Main travel planning function, processes user queries and provides travel itineraries
    
    Parameters:
        query: User query string
        
    Returns:
        Dictionary containing travel planning results
    """
    try:
        # Create travel planning graph and client
        graph, client = await create_travel_planner_graph()
        
        # Convert query format
        formatted_query = {"messages": [{"role": "user", "content": query}]}
        
        # Run graph to process user query
        response = await graph.ainvoke(formatted_query)
        
        # Extract final answer
        final_message = response["messages"][-1] #["content"] #if response["messages"] else "未能生成回答"
        if "content" in final_message:
            final_message = final_message["content"]

        # Return formatted response
        return {
            "status": "success",
            "result": final_message.content,
            "messages": response["messages"]
        }

    except Exception as e:
        # Capture and return error information
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # Ensure resources are properly closed
        graph = None
        client = None



async def run_examples():
    """Run example queries"""
    # Example queries - travel planning
    queries = [
        "I want to travel from Beijing to Xi'an, please help me plan a three-day itinerary, including famous historical sites and local cuisine",
        "My family and I are planning to travel from Shanghai to Hangzhou, we need a two-day itinerary, we have elderly people and children, so we need a relatively relaxed schedule",
        "I plan to drive from Guangzhou to Guilin, please help me plan a five-day route, I want to experience the local natural scenery and ethnic minority cultures"
    ]
    
    # Select a query to run the example
    query = queries[0]
    
    print(f"Processing travel planning query: {query}")
    result = await plan_travel_itinerary(query)
    
    if result["status"] == "success":
        print("\n==== Travel Planning Result ====")
        print(result["result"])
        print("\n==== Planning Complete ====")
    else:
        print(f"Processing error: {result.get('error', 'Unknown error')}")


async def process_user_query(query):
    """
    Process user input query
    
    Parameters:
        query: User query string
        
    Returns:
        Travel planning result
    """
    try:
        result = await plan_travel_itinerary(query)
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error occurred while processing query: {str(e)}"
        }


async def main_async():
    """Main async function"""
    print("Travel Planning Assistant - Based on Gaode Maps MCP + SSE + langchain_mcp_adapters + LangGraph")
    print("Initializing system...")
    await run_examples()


def main():
    """Main function entry point"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
