"""
旅行规划助手实现 - 基于高德地图MCP + SSE + langchain_mcp_adapters + LangGraph
使用LangGraph StateGraph和高德地图工具规划旅行行程
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
import os
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

os.environ['GAODE_MAP_KEY'] = ''
#os.environ["OPENAI_API_KEY"] = '.'
#os.environ["OPENAI_API_KEY"] = 'sk-proj--Axy0c--'
os.environ["OPENAI_API_KEY"] = 'sk-'
os.environ['LANGCHAIN_TRACING_V2'] = "true"   
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"   
os.environ['LANGCHAIN_API_KEY'] = "lsv2pt_7f6ce94edab445cfacc2a9164333b97d_11115ee170"   
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
    创建旅行规划StateGraph，使用高德地图API工具处理用户查询
    
    返回:
        StateGraph实例和MCP客户端
    """
    
    # 获取高德地图API密钥
    gaode_map_key = os.getenv("GAODE_MAP_KEY")
    
    # 创建MCP客户端连接高德地图API - 使用字典配置
    client = MultiServerMCPClient(
        {
            "search": {
                "url": f"https://mcp.amap.com/sse?key={gaode_map_key}",
                "transport": "sse",
            }
        }
    )
    
    # 获取MCP工具
    tools = await client.get_tools()
    
    # 定义调用模型的函数
    def call_model(state: MessagesState):
        """使用工具绑定调用语言模型"""
        # 为模型添加系统提示，指导其如何规划旅行行程
        if state["messages"] and state["messages"][0] != "system":
            system_message = {
                "role": "system",
                "content": """你是一个专业的旅行规划助手，擅长使用高德地图工具创建个性化旅行行程。
                你的目标是理解用户的旅行需求，并使用提供的高德地图工具规划最佳路线、查找景点和安排行程。
                
                在规划旅行行程时，请遵循以下步骤：
                1. 分析用户的旅行需求，包括出发地、目的地、时间限制和偏好
                2. 确定需要使用的高德地图工具来获取所需信息
                3. 首先规划主要路线和交通方式
                4. 查找沿途和目的地的热门景点、美食和住宿选择
                5. 根据旅行时间和用户偏好，优化行程安排
                6. 提供详细的日程安排，包括每天的活动、交通和时间估计
                7. 确保行程合理、可行，并包含足够的详细信息

                请注意以下几点：
                - 如果用户没有指定具体景点，请推荐热门或特色景点
                - 考虑交通时间和参观景点所需时间，避免行程过于紧凑
                - 考虑天气、季节和当地文化因素
                - 提供清晰的行程结构，并说明每个环节的交通方式
                """
            }
            state["messages"] = [system_message] + state["messages"]
        
        # 调用模型并绑定工具
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": state["messages"] + [response]}
    
    # 构建状态图
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
    旅行规划主函数，处理用户查询并提供旅行行程
    
    参数:
        query: 用户查询字符串
        
    返回:
        包含旅行规划结果的字典
    """
    try:
        # 创建旅行规划图和客户端
        graph, client = await create_travel_planner_graph()
        
        # 转换查询格式
        formatted_query = {"messages": [{"role": "user", "content": query}]}
        
        # 运行图处理用户查询
        response = await graph.ainvoke(formatted_query)
        
        # 提取最终回答
        final_message = response["messages"][-1] #["content"] #if response["messages"] else "未能生成回答"
        if "content" in final_message:
            final_message = final_message["content"]

        # 返回格式化的响应
        client = None
        return {
            "status": "success",
            "result": final_message.content,
            "messages": response["messages"]
        }

    except Exception as e:
        # 捕获并返回错误信息
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        # 确保资源正确关闭
        graph = None



async def run_examples():
    """运行示例查询"""
    # 示例查询 - 旅行规划
    queries = [
        "我想从北京到西安旅游，请帮我规划三天的行程，包括著名的历史景点和当地美食",
        "我和家人计划从上海到杭州旅游，需要一个两天的行程，我们有老人和孩子，所以需要比较轻松的行程",
        "我打算从广州到桂林自驾游，请帮我规划一条五天的路线，想体验当地的自然风光和少数民族文化"
    ]
    
    # 选择一个查询来运行示例
    query = queries[0]
    
    print(f"处理旅行规划查询: {query}")
    result = await plan_travel_itinerary(query)
    
    if result["status"] == "success":
        print("\n==== 旅行规划结果 ====")
        print(result["result"])
        print("\n==== 规划完成 ====")
    else:
        print(f"处理出错: {result.get('error', '未知错误')}")


async def process_user_query(query):
    """
    处理用户输入的查询
    
    参数:
        query: 用户查询字符串
        
    返回:
        旅行规划结果
    """
    try:
        result = await plan_travel_itinerary(query)
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": f"处理查询时发生错误: {str(e)}"
        }


async def main_async():
    """主异步函数"""
    print("旅行规划助手 - 基于高德地图MCP + SSE + langchain_mcp_adapters + LangGraph")
    print("正在初始化系统...")
    await run_examples()


def main():
    """主函数入口点"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
