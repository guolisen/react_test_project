
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool


os.environ["OPENAI_API_KEY"] = '.'

gpt35_chat = ChatOpenAI(model="GLM-4-Plus", temperature=0, base_url="https://open.bigmodel.cn/api/paas/v4", verbose=False)

def initialize_llm():
    url = "http://10.226.163.45:11434"
#    return OllamaLLM(model="deepseek-r1:14b", base_url=url, temperature=0.3, top_k=30, top_p=0.2, verbose=True)
    return OllamaLLM(model="llama3.1:8b", base_url=url, temperature=0.8, top_k=10, top_p=0.2, verbose=False,
               num_predict=16384, num_ctx=16384)

# 搜索工具环境变量   
os.environ['SERPAPI_API_KEY'] = 'e30fb0867db7fe3f78662ef26fc5059462c0c9bd2219be8efae54716a1ef6058'      
# LangSmith 环境变量 (可选) ,如果需要使用 LangSmith 功能，请在环境变量中设置以下变量   
os.environ['LANGCHAIN_TRACING_V2'] = "true"   
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"   
os.environ['LANGCHAIN_API_KEY'] = ""   
os.environ['LANGCHAIN_PROJECT'] = "pr-silver-bank-1"   

# 定义状态
class State(TypedDict):
    messages: Annotated[list,add_messages]
 
# 创建 graph
graph_builder = StateGraph(State)

# llm = init_chat_model(
#     "gpt-4o-mini",
#     api_key = os.getenv("DMX_OPENAI_API_KEY"),
#     base_url = os.getenv("DMX_BASE_URL"),
#     model_provider="openai"
# )
 
llm = gpt35_chat
 
# 定义一个执行节点
def chatbot(state:State):
    # 调用大模型，并返回消息（列表）
    # 返回值会触发状态更新 add_messages
    return {"messages": [llm.invoke(state["messages"])]}

# 定义节点以及边
graph_builder.add_node("chatbot",chatbot)
 
# 定义边
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)
 
graph = graph_builder.compile()
 
# 可视化展示这个工作流
try:
    display(Image(data=graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(e)

while True:
    user_input = input("请输入您的问题：")
    if user_input == "exit":
        print("退出程序")
        break
 
     # 向 graph 传入一条消息（触发状态更新 add_messages）
    for message in graph.stream({"messages":[{"role":"user","content":user_input}]}):
        for value in message.values():
            content = value["messages"][-1].content
            print(f"Assistant: {content}")



