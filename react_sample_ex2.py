from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
import subprocess
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os, getpass
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = 'fd69c68ffab3452da1e00bbf6bd4c915.axvFwrXXiDDnJXKx'

gpt35_chat = ChatOpenAI(model="GLM-4-Flash", temperature=0, base_url="https://open.bigmodel.cn/api/paas/v4", verbose=False)

def initialize_llm():
    url = "http://10.226.163.45:11434"
#    return OllamaLLM(model="deepseek-r1:14b", base_url=url, temperature=0.3, top_k=30, top_p=0.2, verbose=True)
    return OllamaLLM(model="llama3.1:8b", base_url=url, temperature=0.8, top_k=10, top_p=0.2, verbose=False,
               num_predict=16384, num_ctx=16384)


import os      # OpenAI 环境变量   
# 搜索工具环境变量   
os.environ['SERPAPI_API_KEY'] = 'e30fb0867db7fe3f78662ef26fc5059462c0c9bd2219be8efae54716a1ef6058'      
# LangSmith 环境变量 (可选) ,如果需要使用 LangSmith 功能，请在环境变量中设置以下变量   
#os.environ['LANGCHAIN_TRACING_V2'] = "true"   
#os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"   
#os.environ['LANGCHAIN_API_KEY'] = "lsv2_pt1c9e"   
#os.environ['LANGCHAIN_PROJECT'] = "hello-agent"   



# 设置工具   
from langchain_core.tools import tool      
# 自定义计算器工具，用于计算鲜花的价格   
@tool   
def calculator(expression: str) -> str:       
    """使用 Python 的 numexpr 库计算数学表达式          
    表达式应该是一个单行的数学表达式          
    例如:           
    "352 * 493" 表示 "352 乘以 493"       
    """       
    import numexpr       
    import math  # 确保导入 math 库
    import re
    
    # Sanitize the expression - remove any potential problematic characters
    # Only allow digits, operators, decimal points, spaces and common math symbols
    clean_expr = re.sub(r'[^\d\s\.\+\-\*\/\(\)\^\%]', '', expression.strip())
    
    try:
        local_dict = {"pi": math.pi, "e": math.e}       
        result = str(           
            numexpr.evaluate(               
                clean_expr,               
                global_dict={}, 
                local_dict=local_dict,
                )       
                )       
        print(f"The result of {clean_expr} is {result}")       
        return result  # 确保返回结果为字符串
    except Exception as e:
        print(f"Error evaluating expression '{clean_expr}': {str(e)}")
        return f"Error: {str(e)}"

from langchain_community.agent_toolkits.load_tools import load_tools      
tools = [calculator]      
loaded_tools = load_tools(["serpapi"], llm=gpt35_chat)   
tools += loaded_tools   

# 设置提示模板   
from langchain.prompts import PromptTemplate      
template = '''
       尽你所能用中文回答以下问题。如果能力不够你可以使用以下工具:  {tools}
              Use the following format:              
              Question: the input question you must answer       
              Thought: you should always think about what to do       
              Action: the action to take, should be one of [{tool_names}]     

              Action Input: the input to the action       
              Observation: the result of the action       
              ... (this Thought/Action/Action Input/Observation can repeat N times)       
              Thought: I now know the final answer       

              Final Answer: the final answer to the original input question              
              Begin!              
              Question: {input}       
              Thought:{agent_scratchpad}       
            '''      
prompt = PromptTemplate.from_template(template)      # from langsmith import Client   #   # client = Client()   # prompt  = client.pull_prompt("hwchase17/react")      print("提示词：")   print(prompt)   

# 初始化Agent   
from langchain.agents import create_react_agent      
agent = create_react_agent(gpt35_chat, tools, prompt)      # 构建AgentExecutor   from langchain.agents import AgentExecutor      agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=False, verbose=True)   


# 执行
# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input":                             
                       """今天市场上玫瑰花的一般进货价格是多少？                              
                       如果我在此基础上加价5%，应该如何定价？"""})   
print(result["output"])
