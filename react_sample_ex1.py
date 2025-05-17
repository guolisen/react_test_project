from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
import subprocess
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os, getpass
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"] = '.'

gpt35_chat = ChatOpenAI(model="GLM-4-Plus", temperature=0, base_url="https://open.bigmodel.cn/api/paas/v4", verbose=False)

def initialize_llm():
    url = "http://10.226.163.45:11434"
#    return OllamaLLM(model="deepseek-r1:14b", base_url=url, temperature=0.3, top_k=30, top_p=0.2, verbose=True)
    return OllamaLLM(model="llama3.1:8b", base_url=url, temperature=0.8, top_k=10, top_p=0.2, verbose=False,
               num_predict=16384, num_ctx=16384)

# Define a tool to execute Linux commands
def run_command(command: str) -> str:
    print(f"!!!!!!!!!!!!!!!!!!!!! Running command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Define tools available to the agent
tools = [
    Tool(
        name="RunCommand",
        func=run_command,
        description="Execute a Linux command and return the output. Use for checking service status, reading logs, or running fixes."
    )
]


# Define the ReAct prompt template
react_prompt = PromptTemplate.from_template("""
You are a Linux system administrator assistant. Your task is to diagnose and fix Linux system issues by reasoning through the problem and using available tools. Follow the ReAct (Reasoning and Acting) pattern: think step-by-step, perform actions, and observe results.

For each step:
1. **Thought**: Explain your reasoning about the current state and what to do next.
2. **Action**: Specify the tool and input (e.g., `RunCommand: systemctl status apache2`).
3. **Observation**: Interpret the result of the action and decide the next step.

Available tools:
{tools}

Tool names: {tool_names}

Task: {input}

Agent scratchpad (intermediate steps): {agent_scratchpad}

Begin with an initial thoughtAren't you impressed with the picture quality?
Thought: ...
""")

# Initialize the model (LLaMA 3.1 via Ollama)
llm = initialize_llm()

# Create the ReAct agent
agent = create_react_agent(llm=gpt35_chat, tools=tools, prompt=react_prompt)

# Create the agent executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example: Diagnose why Apache is not running
if __name__ == "__main__":
    result = executor.invoke({"input": "Check the file name in the current directory."})
    print(result["output"])


