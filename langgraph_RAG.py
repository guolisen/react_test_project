from typing import Annotated, Literal, Tuple, Any, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Send, Interrupt, Command

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.schema import HumanMessage, AIMessage
from langchain_community.utilities import SerpAPIWrapper

# Vector store and embedding
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import subprocess
import getpass

import langchain
#langchain.debug = True

# Environment variables and configuration
os.environ["OPENAI_API_KEY"] = 'fd69c68ffab3452da1e00bbf6bd4c915.axvFwrXXiDDnJXKx'
os.environ["DASHSCOPE_API_KEY"] = 'sk-75c44da9706a4b039ebc0f81b5924786'

# Initialize LLM
gpt35_chat = ChatOpenAI(model="GLM-4-Plus", temperature=0, base_url="https://open.bigmodel.cn/api/paas/v4", verbose=False)

def initialize_llm():
    url = "http://10.226.163.45:11434"
    return OllamaLLM(model="llama3.1:8b", base_url=url, temperature=0.8, top_k=10, top_p=0.2, verbose=False,
               num_predict=16384, num_ctx=16384)

# Additional API configurations
os.environ['SERPAPI_API_KEY'] = 'e30fb0867db7fe3f78662ef26fc5059462c0c9bd2219be8efae54716a1ef6058'      
os.environ['LANGCHAIN_TRACING_V2'] = "true"   
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"   
os.environ['LANGCHAIN_API_KEY'] = "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"   
os.environ['LANGCHAIN_PROJECT'] = "pr-silver-bank-1"

# Initialize LLM
llm = gpt35_chat

# Initialize embedding model - using OpenAI embeddings which should be available
'''
embedding_model = OpenAIEmbeddings(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    #model="text-embedding-ada-002",
    model="embedding-3",
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
'''


from langchain_community.embeddings import DashScopeEmbeddings
embedding_model = DashScopeEmbeddings(
    model = 'text-embedding-v1',
    dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
)


# Set up vector database directory
db_dir = "./faiss_db"

# Function to create vector database from PDF
def create_db(pdf_path="sample.pdf") -> FAISS:
    """
    Create a FAISS vector database from a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        FAISS vector store
    """
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        docs = pdf_loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=2048, 
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embedding_model)
        vector_store.save_local(db_dir)
        print(f"Vector database created successfully with {len(chunks)} chunks")
        return vector_store
    except Exception as e:
        print(f"Error creating database: {e}")
        # Return an empty vector store if creation fails
        return FAISS.from_texts(["Error loading document"], embedding_model)

# Function to load or create vector database
def get_vector_store(pdf_path="sample.pdf") -> FAISS:
    """
    Load an existing vector store or create a new one if it doesn't exist
    
    Args:
        pdf_path: Path to the PDF file if creating new database
        
    Returns:
        FAISS vector store
    """
    is_db_exist = os.path.exists(db_dir)
    if is_db_exist:
        try:
            vector_store = FAISS.load_local(
                db_dir, 
                embedding_model,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector database")
            return vector_store
        except Exception as e:
            print(f"Error loading database: {e}")
            return create_db(pdf_path)
    else:
        return create_db(pdf_path)

# Initialize vector store with fallback mechanism
try:
    # Check if the PDF exists
    pdf_path = '/home/guolisen/react_test_project/2210.03629v3.pdf'
    if not os.path.exists(pdf_path):
        # If not, create a simple text file with sample content
        print(f"Warning: PDF file not found at {pdf_path}")
        print("Using text content for demonstration purposes")
        
        # Initialize with simple text for demonstration
        vector_store = FAISS.from_texts(
            ["This is a sample document for demonstrating RAG capabilities with LangGraph",
             "LangGraph is a framework for building stateful, multi-step applications with LLMs",
             "RAG (Retrieval Augmented Generation) enhances LLM responses with relevant document content",
             "FAISS is a library for efficient similarity search and clustering of dense vectors"], 
            embedding_model
        )
    else:
        vector_store = get_vector_store(pdf_path)
except Exception as e:
    print(f"Could not initialize vector store: {e}")
    # Fallback to a simple in-memory store with minimal content
    vector_store = FAISS.from_texts(["Fallback content for demonstration purposes"], embedding_model)

# Define the state for the graph
class State(TypedDict):
    """State for the RAG graph with web search capability"""
    messages: Annotated[list, add_messages]
    question: str

# Define the retrieval node
def retrieval(state: State):
    """
    Retrieve relevant documents for the user query
    
    Args:
        state: Current state with messages
        
    Returns:
        Updated state with retrieval results as a message
    """
    # Extract the user's query from the latest message
    if len(state["messages"]) >= 1:
        user_message = state["messages"][-1]
        user_query = user_message.content if hasattr(user_message, "content") else str(user_message)
    else:
        return {"messages": []}
    
    # Save question for later use
    state_updates = {"question": user_query}
    
    # Search for relevant documents
    try:
        docs = vector_store.similarity_search(user_query, k=3)
        
        # Create a prompt with the retrieved information
        template = """Please answer the user question based on the following retrieved information:

Retrieved Information:
{context}

User Question:
{question}

If the retrieved information doesn't provide enough context to answer the question accurately, 
respond with 'I don't have sufficient information to answer this question.'
"""
        
        # Format the context from retrieved documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Create a message with the retrieval results
        prompt_message = HumanMessage(
            content=template.format(context=context, question=user_query)
        )
        
        # Add the formatted message to the state
        state_updates["messages"] = [prompt_message]
        
        return state_updates
    except Exception as e:
        # If retrieval fails, return a message indicating the error
        error_message = HumanMessage(
            content=f"Error during retrieval: {str(e)}. Please try a different question."
        )
        state_updates["messages"] = [error_message]
        return state_updates

# Define the AI chatbot node
def chat_bot(state: State):
    """
    Generate an AI response based on the messages in the state
    
    Args:
        state: Current state with messages
        
    Returns:
        Updated state with AI response added to messages
    """
    # Generate a response using the LLM
    response = llm.invoke(state["messages"])
    
    # Return the response to be added to messages
    return {"messages": [response]}

# Define the SERPAPI search node
def serpapi_search(state: State) -> Dict:
    """
    Search the web using SERPAPI and combine results with retrieved information
    
    Args:
        state: Current state with messages and question
        
    Returns:
        Updated state with combined information
    """
    question = state.get("question", "No question available")
    # Get the original retrieved information
    original_info = state["messages"][0].content if state["messages"] else ""
    
    # Initialize SerpAPI wrapper
    search = SerpAPIWrapper()
    
    # Perform the search
    try:
        search_results = search.run(question)
        
        # Create a combined prompt with both retrieved and web information
        combined_prompt = f"""Please answer the user question based on ALL the following information:

Original Retrieved Information:
{original_info}

Web Search Results:
{search_results}

User Question:
{question}

If you still don't have sufficient information to answer the question accurately,
respond with 'I cannot answer this question with the available information.'
"""
        # Create a message with the combined information
        combined_message = HumanMessage(content=combined_prompt)
        
        # Return the combined message
        return {"messages": [combined_message]}
    except Exception as e:
        # Handle any errors during search
        error_message = HumanMessage(
            content=f"Error during web search: {str(e)}. Using only the original information."
        )
        return {"messages": [error_message]}

# Define verification function for conditional branching
def verify(state: State) -> Literal["chat_bot", "serpapi"]:
    """
    Verify if the retrieved information can answer the user's question
    
    Args:
        state: Current state with messages
        
    Returns:
        Next node identifier: either "chat_bot" or "serpapi"
    """
    # Create a verification prompt for the LLM
    verification_message = HumanMessage(
        content="Based on the retrieved information, can you answer the user's question accurately? Respond with 'Y' if you can, or 'N' if you need more information from web search."
    )
    
    # Get the verification result
    all_messages = state["messages"] + [verification_message]
    response = llm.invoke(all_messages)
    
    # Determine the next node based on the response
    if "Y" in response.content:
        print("AI can answer this question with available information.")
        return "chat_bot"
    else:
        print("AI cannot answer this question, searching the web.")
        return "serpapi"

# Build the graph
def build_graph():
    """
    Construct the StateGraph with nodes and edges
    
    Returns:
        Compiled StateGraph
    """
    # Initialize memory saver for checkpointing
    memory = MemorySaver()
    
    # Create the graph builder
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("retrieval", retrieval)
    graph_builder.add_node("chat_bot", chat_bot)
    graph_builder.add_node("serpapi", serpapi_search)
    
    # Add edges
    graph_builder.add_edge(START, "retrieval")
    graph_builder.add_conditional_edges("retrieval", verify)
    graph_builder.add_edge("chat_bot", END)
    # SERPAPI results go to chat_bot instead of a separate final_answer node
    graph_builder.add_edge("serpapi", "chat_bot")
    
    # Compile the graph with checkpointing
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph

# Create thread configuration for the graph
thread_config = {"configurable": {"thread_id": "rag_thread_id"}}

# Global variable to track the current thread
current_thread_id = None

# Function to process the graph stream and handle updates
from langgraph.errors import GraphInterrupt

def stream_graph_updates(user_input: str, graph):
    """
    Process user input through the graph and display responses
    
    Args:
        user_input: User's question
        graph: Compiled StateGraph
    """
    global current_thread_id
    
    try:
        # Create a thread config if one doesn't exist yet, or use the existing one
        config = {"configurable": {"thread_id": current_thread_id or "rag_thread_id"}}
        
        # Pass user input to the graph
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        ):
            # Save the thread_id for later use
            if current_thread_id is None and hasattr(event, 'thread_id'):
                current_thread_id = event.thread_id
                print(f"Thread created with ID: {current_thread_id}")
            for key, value in event.items():
                # Check if we have an AI message to display
                if "messages" in value and value["messages"] and isinstance(value["messages"][-1], AIMessage):
                    print("Assistant:", value["messages"][-1].content)
    except Exception as e:
        print(f"Error in stream_graph_updates: {e}")

    
# Main execution function
def run():
    """
    Main execution loop for the RAG system with web search capability
    """
    # Build the graph
    graph = build_graph()
    
    print("=== LangGraph RAG System with Web Search ===")
    print("Type 'exit' to quit the program.\n")
    
    # Run the interaction loop
    while True:
        user_input = input("User: ").strip()
        
        # Check if the user wants to exit
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Process user input through the graph
        stream_graph_updates(user_input, graph)

# Run the program if executed directly
if __name__ == "__main__":
    run()
