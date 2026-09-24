from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version

import importlib.metadata
try:
    graph_version = importlib.metadata.version("langgraph")
except importlib.metadata.PackageNotFoundError:
    graph_version = "unknown (Not installed)"

from langchain_ollama import __version__ as ollama_version
from langchain_ollama import ChatOllama

def main():

    print(f"langchain-core version {core_version}")
    print(f"langgraph version {graph_version})")
    print(f"langchain_ollama version {ollama_version}")
    print(f"Langgraph version: {graph_version}")

# Test Ollama
    llm = ChatOllama(
        model="gemma4:cloud",
        temperature=0.7
    )
    response = llm.invoke("Say 'setup is working' in a sentence.")
    print(f"Response from Ollama: {response.content}")
        
    print("this is a main is working")
    
if __name__ == "__main__":
    main()