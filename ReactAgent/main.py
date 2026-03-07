from graph import build_graph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
import threading
from rich import print
import sys



def main():
    
    print(sys.path)
    
    config = RunnableConfig(configurable={"thread_id": threading.get_ident()})
    graph = build_graph()

    user_input = "Olá, eu sou Caio"
    human_message = HumanMessage(user_input)
    current_messages = [human_message]
    
    result = graph.invoke({"messages": current_messages}, config=config)

    print(result)



if __name__ == "__main__":
    main()