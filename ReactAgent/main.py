from graph import build_graph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, BaseMessage
import threading
from prompts import SYSTEM_PROMPT
from rich import print
from rich.markdown import Markdown



def main():

    config = RunnableConfig(configurable={"thread_id": threading.get_ident()})
    graph = build_graph()
    
    while True:    
        all_messages: list[BaseMessage] = []
        print("[bold cyan]Você: ")
        user_input = input("")
        
        human_message = HumanMessage(user_input)
        current_loop_messages = [human_message]
        
        if user_input.lower() in ["q", "quit"]:
            break
        
        if len(all_messages) == 0:
            current_loop_messages = [SYSTEM_PROMPT, human_message]         
        
        result = graph.invoke({"messages": current_loop_messages}, config=config)
        print("[bold cyan]Resposta: \n")
        print(Markdown(result["messages"][-1].content))
        print(Markdown("\n\n  ---  \n\n"))
    
        all_messages = [result["messages"]]

if __name__ == "__main__":
    main()