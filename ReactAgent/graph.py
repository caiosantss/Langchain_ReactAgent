from langchain_core.messages import ToolMessage
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver
from state import State
from utils import load_llm
from tools import TOOLS, TOOL_BY_NAME
from typing import Literal
from langchain_core.messages import AIMessage

#node
def call_llm(state: State) -> State:
    print(">call_llm")
    result = load_llm().bind_tools(TOOLS).invoke(state["messages"])
    
    #State=messages: [Anottated[Sequence[BaseMessage], add_message]]
    return {"messages" : [result]}


#node
def tool_node(state: State) -> State:
    print(">tool_node")
    #Last message in State -> have to ve a tool_calls
    llm_response = state["messages"][-1]
    
    if not getattr(llm_response, "tool_calls"):
        return state
    
    #If have tool_calls, get the last one and execute the tool
    if isinstance(llm_response, AIMessage):
        call = llm_response.tool_calls[-1]

    #Get the atributes of the call
    id_, name, args = call["id"], call["name"], call["args"]
    
    #Link the name attribute to our tool
    try:
        #invoke(): content = get_weather(**args)
        content = TOOL_BY_NAME[name].invoke(args)
    except (KeyError, IndexError, TypeError) as error:
        content = f"Please, fix the error: {error}"
        
    return {"messages": [ToolMessage(content=content, tool_call_id=id_)]}
    
    
#router - conditional
def router(state: State) -> Literal["tool_node", "__end__"]:
    print(">router")
    llm_response = state["messages"][-1]
    
    if(getattr(llm_response, "tool_calls")):
        return "tool_node"
    
    return "__end__"    

def build_graph() -> CompiledStateGraph:
    print(">build_graph")
    builder = StateGraph(State)
    
    #Create Nodes
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_node", tool_node)
    
    #Create Edges
    builder.add_edge(START, "call_llm")
    
    #Loop with tool call_llm -> tool -> call_llm -> END
    # or call_llm -> end
    builder.add_conditional_edges("call_llm", router, ["tool_node", "__end__"]) 
    builder.add_edge("tool_node", "call_llm")
    
    return builder.compile(checkpointer=InMemorySaver())