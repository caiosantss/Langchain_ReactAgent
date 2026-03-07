from langchain_core.messages import ToolMessage
from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.constants import START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import State
from utils import load_llm
from tools import TOOLS, TOOL_BY_NAME

#node
def call_llm(state: State) -> State:
    result = load_llm().bind_tools(TOOLS).invoke(state["messages"])
    
    #State=messages: [Anottated[Sequence[BaseMessage], add_message]]
    return {"messages" : [result]}


#node
def tool_node(state: State) -> State:
    #Last message in State -> have to ve a tool_calls
    llm_response = state["messages"][-1]
    
    if not getattr(llm_response, "tool_calls"):
        return state
    
    #If have tool_calls, get the last one and execute the tool
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
    
    
    

def build_graph() -> CompiledStateGraph:
    builder = StateGraph(State)
    
    builder.add_node("call_llm", call_llm)
    
    builder.add_edge(START, "call_llm")
    builder.add_edge("call_llm", END)
    
    return builder.compile(checkpointer=InMemorySaver())