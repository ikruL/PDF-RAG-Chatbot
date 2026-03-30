
from langgraph.graph import StateGraph, END
from typing import Sequence, Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import config
from src.prompts import get_system_prompt


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def retriver_agent(state: AgentState) -> AgentState:
    """Main agent node responsible for reasoning and deciding whether to use tools.
    Args:
        state (AgentState): Current state containing the message history
    Returns:
        AgentState: Updated state with the new AI message appended
    """

    system_prompt = get_system_prompt()

    messages = [system_prompt] + list(state["messages"])

    res = llm.invoke(messages)

    return {"messages": [res]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"

    if isinstance(last_message, ToolMessage):
        return "continue"

    return END


def create_agent_graph(gemini_api_key, max_tokens=700, tools=None):
    """Create Agent with API Key from user"""

    global llm

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, streaming=True, api_key=gemini_api_key, temperature=0, max_tokens=max_tokens).bind_tools(tools=tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", retriver_agent)
    workflow.add_node("tools", ToolNode(tools=tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "continue": "tools",
        END: END
    })
    workflow.add_edge("tools", "agent")

    return workflow.compile()
