from typing import Optional, TypedDict, Annotated, Dict, Any, Literal, Tuple

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from src.schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from langgraph.checkpoint.memory import InMemorySaver
from src.prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT

NextStep = Literal["classify_intent", "qa_agent", "summarization_agent", "calculation_agent", "update_memory", "__end__"]

class AgentState(TypedDict, total=False):
    # Current conversation
    user_input: str
    messages: Annotated[list[BaseMessage], add_messages]
    # Intent and routing
    intent: Optional[UserIntent]
    next_step: NextStep
    # Memory and context
    # conversation_history: list[BaseMessage]
    conversation_summary: str
    active_documents: list[str]
    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: list[str]
    # Session management
    session_id: str
    user_id: str
    actions_taken: Annotated[list[str], operator.add]


def invoke_react_agent(
    response_schema: type[BaseModel],
    messages: list[BaseMessage],
    llm,          # BaseChatModel
    tools,        # Sequence[BaseTool | Callable | dict]
) -> Tuple[Dict[str, Any], list[str]]:
    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=response_schema,
    )

    result: Dict[str, Any] = agent.invoke({"messages": messages}) # type: ignore

    tools_used: list[str] = []
    for message in result.get("messages", []):
        if isinstance(message, ToolMessage) and message.name != None:
            tools_used.append(message.name)

    return result, tools_used


# This function should classify the user's intent and set the next step in the workflow.
def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable", {}).get("llm")
    history = state.get("messages", [])
    
    if llm is None:
        raise ValueError("Missing 'llm' in config['configurable']")

    # Configure the llm chat model for structured output
    structured_llm = llm.with_structured_output(UserIntent)

    # Create a formatted prompt with conversation history and user input
    prompt = get_intent_classification_prompt().format(
        user_input=state.get("user_input", ""), 
        conversation_history=history
    )

    intent_response: UserIntent = structured_llm.invoke(prompt)
    intent = intent_response.intent_type
    
    # Add conditional logic to set next_step based on intent
    next_step: NextStep
    match intent:
        case "summarization":
            next_step = "summarization_agent"
        case "calculation":
            next_step = "calculation_agent"
        case _:
            next_step = "qa_agent"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent_response,
        "next_step": next_step
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools") or []
    
    if llm is None:
        raise ValueError("Missing 'llm' in config['configurable']")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state.get("user_input", ""),
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools") or []

    if llm is None:
        raise ValueError("Missing 'llm' in config['configurable']")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "input": state.get("user_input", ""),
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    tools = config.get("configurable", {}).get("tools") or []

    if llm is None:
        raise ValueError("Missing 'llm' in config['configurable']")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state.get("user_input", ""),
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """
    llm = config.get("configurable", {}).get("llm")
    
    if llm is None:
        raise ValueError("Missing 'llm' in config['configurable']")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    response: UpdateMemoryResponse = structured_llm.invoke(prompt_with_history)
    return {
        "actions_taken": ["update_memory"],
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "__end__",
    }

def should_continue(state: AgentState) -> NextStep:
    """Router function"""
    return state.get("next_step", "__end__")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "__end__": END,
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    return workflow.compile(checkpointer=InMemorySaver())
