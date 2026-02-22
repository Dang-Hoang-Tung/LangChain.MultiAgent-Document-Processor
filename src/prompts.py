from typing import assert_never
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from src.types import IntentType


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an intent classifier for a document processing assistant.

Given the user input and conversation history, classify the user's intent into one of these categories:
- qa: Questions about documents or records that do not require calculations.
- summarization: Requests to summarize or extract key points from documents that do not require calculations.
- calculation: Mathematical operations or numerical computations. Or questions about documents that may require calculations
- unknown: Cannot determine the intent clearly

User Input: {user_input}

Recent Conversation History:
{conversation_history}

Analyze the user's request and classify their intent with a confidence score and brief reasoning.
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. Always search for relevant documents before answering
2. Cite specific document IDs when referencing information
3. If information is not found, say so clearly
4. Be precise with numbers and dates
5. Maintain professional tone

"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

# Calculation System Prompt
CALCULATION_SYSTEM_PROMPT = """You are a calculation agent responsible for performing precise mathematical computations based on user requests and document data.

Your responsibilities:
- Identify which document(s) are required to answer the user's request
- Retrieve the necessary document(s) using the document reader tool
- Determine the exact mathematical expression needed based on the user's input and document content
- Perform ALL calculations using the calculator tool, without exception

Guidelines:
1. Always determine whether a document must be retrieved before calculating
2. Read and extract the required numerical values from the document using the document reader tool
3. Explicitly form the mathematical expression needed to answer the question
4. Use the calculator tool for every calculation, no matter how simple
5. Do not perform mental math or inline arithmetic
6. If required data is missing, state clearly what cannot be calculated and why
7. Return the final calculated result clearly and concisely
"""


# Return the correct prompt based on intent type
def get_chat_prompt_template(intent_type: IntentType) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent.
    """
    match intent_type:
        case 'qa':
            system_prompt = QA_SYSTEM_PROMPT
        case 'summarization':
            system_prompt = SUMMARIZATION_SYSTEM_PROMPT
        case 'calculation':
            system_prompt = CALCULATION_SYSTEM_PROMPT
        case 'unknown':
            system_prompt = QA_SYSTEM_PROMPT
        case _:
            assert_never(intent_type)
            system_prompt = QA_SYSTEM_PROMPT # Default system prompt

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""
