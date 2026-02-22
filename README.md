## Running the Assistant

```
python main.py
```

## Project Structure

```
report-building-agent
├── src/
│ ├── schemas.py # Pydantic models
│ ├── retrieval.py # Document retrieval
│ ├── tools.py # Agent tools
│ ├── prompts.py # Prompt templates
│ ├── agent.py # LangGraph workflow
│ └── assistant.py # Main agent
├── sessions/ # Saved conversation sessions
├── main.py # Entry point
├── requirements.txt # Dependencies
└── README.md # This file
```

## Key Concepts

### 1. LangChain Tool Pattern

Tools are functions decorated with `@tool` that can be called by LLMs. They must:

- Have clear docstrings describing their purpose and parameters
- Handle errors gracefully* Return string results
- Log their usage for debugging

### 2. LangGraph State Management

The state flows through nodes and gets updated at each step. Key principles:

- Always return the updated state from node functions
- Use the state to pass information between nodes
- The state persists conversation context and intermediate results

### 3. Structured Output

Use `llm.with_structured_output(YourSchema)` to get reliable, typed responses from LLMs instead of parsing strings.

### 4. Conversation Memory

The system maintains conversation via the InMemorySaver checkpointer:

- Storing conversation messages with metadata
- Tracking active documents
- Summarizing conversations
- Providing context to subsequent requests


## Expected Behavior

The assistant should be able to:

- Classify user intents correctly
- Search and retrieve relevant documents
- Answer questions with proper source citations
- Generate comprehensive summaries
- Perform calculations on document data
- Maintain conversation context across turns
