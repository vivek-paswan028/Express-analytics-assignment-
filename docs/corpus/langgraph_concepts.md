# LangGraph Core Concepts

> Source: LangGraph official documentation (https://langchain-ai.github.io/langgraph/)
> License: MIT

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. It extends LangChain by adding the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

LangGraph is inspired by [Pregel](https://research.google/pubs/pub36726/) and [Apache Beam](https://beam.apache.org/), and is publicly inspired by [NetworkX](https://networkx.org/) in how the graph is defined.

## Key Concepts

### StateGraph

The `StateGraph` is the primary graph class in LangGraph. It is parameterized by a state definition (a TypedDict or Pydantic model) that all nodes read from and write to.

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict

class MyState(TypedDict):
    messages: list
    count: int

graph_builder = StateGraph(MyState)
```

The state definition determines:
1. What data flows between nodes
2. How updates from nodes are merged (using `Annotated` with reducer functions)
3. What the graph's input and output schemas look like

### Nodes

Nodes are Python functions (or callables) that receive the current state and return a dictionary of updates to apply to the state. Every node takes the full state as input and returns a partial state update.

```python
def my_node(state: MyState) -> dict:
    # Read from state
    current_count = state["count"]
    # Return updates — only the keys you want to change
    return {"count": current_count + 1}

graph_builder.add_node("my_node", my_node)
```

**Important:** Nodes do not mutate state directly. They return a dictionary, and LangGraph applies those updates according to the reducer functions defined on each field.

### Edges

Edges define the flow of execution between nodes. LangGraph supports two types of edges:

**Normal edges** always go from one node to another:
```python
graph_builder.add_edge("node_a", "node_b")
```

**Conditional edges** dynamically choose the next node based on the current state:
```python
def route_function(state: MyState) -> str:
    if state["count"] > 5:
        return "end_node"
    return "increment_node"

graph_builder.add_conditional_edges(
    "check_node",          # source node
    route_function,        # function that returns the name of the next node
    {
        "end_node": END,           # mapping from return value to node name
        "increment_node": "increment_node",
    }
)
```

The routing function receives the current state and returns a string key. The dictionary maps those keys to actual node names (or `END`).

### START and END

- `START` is a special node that represents the entry point of the graph
- `END` is a special node that represents a terminal state — execution stops when a node routes to `END`

```python
from langgraph.graph import END, START

graph_builder.add_edge(START, "first_node")
graph_builder.add_edge("last_node", END)
```

### Compiling the Graph

Before the graph can run, it must be compiled:

```python
graph = graph_builder.compile()
```

`compile()` validates the graph structure, checks for unreachable nodes, and returns a `CompiledGraph` object that can be invoked.

### Invoking the Graph

```python
result = graph.invoke({"messages": [], "count": 0})
```

`invoke()` runs the graph synchronously from `START` to `END` and returns the final state. For streaming, use `graph.stream()`.

## Reducers and State Updates

By default, state updates **overwrite** the existing value for a key. To accumulate values (like appending to a list), use `Annotated` with a reducer:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    # add_messages appends new messages instead of replacing the list
    messages: Annotated[list, add_messages]
    retry_count: int  # this will be overwritten by each update
```

Custom reducers are just functions:
```python
def increment(existing: int, new: int) -> int:
    return existing + new

class CountState(TypedDict):
    count: Annotated[int, increment]
```

## Cycles and Self-Correction

LangGraph explicitly supports cycles in the graph — a node can route back to an earlier node. This is the mechanism that enables self-corrective RAG workflows:

```python
# Node A → Node B → Node C → (conditional) → Node A (retry) or END
graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite_query",   # loops back
        "fallback": "fallback",
    }
)
```

To prevent infinite loops, always track retry counts in state:

```python
class RAGState(TypedDict):
    retry_count: int  # increment in the rewrite node; check in router
```

## Checkpointing and Persistence

LangGraph supports persistent checkpointing via `MemorySaver` (in-memory) or `SqliteSaver` (disk-based). Checkpoints allow graphs to be resumed after interruption and enable human-in-the-loop workflows.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Each invocation with a thread_id is checkpointed independently
config = {"configurable": {"thread_id": "session_001"}}
result = graph.invoke(input_data, config=config)
```

## Graph Visualization

Compiled graphs can be visualized as Mermaid diagrams:

```python
print(graph.get_graph().draw_mermaid())
```

This is useful for debugging graph structure and verifying that edges and conditional routing are wired correctly.

## Common Patterns

### Agentic RAG Pattern

The standard self-corrective RAG pattern with LangGraph:

1. `query_analysis` node — rewrites and classifies the user question
2. `retrieval` node — searches the vector store
3. `grade_documents` node — LLM grades each chunk as relevant or irrelevant
4. Conditional edge — routes to `generate` if relevant docs exist, else to `rewrite_query`
5. `rewrite_query` node — reformulates the question and loops back to `retrieval`
6. `generate` node — produces the final answer with citations
7. `fallback` node — returns "I don't know" after max retries

### Parallel Node Execution

Nodes without a dependency path between them can execute in parallel using `Send`:

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("process_item", {"item": item}) for item in state["items"]]
```

## Error Handling in Nodes

Nodes should handle exceptions gracefully and propagate errors through state rather than crashing the graph:

```python
def retrieval_node(state: RAGState) -> dict:
    try:
        docs = vector_store.similarity_search(state["rewritten_query"], k=5)
        return {"retrieved_documents": docs}
    except Exception as e:
        return {"error": f"Retrieval failed: {str(e)}", "retrieved_documents": []}
```

