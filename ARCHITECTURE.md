# ShopEasy Support Agent - Architectural Design Document

This document details the architectural layout, data flow, state machine representation, and component configurations for the **ShopEasy Customer Support Agent**.

---

## 1. System Overview

The application features a decoupled architecture composed of:
1. **Frontend (Streamlit)**: A chat interface providing real-time interactions, message history preservation, and session administration.
2. **Backend (FastAPI)**: An asynchronous REST API hosting the agent runtime, endpoint validation, and service configuration.
3. **Agent Runtime (LangGraph)**: A stateful graph executor managing conversational memory, vector database retrieval, security guardrails, and tool execution.
4. **Vector Store (Pinecone)**: A managed vector database storing dense embeddings of ShopEasy internal policies.

```
+-------------------------------------------------------------+
|                     1. Streamlit UI                         |
|  - Renders Chat Input/Output                                |
|  - Manages Session State (UUID)                             |
|  - Sends HTTP POST Requests to FastAPI                      |
+------------------------------+------------------------------+
                               |
                               | HTTP JSON Payload:
                               | { "query": "...", "session_id": "..." }
                               v
+-------------------------------------------------------------+
|                     2. FastAPI Server                       |
|  - Endpoints: GET / (health), POST /trigger_rag             |
|  - Payload validation (Pydantic schema: GetQuery)           |
|  - Invokes LangGraph engine using thread_id                 |
+------------------------------+------------------------------+
                               |
                               | Graph Invocation
                               v
+-------------------------------------------------------------+
|                     3. LangGraph Orchestrator               |
|  State: CustomerAgent TypedDict                             |
|  Checkpointing: MemorySaver (Stateful session store)        |
|                                                             |
|   +-------------------+                                     |
|   |  Input Guardrail  +----(Unsafe Input)-----+             |
|   +---------+---------+                       |             |
|             | (Safe)                          v             |
|   +---------v---------+                +------+------+      |
|   | Retrieve Context  |                |     END     |      |
|   +---------+---------+                +------^------+      |
|             |                                 |             |
|   +---------v---------+                       |             |
|   |  Agent Generate   |                       |             |
|   +----+---------+----+                       |             |
|        |         | (Final Answer)             |             |
|  (Tool |         v                            |             |
|  Call) |  +------+-----------+                |             |
|        |  | Output Guardrail +--(Pass/Fail)---+             |
|        |  +------------------+                              |
|   +----v----+                                               |
|   |  Tools  |                                               |
|   +---------+                                               |
+------------------------------+------------------------------+
```

---

## 2. LangGraph State Machine & Node Specifications

The agent is compiled using `langgraph.graph.StateGraph` and relies on the state class `CustomerAgent` for tracking state across nodes:

```python
class CustomerAgent(TypedDict):
    query: str                          # Original user query
    messages: Annotated[list, add_messages]  # Chat message history (appended automatically)
    documents: Optional[List[Document]] # Retrieved policy chunks
    context: Optional[str]              # Formatted text of retrieved docs
    answer: Optional[str]               # Compiled final response
    guardrail_status: Optional[str]     # security status: "safe" | "blocked"
    guardrail_reason: Optional[str]     # Diagnostic explanation if blocked
```

### Nodes Configuration

#### A. Input Guardrail (`input_guardrail`)
- **Location**: [utils/graph.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/utils/graph.py#L45-L108)
- **Objective**: Prevent prompt leakages, jailbreaks, and injection attacks.
- **Implementation**:
  1. **Regex/Rule layer**: Fast check against known prompt-leak keywords (e.g. `system prompt`, `ignore instructions`, `developer prompt`). Matches exit immediately, returning a default support rejection message with low latency.
  2. **LLM layer**: A structured classification prompt sent to `llama-3.3-70b-versatile` classifying the query as either `SAFE` or `UNSAFE`.
- **Transitions**:
  - `route_after_input_guardrail` determines:
    - If `guardrail_status` is `"blocked"`, route to `END`.
    - Otherwise, route to `retrieve_context`.

#### B. Retrieve Context (`retrieve_context`)
- **Location**: [utils/graph.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/utils/graph.py#L157-L160)
- **Objective**: Extract relevant policy sections from vector memory.
- **Implementation**: Queries the Pinecone vector retriever based on `state["query"]`. Formats the list of retrieved `Document` instances into a single string under the `context` key.
- **Transitions**: Unconditionally routes to `agent_generate`.

#### C. Agent Generate (`agent_generate`)
- **Location**: [utils/graph.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/utils/graph.py#L163-L188)
- **Objective**: Generate a conversation response or decide if external tooling is required.
- **Implementation**:
  - System Prompt establishes the agent Persona ("Customer Support Representative for ShopEasy"), rules on context utilization, instructions to state when external search is used, and a directive to avoid technical terms.
  - Binds the `web_search` tool configuration to the LLM.
- **Transitions**:
  - Evaluates standard LangGraph `tools_condition`:
    - If LLM specifies a `tool_calls` request: Routes to the `tools` node.
    - If LLM provides a terminal response text: Routes to `output_guardrail`.

#### D. Tools Execution (`tools`)
- **Location**: Prebuilt `ToolNode` instantiated with `[web_search]`.
- **Objective**: Execute auxiliary searches for queries not resolved by internal documents.
- **Implementation**: Runs the search query using DuckDuckGo search tool, logging results, and pushing tool messages back into the state's `messages` list.
- **Transitions**: Loops back to `agent_generate`.

#### E. Output Guardrail (`output_guardrail`)
- **Location**: [utils/graph.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/utils/graph.py#L119-L149)
- **Objective**: Prevent hallucinations, bad tone, and PII leakage.
- **Implementation**: Prompts the LLM as a QA reviewer to audit the proposed response text. Returns `PASS` or `FAIL`. If `FAIL`, the response is overridden with: *"I'm sorry, I wasn't able to generate a reliable answer. Please contact ShopEasy support directly."*
- **Transitions**: Unconditionally routes to `END`.

---

## 3. RAG Pipeline & Ingestion Strategy

All RAG operations are situated in [utils/rag.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/utils/rag.py).

```
+--------------------------------------------------------------+
|                    Document Ingestion                        |
|                                                              |
| 1. PyPDFLoader loads ShopEasy PDF                            |
| 2. RecursiveCharacterTextSplitter (chunk: 1000, overlap: 200)|
+------------------------------+-------------------------------+
                               | Document Chunks
                               v
+--------------------------------------------------------------+
|                    Embedding Generation                      |
|                                                              |
| HuggingFace Embedding Model: all-MiniLM-L6-v2                |
| Generates 384-dimensional floating-point vectors             |
+------------------------------+-------------------------------+
                               | Chunks & Embeddings
                               v
+--------------------------------------------------------------+
|                    Vector Store Indexing                     |
|                                                              |
| Pinecone Client connects to Index:                           |
|   - Name: customer-support-agent-master                      |
|   - Metric: Cosine Similarity                                |
|   - Setup: AWS Serverless (us-east-1)                        |
+--------------------------------------------------------------+
```

### Document Loading and Chunking
- Loader: `PyPDFLoader` handles extraction from the source document (referenced by the `PDF_PATH` environment variable).
- Chunking: `RecursiveCharacterTextSplitter` segments text into chunks of 1000 characters with a 200-character overlap to preserve semantic context across chunk edges.

### Vector Generation & Indexing
- Embedding Model: HuggingFace's `all-MiniLM-L6-v2` is loaded locally. It produces dense embeddings of size 384.
- Index Verification:
  - If `customer-support-agent-master` index does not exist in Pinecone, it programmatically creates it with Cosine metric similarity on serverless AWS container.
  - Chunks are embedded and stored in bulk.
  - If the index is already present, it attaches directly using `PineconeVectorStore.from_existing_index`.

---

## 4. Conversation Stateful Memory

LangGraph's checkpointing mechanism tracks conversation contexts:
- A `MemorySaver` checkpointer is compiled with the graph: `builder.compile(checkpointer=MemorySaver())`.
- State configurations specify a `thread_id` generated on the Streamlit frontend (`st.session_state.session_id` using UUIDv4).
- The state history of the agent (all messages, documents, and node variables) is saved in memory after each transaction, permitting continuous multi-turn conversations without manual state merging.

---

## 5. Security & Fallback Logic

1. **Prompt Injection & Leak Defense**: Restricts malicious system inquiries via string checking and custom LLM validation.
2. **Hallucination Prevention**: Output Guardrail reviews outputs, and the LLM generation rules specify that any external tool facts must be labeled as external references, discouraging false policy creation.
3. **Graceful Failures**: High-level try-except blocks in [main.py](file:///d:/Projects/day_28_final_project/shopEasy-customer-support-agent/main.py#L39-L41) capture graph failures and return unified customer-friendly warnings instead of raw tracebacks.
