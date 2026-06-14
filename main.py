from fastapi import FastAPI
from schema import GetQuery, ReturnQuery
from utils.rag import _logger
from utils.graph import customer_graph
import traceback

app = FastAPI(title="FAST API RAG APPLICATION")

_logger.info("---------MAIN FILE IS GETTING LOADED----------")

@app.get("/")
def read_root():
    return {"message": "Hello World! The server booted up instantly."}


@app.post("/trigger_rag", response_model=ReturnQuery)
def trigger_rag(payload: GetQuery) -> ReturnQuery:
    _logger.info("----------INSIDE TRIGGER RAG")

    config = {
        "configurable": {
            "thread_id": payload.session_id
        }
    }

    try:
        result = customer_graph.invoke(
            {
                "query": payload.query,
                "messages": [("user", payload.query)]
            },
            config=config
        )

        answer = result.get("answer") or "I'm sorry, I couldn't process that. Please try again."
        _logger.info(f"Graph result answer: {answer}")
        return ReturnQuery(response=answer)

    except Exception as e:
        _logger.error(f"Graph invocation failed: {e}", exc_info=True)  # exc_info=True prints full traceback
        return ReturnQuery(response="Something went wrong on our end. Please try again.")
