import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from langchain_core.messages import HumanMessage, AIMessage
from config import DB, CHAIN, POEM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(debug=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHAT_HISTORY = []


class Query(BaseModel):
    question: str


@app.post("/ask/")
def ask(query: Query) -> dict[str, str]:
    logger.info(f"Question: {query.question}")

    if DB._collection.count() == 0:
        logger.info("Poem not found. Adding poem to the database")
        DB.add_documents(POEM)

    logger.info("Running the chain")
    answer = CHAIN.invoke({"input": query.question, "chat_history": CHAT_HISTORY})

    CHAT_HISTORY.append((HumanMessage(query.question), AIMessage(answer["response"])))

    return {"answer": answer["response"]}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get("PORT", 8080))
