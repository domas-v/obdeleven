from operator import itemgetter
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from config import DB, PROMPT, RETRIEVER, LLM, POEM_DOCS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


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
        DB.add_documents(POEM_DOCS)

    chain = (
        {
            "context": itemgetter("input") | RETRIEVER,
            "chat_history": lambda x: x["chat_history"],
            "input": itemgetter("input"),
        }
        | PROMPT
        | LLM
        | StrOutputParser()
    )

    logger.info("Running the chain")
    answer = chain.invoke({"input": query.question, "chat_history": CHAT_HISTORY})
    logger.info(f"Answer: {answer}")

    CHAT_HISTORY.append((HumanMessage(query.question), AIMessage(answer)))

    return {"answer": answer}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
