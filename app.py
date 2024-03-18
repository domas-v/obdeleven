from operator import itemgetter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from config import DB, RETRIEVER, LLM, POEM_DOCS, TEMPLATE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

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
def ask(query: Query) -> dict:
    logger.info(f"Question: {query.question}")

    if DB._collection.count() == 0:
        logger.info("Document not found. Adding documents to the database")
        DB.add_documents(POEM_DOCS)

    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = ChatHuggingFace(llm=LLM)
    chain = (
        {
            "context": itemgetter("input") | RETRIEVER,
            "chat_history": lambda x: x["chat_history"],
            "input": itemgetter("input"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Running the chain")
    answer = chain.invoke({"input": query.question, "chat_history": CHAT_HISTORY})
    logger.info(f"Answer: {answer}")

    CHAT_HISTORY.append((HumanMessage(query.question), AIMessage(answer)))

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
