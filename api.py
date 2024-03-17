from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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


class Query(BaseModel):
    question: str


@app.post("/ask/")
def ask(query: Query) -> dict:
    logger.info(f"Question: {query.question}")

    if DB._collection.count() == 0:
        logger.info("Document not found. Adding documents to the database")
        DB.add_documents(POEM_DOCS)

    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "query"])
    logger.info(f"Prompt: {prompt.template}")

    llm = ChatHuggingFace(llm=LLM)
    chain = (
        {"context": RETRIEVER, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("Running the chain")
    answer = chain.invoke(query.question)
    logger.info(f"Answer: {answer}")

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
