from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts import PromptTemplate
from config import DB, RETRIEVER, LLM, POEM_DOCS, TEMPLATE


app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/ask/")
def ask(query: Query):
    # question = "Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'"
    question = query.question

    if DB._collection.count() == 0:
        DB.add_documents(POEM_DOCS)

    prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "query"])
    chain = RetrievalQA.from_chain_type(
        llm=ChatHuggingFace(llm=LLM),
        retriever=RETRIEVER,
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    answer = chain({"query": question})
    return {"answer": answer["result"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
