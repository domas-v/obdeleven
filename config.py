import os
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


LLM = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_TOKEN"),
    )
)

POEM = TextLoader("./poem.txt").load()
EMBEDDING_FUNCTION = HuggingFaceHubEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2"
)
DB = Chroma(persist_directory="./chroma", embedding_function=EMBEDDING_FUNCTION)

RETRIEVER = DB.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 1},
)

PROMPT = ChatPromptTemplate.from_template(
    """
<s>[INST]
Answer the question in the style of an old farmer based on the chat history:
{chat_history}
[/INST] </s>
[INST]
Recite the lithuanian folk poem 'Du gaideliai' fully if asked either in english or lithuanian. Do not translate it to english
{context}
[/INST]
[INST]
Othwerise, ignore it and answer the question:
{input}
[/INST] """
)

CHAIN = (
    {
        "context": itemgetter("input") | RETRIEVER,
        "chat_history": lambda x: x["chat_history"],
        "input": itemgetter("input"),
    }
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": PROMPT | LLM | StrOutputParser(), "context": itemgetter("context")}
)
