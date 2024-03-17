from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma


LLM = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token="",
)

POEM_DOCS = TextLoader("./poem.txt").load()
DB = Chroma(
    persist_directory="./chroma",
    embedding_function=HuggingFaceHubEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    ),
)

RETRIEVER = DB.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4, "k": 1},
)


TEMPLATE = """ <s>[INST] The lithuanian folk poem 'Du gaideliai' goes as follows. ONLY cite (without translation) it in full if asked EXPLICITLY:\n{context} [/INST]  </s> [INST] Answer the question: {question} [/INST] """  # noqa