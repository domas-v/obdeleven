from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate


LLM = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token="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",
    )
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
    search_kwargs={"score_threshold": 0.3, "k": 1},
)

PROMPT = ChatPromptTemplate.from_template(
    """
<s>[INST]
The lithuanian folk poem 'Du gaideliai' goes as follows. Cite it in full if asked:
{context}
[/INST] </s>
[INST] Answer the question based on the chat history:
{chat_history}
[/INST]
[INST]
{input}
[/INST] """
)
