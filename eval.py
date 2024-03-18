from config import CHAIN, EMBEDDING_FUNCTION
from numpy import dot
from numpy.linalg import norm

question = "Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'"

response = CHAIN.invoke({"input": question, "chat_history": []})

answer = response["response"]
context = response["context"][0].page_content

answer_embeddings = EMBEDDING_FUNCTION.embed_query(answer)
context_embeddings = EMBEDDING_FUNCTION.embed_query(context)

cos_sim = dot(answer_embeddings, context_embeddings) / (
    norm(answer_embeddings) * norm(context_embeddings)
)

assert cos_sim > 0.8
