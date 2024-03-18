from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
)
from ragas import evaluate
from datasets import Dataset
from config import CHAIN, POEM


question = "Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'"

response = CHAIN.invoke({"input": question, "chat_history": []})

answer = response["response"]
context = [context.page_content for context in response["context"]]

response_dataset = Dataset.from_dict(
    {
        "question": [question],
        "answer": [answer],
        "contexts": [context],
        "ground_truth": [POEM[0].page_content],
    }
)


metrics = [
    faithfulness,
    context_precision,
    answer_correctness,
]

results = evaluate(response_dataset, metrics)

assert results["faithfulness"] >= 0.9
assert results["context_precision"] >= 0.9
assert results["answer_correctness"] >= 0.9
