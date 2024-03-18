# Chatbot

This chatbot is a simple `langchain` implementation that can quote the lithuanian folk poem "Du gaideliai".

# Running locally

First, you will need a hugging face api key. Either set it to environment variable or pass it as an argument to the `LLM` variable initialization in `config.py`.

Then you can use docker to run the chatbot:

```bash
docker build -t chatbot .
docker run -p 8080:800 --name chatbot -d chatbot
```

You should be able to access the chatbot at `http://localhost:8080`

# Evaluation

The chatbot is evaluated using the `eval.py` script.
It uses a simple cosine similarity check to see if the response to the question `Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'` is similar to the expected response.

To run it:

```bash
docker exec -it chatbot python /app/eval.py
```