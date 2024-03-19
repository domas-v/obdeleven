# Chatbot

This chatbot is a simple `langchain` implementation that can quote the lithuanian folk poem "Du gaideliai".

# Running locally

First, you will need a hugging face api key. Either set it to environment variable or pass it as an argument to the `LLM` variable initialization in `config.py`.

Then you can use docker to run the chatbot:

```bash
docker build -t chatbot .
docker run -p 8080:8080 --name chatbot -d chatbot
```

You should be able to access the chatbot at `http://localhost:8080`

# Evaluation

The chatbot is evaluated using the `eval.py` script.
It uses a simple cosine similarity check to see if the response to the question `Pacituok lietuvių liaudies eilėraštį 'Du gaideliai'` is similar to the expected response.

To run it:

```bash
docker exec -it chatbot python /app/eval.py
```

# Deployment

The chatbot is deployed to Google Cloud Run with a simple `Dockerfile` and a couple of commands.
You will need to set up a project in Google Cloud and install the `gcloud` command line tool, which is outside the scope of this README.

If you're not on linux/amd64, you will need to build the image with the correct architecture:

```bash
docker build -t eu.gcr.io/${PROJECT_ID}/chatbot --platform linux/amd64 .
```

Otherwise, tag the image and push it to the Google Cloud Registry or deploy it to another container registry of your choice.:

```bash
docker tag chatbot eu.gcr.io/${PROJECT_ID}/chatbot
```

Lastly, deploy the container to Google Cloud Run. Or to any other service that support docker containers.

```bash
docker push eu.gcr.io/${PROJECT_ID}/chatbot
gcloud run deploy chatbot --image eu.gcr.io/${PROJECT_ID}/chatbot
```
