# Medical Chatbot with LLMs, LangChain, Pinecone and Flask

# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/sac23nht/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-.git
cd Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
(`requirements.txt` is the lightweight set the web app needs. Building the
Pinecone index once needs the heavier `requirements-indexing.txt`, see below.)


### Create a `.env` file in the root directory with your credentials:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Free token from https://huggingface.co/settings/tokens
# (fine-grained, tick "Make calls to Inference Providers").
# Used for the hosted embedding model and, by default, the LLM.
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Optional
# PINECONE_INDEX_NAME = "medical-chatbot"
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# LLM_BASE_URL = "https://router.huggingface.co/v1"   # any OpenAI-compatible API
# LLM_API_KEY = ""                                     # defaults to HF_TOKEN
```


```bash
# Run ONCE to store the book's embeddings in Pinecone (the app cannot answer
# anything until this index exists). This is the only step that needs torch.
pip install -r requirements-indexing.txt
python src/store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Pinecone (vector database)
- Hugging Face Inference API (`all-MiniLM-L6-v2` embeddings)
- Any OpenAI-compatible LLM API (Llama 3.1 8B via Hugging Face by default)



# Deployment (Docker: Render, Railway, Fly.io, any container host)

The models run on hosted APIs, so the app itself is light: about 150 MB of RAM,
no GPU. That fits free tiers with 512 MB (Render, ...).

```bash
docker build -t medibot .
docker run -p 8080:8080 -e PINECONE_API_KEY=... -e HF_TOKEN=... medibot
```

**Environment variables** (set them as secrets on your host, never commit them):

| Variable | Required | Notes |
|---|---|---|
| `PINECONE_API_KEY` | yes | The index must already be populated (`python src/store_index.py`) |
| `HF_TOKEN` | yes | Hugging Face token with "Make calls to Inference Providers" |
| `LLM_MODEL` | no | Default `meta-llama/Llama-3.1-8B-Instruct` |
| `LLM_BASE_URL` | no | Default `https://router.huggingface.co/v1`. Point it at Groq, OpenRouter, OpenAI, ... to switch provider |
| `LLM_API_KEY` | no | Key for `LLM_BASE_URL`; defaults to `HF_TOKEN` |
| `PINECONE_INDEX_NAME` | no | Default `medical-chatbot` |
| `PORT` | no | Set automatically by most hosts |

**Render:** New Web Service > connect this repo > Language *Docker* > Instance
type *Free* > add the two required variables under Environment > set the
health check path to `/health`. Free instances sleep when idle, so the first
request after a pause takes about a minute.

Note: the free Hugging Face token has a limited monthly allowance for hosted
inference. For heavier use, switch the LLM to another provider with the three
`LLM_*` variables above.
