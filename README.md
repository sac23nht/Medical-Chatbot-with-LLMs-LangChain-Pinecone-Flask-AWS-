# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

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


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# optional, defaults to "medical-chatbot"
# PINECONE_INDEX_NAME = "medical-chatbot"
```


```bash
# run the following command ONCE to store embeddings to pinecone
# (the app cannot answer anything until this index exists)
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
- GPT
- Pinecone



# Deployment (Docker: Render, Hugging Face Spaces, any container host)

The `Dockerfile` builds a production image served by gunicorn on `$PORT`
(default 8080). It downloads the models at build time, so start-up does not
need to fetch ~1 GB from the Hugging Face Hub.

```bash
docker build -t medibot .
docker run -p 8080:8080 -e PINECONE_API_KEY=your-key medibot
```

**Environment variables** (set them as secrets on your host, never commit them):

| Variable | Required | Notes |
|---|---|---|
| `PINECONE_API_KEY` | yes | The Pinecone index must already be populated (`python src/store_index.py`) |
| `PINECONE_INDEX_NAME` | no | Defaults to `medical-chatbot` |
| `PORT` | no | Set automatically by most hosts |

**Memory: at least 2 GB of RAM is required.** The app loads `flan-t5-base`
plus an embedding model and uses about 1.3 GB after the first question.
It will be killed on 512 MB free tiers.

- **Hugging Face Spaces (free, 16 GB RAM):** create a *Docker* Space and add
  this to the top of the Space's `README.md`:
  ```yaml
  ---
  title: Medical Chatbot
  sdk: docker
  app_port: 8080
  ---
  ```
  then add `PINECONE_API_KEY` under Settings > Secrets.
- **Render:** New Web Service > Language *Docker* > an instance with 2 GB or
  more RAM. Set the health check path to `/health`.

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - PINECONE_API_KEY
   - OPENAI_API_KEY