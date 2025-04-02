import os
import getpass
from fastapi.responses import StreamingResponse
from ibm_watsonx_ai.foundation_models import Model
import subprocess
import gzip
import json
import chromadb
import random
import string
from pydantic import BaseModel
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Response,status
import uvicorn
import json
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()
origins = ["*"]

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.getenv("API_KEY")
	}

class MessageInput(BaseModel):
    question: str


project_id = os.getenv("PROJECT_ID")



def get_vector_index():
    project_id = os.getenv("PROJECT_ID")
    wml_credentials = get_credentials()
    client = APIClient(credentials=wml_credentials, project_id=project_id)

    #vector_index_id = "4c41ae0f-1a32-4e9b-8126-449ac0fac50d"
    #vector_index_id = "995d66a3-65b1-4af6-9417-14105beecf35"
    #vector_index_id = "7846c928-d7c7-40de-8ac6-7e103d121f29"
    #vector_index_id = "5e2b8fb7-2d77-44a9-9b2d-f2b85e70dd88"
    #vector_index_id = "67886594-dc8d-4583-8b2b-afd9b5c30f40"
    vector_index_id = "814deecd-4a71-48fe-9e4e-9f7917ec5e09"
    print(project_id)
    print(vector_index_id)
    vector_index_details = client.data_assets.get_details(vector_index_id)
    vector_index_properties = vector_index_details["entity"]["vector_index"]

    return vector_index_properties,vector_index_id,client

def hydrate_chromadb():
    vector_index_properties,vector_index_id,client = get_vector_index()
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = str(content, "utf-8")
    vectors = json.loads(stringified_vectors)

    chroma_client = chromadb.Client()

    # make sure collection is empty if it already existed
    collection_name = "my_collection"
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print("Collection didn't exist.")
        collection = chroma_client.create_collection(name=collection_name)

        vector_embeddings = []
        vector_documents = []
        vector_metadatas = []
        vector_ids = []

        for vector in vectors:
            vector_embeddings.append(vector["embedding"])
            vector_documents.append(vector["content"])
            metadata = vector["metadata"]
            lines = metadata["loc"]["lines"]
            clean_metadata = {}
            clean_metadata["asset_id"] = metadata["asset_id"]
            clean_metadata["asset_name"] = metadata["asset_name"]
            clean_metadata["url"] = metadata["url"]
            clean_metadata["from"] = lines["from"]
            clean_metadata["to"] = lines["to"]
            vector_metadatas.append(clean_metadata)
            asset_id = vector["metadata"]["asset_id"]
            random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            id = "{}:{}-{}-{}".format(asset_id, lines["from"], lines["to"], random_string)
            vector_ids.append(id)

        collection.add(
            embeddings=vector_embeddings,
            documents=vector_documents,
            metadatas=vector_metadatas,
            ids=vector_ids
        )
    return collection, vector_index_properties

def get_chroma_collection():
    chroma_collection,vector_index_properties = hydrate_chromadb()
    return chroma_collection, vector_index_properties


def proximity_search( question ):
    chroma_collection, vector_index_properties = get_chroma_collection()
    wml_credentials = get_credentials()
    emb = Embeddings(
    model_id=vector_index_properties["settings"]["embedding_model_id"],
    credentials=wml_credentials,
    project_id=project_id,
    params={
        "truncate_input_tokens": 128
    }
)
    query_vectors = emb.embed_query(question)
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )

    documents = list(reversed(query_result["documents"][0]))
    
    return "\n".join(documents)



@app.post("/call_watsonx")
async def call_watsonx(input:MessageInput):

    question = input.question
    model_id = "meta-llama/llama-3-3-70b-instruct"
    #llama-3-3-70b-instruct

    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}
    model = Model(
	model_id = model_id,
	params = parameters,
	credentials = get_credentials(),
	project_id = project_id
	)
    print("Fetching answer")
    grounding = proximity_search(question)

    prompt_input = f"""<|start_header_id|>system<|end_header_id|>

You always answer the questions with markdown formatting. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.

Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.

Given the document and the current conversation between a user and an assistant, your task is as follows: answer any user query by using information from the document. Always answer as helpfully as possible, while being safe. When the question cannot be answered using the context or document, output the following response: "I cannot answer that question based on the provided document.".
Always answer only to the question asked. Do not provide unnecessary details or unrelated information.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>

__grounding__"""
    
    formattedQuestion = f"""<|begin_of_text|><|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    prompt = f"""{prompt_input}{formattedQuestion}"""
    print("LLM generation ongoing")
    generated_response = model.generate_text(prompt=prompt.replace("__grounding__", grounding), guardrails=False)
    print("LLM generation completed")
    return {"response":generated_response}
    # return StreamingResponse(model.generate_text_stream(prompt=prompt.replace("__grounding__", grounding)), media_type="text/event-stream")

if __name__ == "__main__":
	uvicorn.run(app, host="localhost", port=8001)
    #uvicorn.run(app, host="0.0.0.0", port=8501)