import torch
import os
from fastapi import FastAPI
from transformers import pipeline
from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 256
    temperature: float = 0.7

class ChatResponse(BaseModel):
    messages: List[Dict[str, str]]
    

def authenticate():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise KeyError("Huggingface token not set in .env file as HF_TOKEN")
    login(token=hf_token)

def get_llama_model_pipeline():
    authenticate()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Using device:", device)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    return pipe

llama_pipe = get_llama_model_pipeline()
app = FastAPI()

@app.post("/chat/")
async def chat(request: ChatRequest):
    response = llama_pipe(
        text_inputs = request.messages, 
        max_new_tokens = request.max_new_tokens,
        temperature = request.temperature
        )
    
    return ChatResponse(messages=response[0]['generated_text'])
