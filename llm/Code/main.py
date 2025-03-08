from AILLM import QwenModel
from timer import Timer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import shutil
import torch
import os
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")

timer = Timer()

# Initialize FastAPI app
app = FastAPI()

model = QwenModel()

class InferenceRequest(BaseModel):
    prompts: list[str]

@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/generate")
def generate(request: InferenceRequest):
    responses = model.generate_responses(request.prompts)
    return {"responses": responses}

