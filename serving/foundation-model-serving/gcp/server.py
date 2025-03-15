import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

# === CONFIGURE MODEL ===
MODEL_NAME = os.getenv("MODEL_NAME")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded!")


# === FASTAPI SERVER ===
app = FastAPI()


class VertexRequest(BaseModel):
    instances: List[Dict]  # List of input dictionaries


@app.post("/predict")
def predict(request: VertexRequest):
    """
    Vertex AI-compatible endpoint.
    Expects JSON: { "instances": [ {"text": "input_text"} ] }
    Returns JSON: { "predictions": ["generated_text"] }
    """
    texts = [instance["text"] for instance in request.instances]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )

    generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    return {"predictions": generated_texts}


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "running", "model": MODEL_NAME}