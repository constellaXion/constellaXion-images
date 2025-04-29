import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

# Prevent any TorchDynamo compile attempts from crashing
torch._dynamo.config.suppress_errors = True

# Disable GEMM autotuning to avoid SM-related warnings
torch._inductor.config.max_autotune_gemm = False

# === CONFIGURATION ===
MODEL_NAME = os.getenv("MODEL_NAME")
DTYPE = os.getenv("DTYPE")
HF_TOKEN = os.getenv("HF_TOKEN")

# === Initialize FastAPI ===
app = FastAPI()

# === Initialize vLLM Engine ===
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    trust_remote_code=True,
    dtype="auto" if not DTYPE or DTYPE == "None" else DTYPE,
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    tokenizer_mode="auto",
)
engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

# === Request Schema ===
class PromptInstance(BaseModel):
    """Schema for a single prompt instance with generation parameters."""
    prompt: str
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 100
    request_id: str = "0"

class PromptRequest(BaseModel):
    """Schema for a batch of prompt instances."""
    instances: list[PromptInstance]

class DefaultVLLMInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    """Handler for vLLM model inference."""
    
    def default_model_fn(self, model_dir):
        """Loads the vLLM engine."""
        return engine

    def default_input_fn(self, input_data, content_type):
        """Processes input data into the expected format."""
        if content_type == content_types.JSON:
            return decoder.decode(input_data, content_type)
        raise errors.UnsupportedFormatError(content_type)

    def default_predict_fn(self, data, model):
        """Generates text predictions using the vLLM engine."""
        results = []
        for instance in data.instances:
            sampling_params = SamplingParams(
                temperature=instance.temperature,
                top_k=instance.top_k,
                top_p=instance.top_p,
                max_tokens=instance.max_tokens,
            )
            final_text = ""
            async for output in model.generate(
                instance.prompt,
                sampling_params,
                request_id=instance.request_id
            ):
                if output.finished:
                    final_text = output.outputs[0].text
            results.append({"prediction": final_text})
        return {"predictions": results}

    def default_output_fn(self, prediction, accept):
        """Formats the prediction output."""
        if accept == content_types.JSON:
            return encoder.encode(prediction, accept)
        raise errors.UnsupportedFormatError(accept)

# === /predict Endpoint ===
@app.post("/predict")
async def predict(req: PromptRequest):
    """Generate text predictions for given prompts using the LLM engine."""
    handler = DefaultVLLMInferenceHandler()
    model = handler.default_model_fn(None)
    input_data = handler.default_input_fn(req, content_types.JSON)
    prediction = handler.default_predict_fn(input_data, model)
    return handler.default_output_fn(prediction, content_types.JSON)

# === /health Endpoint ===
@app.get("/health")
def health_check():
    """Return server health status and model information."""
    return {"status": "running", "model": MODEL_NAME} 