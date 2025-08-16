import os
import subprocess
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from contextlib import asynccontextmanager

# Prevent any TorchDynamo compile attempts from crashing
torch._dynamo.config.suppress_errors = True

# Disable GEMM autotuning to avoid SM-related warnings
torch._inductor.config.max_autotune_gemm = False

# === CONFIGURATION ===
MODEL_NAME = os.getenv("MODEL_NAME")
DTYPE = os.getenv("DTYPE")
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"DTYPE: {DTYPE}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log GPU information using nvidia-smi at server startup."""
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print("nvidia-smi output:\n", output)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Error running nvidia-smi:", e)
    yield

# === Initialize FastAPI ===
app = FastAPI(lifespan=lifespan)

# === Initialize vLLM Engine ===
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    trust_remote_code=True,
    dtype="auto" if not DTYPE or DTYPE == "None" else DTYPE,
    gpu_memory_utilization=0.9,
    enforce_eager=True,
    # max_num_batched_tokens=2048,
    # max_num_seqs=64,
    # pipeline_parallel_size=2,
    tokenizer_mode="auto",
)
print(engine_args)
engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

# === Request Schema ===
class PromptInstance(BaseModel):
    """Schema for a single prompt instance with generation parameters."""
    prompt: str
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 100  # Added default value
    request_id: str = "0"

class PromptRequest(BaseModel):
    """Schema for a batch of prompt instances."""
    instances: list[PromptInstance]


# def check_shm_size():
#     """Check and print the size of shared memory (/dev/shm)."""
#     stat = os.statvfs("/dev/shm")
#     size = stat.f_frsize * stat.f_blocks  # in bytes
#     print(f"/dev/shm size: {size / (1024*1024)} MB")

# check_shm_size()
# === /predict Endpoint: Returns Final Generated Text ===
@app.post("/predict")
async def predict(req: PromptRequest):
    """Generate text predictions for given prompts using the LLM engine."""
    prompt = req.instances[0].prompt if req.instances else ""
    sampling_params = SamplingParams(
        temperature=req.instances[0].temperature if req.instances else 0.7,
        top_k=req.instances[0].top_k if req.instances else 50,
        top_p=req.instances[0].top_p if req.instances else 0.9,
        max_tokens=req.instances[0].max_tokens if req.instances else 100,
    )
    request_id = req.instances[0].request_id if req.instances else "0"
    async def token_stream():
        async for output in engine.generate(
            prompt,
            sampling_params,
            request_id=request_id
        ):
            response = output.outputs[0].text
            yield f"data: {response}"
        yield "data: [DONE]"
    return StreamingResponse(token_stream(), media_type="text/event-stream")
# === /health Endpoint: Simple Health Check ===
@app.get("/health")
def health_check():
    """Return server health status and model information."""
    return {"status": "running", "model": MODEL_NAME}