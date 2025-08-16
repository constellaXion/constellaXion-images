import os
from contextlib import asynccontextmanager
from unsloth import FastLanguageModel  # Must be first non-standard import
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from constellaxion_utils.gcs.tools import ModelManager

# Prevent TorchDynamo crashes and disable autotuning warnings
torch._dynamo.config.suppress_errors = True
torch._inductor.config.max_autotune_gemm = False

# === CONFIGURATION ===
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
DTYPE = os.getenv("DTYPE")
MODEL_NAME = os.getenv("MODEL_NAME")

# === Prepare model checkpoint ===
model_manager = ModelManager()
LOCAL_MODEL_PATH = os.path.join(os.getcwd(), "model")
model_manager.get_model(GCS_BUCKET_NAME, f"{MODEL_NAME}/model", LOCAL_MODEL_PATH)

# === Request Schemas ===
class PromptInstance(BaseModel):
    prompt: str
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 1024
    request_id: str = "0"

class PromptRequest(BaseModel):
    instances: list[PromptInstance]

# === Lifespan Context Manager (modern startup/shutdown handling) ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Initializing vLLM engine...")
    engine_args = AsyncEngineArgs(
        model=LOCAL_MODEL_PATH,
        tokenizer=LOCAL_MODEL_PATH,
        trust_remote_code=True,
        dtype="auto" if not DTYPE or DTYPE == "None" else DTYPE,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        tokenizer_mode="auto",
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    print("Engine args:", engine_args)
    app.state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    yield
    print("Shutting down: cleaning up vLLM engine...")
    app.state.engine.shutdown()  # Graceful cleanup

# === Initialize FastAPI ===
app = FastAPI(lifespan=lifespan)

# === Routes ===
@app.post("/predict")
async def predict(req: PromptRequest):
    """Generate predictions for prompts."""
    eng: AsyncLLMEngine = app.state.engine
    results = []
    for instance in req.instances:
        sampling_params = SamplingParams(
            temperature=instance.temperature,
            top_k=instance.top_k,
            top_p=instance.top_p,
            max_tokens=instance.max_tokens,
        )
        final_text = ""
        async for output in eng.generate(
            instance.prompt,
            sampling_params,
            request_id=instance.request_id,
        ):
            if output.finished:
                final_text = output.outputs[0].text
        results.append({"response": final_text})
    return {"predictions": results}

@app.get("/health")
def health_check():
    return {"status": "running", "model": MODEL_NAME}

# === Entrypoint ===
if __name__ == "__main__":
    # Run the FastAPI app
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)