import os
import torch
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    model_server,
    transformer,
)
from sagemaker_inference.default_handler_service import DefaultHandlerService
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# Prevent any TorchDynamo compile attempts from crashing
torch._dynamo.config.suppress_errors = True

# Disable GEMM autotuning to avoid SM-related warnings
torch._inductor.config.max_autotune_gemm = False

# === CONFIGURATION ===
MODEL_NAME = os.getenv("MODEL_NAME")
DTYPE = os.getenv("DTYPE")
HF_TOKEN = os.getenv("HF_TOKEN")

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

class VLLMInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    """Handler for vLLM model inference."""
    def default_model_fn(self, model_dir):
        """Loads the vLLM engine.
        
        Args:
            model_dir: a directory where model is saved (not used in this case)
            
        Returns:
            The initialized vLLM engine
        """
        return engine

    def default_input_fn(self, input_data, content_type):
        """Processes input data into the expected format.
        
        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type
            
        Returns:
            input_data deserialized into the expected format
        """
        if content_type == content_types.JSON:
            return decoder.decode(input_data, content_type)
        raise errors.UnsupportedFormatError(content_type)

    def default_predict_fn(self, data, model):
        """Generates text predictions using the vLLM engine.
        
        Args:
            data: input data for prediction deserialized by input_fn
            model: vLLM engine loaded in memory by model_fn
            
        Returns:
            a prediction result
        """
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
        """Formats the prediction output.
        
        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized
            
        Returns:
            output data serialized
        """
        if accept == content_types.JSON:
            return encoder.encode(prediction, accept)
        raise errors.UnsupportedFormatError(accept)

class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    """
    def __init__(self):
        """Initialize the handler service with the vLLM inference handler."""
        transformer_obj = transformer.Transformer(
            default_inference_handler=VLLMInferenceHandler()
        )
        super(HandlerService, self).__init__(transformer=transformer_obj)

def main():
    """Start the model server."""
    model_server.start_model_server(handler_service=HandlerService())

if __name__ == "__main__":
    main()
