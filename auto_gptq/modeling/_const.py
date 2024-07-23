import torch

from ..utils.import_utils import compare_transformers_version


CPU = torch.device("cpu")
device_memory = []
for i in range(torch.cuda.device_count()):
    device_memory.append(torch.cuda.mem_get_info(i)[0])
CUDA_0 = torch.device(f"cuda:{device_memory.index(max(device_memory))}")

SUPPORTED_MODELS = [
    "bloom",
    "gptj",
    "gpt2",
    "gpt_neox",
    "opt",
    "moss",
    "gpt_bigcode",
    "codegen",
    "RefinedWebModel",
    "RefinedWeb",
    "baichuan",
    "internlm",
    "qwen",
    "xverse",
    "deci",
    "stablelm_epoch",
    "mpt",
    "cohere",
]
if compare_transformers_version("v4.28.0", op="ge"):
    SUPPORTED_MODELS.append("llama")
if compare_transformers_version("v4.30.0", op="ge"):
    SUPPORTED_MODELS.append("longllama")
if compare_transformers_version("v4.33.0", op="ge"):
    SUPPORTED_MODELS.append("falcon")
if compare_transformers_version("v4.34.0", op="ge"):
    SUPPORTED_MODELS.append("mistral")
    SUPPORTED_MODELS.append("Yi")
if compare_transformers_version("v4.36.0", op="ge"):
    SUPPORTED_MODELS.append("mixtral")
if compare_transformers_version("v4.37.0", op="ge"):
    SUPPORTED_MODELS.append("qwen2")
    SUPPORTED_MODELS.append("phi")
if compare_transformers_version("v4.38.0", op="ge"):
    SUPPORTED_MODELS.append("gemma")
if compare_transformers_version("v4.39.0.dev0", op="ge"):
    SUPPORTED_MODELS.append("starcoder2")

EXLLAMA_DEFAULT_MAX_INPUT_LENGTH = 2048

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS", "EXLLAMA_DEFAULT_MAX_INPUT_LENGTH"]
