import sys
import os
import time
from typing import List, Optional, Literal, Tuple
from pathlib import Path
import json

from jsonargparse import CLI
import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, merge_lora_weights
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    gptq_quantization,
    lazy_load,
)
from scripts.prepare_alpaca import generate_prompt

from datasets import Dataset
from huggingface_hub import snapshot_download
from utils.discord import send_embedded_message


lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True
lora_head = True


def infer(
    data_dir: Path = Path("data/test.json"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    lora_repo: str = "reinforz/lora-alpaca",
    model_name: str = "stablelm-base-alpha-3b",
    lora_dir: Optional[Path] = None,
    resume_index: Optional[int] = 0,
) -> None:
    """Generates a dataset of responses for the given test data prompts and saves it to Huggingface.

    Args:
        test_data (Path): Path to the test data file.
        model_dir (Path): Path to the model checkpoint directory.
        lora_dir (Path): Path to the lora checkpoint directory.
        model_name (str): Name of the model to be .
    """

    token = os.getenv("HUGGINGFACE_TOKEN")

    with open(data_dir, "r", encoding="utf-8") as file:
        data = json.load(file)
        data = data[resume_index + 1 :]

    if not lora_dir:
        lora_dir = Path(f"out/lora/{model_name}")
        snapshot_download(
            repo_id=lora_repo,
            local_dir=lora_dir,
            token=token,
        )

    results = []  # list of dicts

    model, tokenizer, fabric, max_return_token = setup_model(
        lora_path=lora_dir,
        checkpoint_dir=checkpoint_dir,
        quantize="bnb.nf4",
        max_new_tokens=500,
        precision="bf16-true",
        data=data,
    )

    total_samples = len(data)

    for i, sample in enumerate(data):
        prompt, response = infer_sample(
            prompt=sample["instruction"],
            input=sample["input"],
            max_returned_tokens=max_return_token,
            tokenizer=tokenizer,
            model=model,
            fabric=fabric,
            top_k=200,
            temperature=0.8,
        )
        results.append({"prompt": prompt, "response": response})
        if (i + 1) % 25 == 0:
            dataset = Dataset.from_dict(results)
            dataset.push_to_hub(f"reinforz/{model_name}-inference", token=token)

            send_embedded_message("Inference", f"Finished {i+1}/{total_samples}.")
    dataset = Dataset.from_dict(results)

    dataset.push_to_hub(f"reinforz/{model_name}-inference", token=token)

    send_embedded_message("Inference", "Completed.", mentionTeam=True)


def infer_sample(
    prompt: str,
    input: str,
    max_returned_tokens: int,
    tokenizer: Tokenizer,
    model: GPT,
    fabric: L.Fabric,
    top_k: int = 200,
    temperature: float = 0.8,
) -> Tuple[str, str]:
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    # max_returned_tokens = prompt_length + max_new_tokens

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    response = tokenizer.decode(y)
    response = response.split("### Response:")[1].strip()
    # fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(
        f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
        file=sys.stderr,
    )
    if fabric.device.type == "cuda":
        fabric.print(
            f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
            file=sys.stderr,
        )

    return prompt, response


def setup_model(
    lora_path: Path,
    checkpoint_dir: Path,
    quantize: Optional[
        Literal[
            "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"
        ]
    ],
    max_new_tokens: int,
    precision: str,
    data: List[dict],
) -> Tuple[GPT, Tokenizer, L.Fabric, int]:
    """Generates a response based on a given instruction and an optional input."""
    # precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    tokenizer = Tokenizer(checkpoint_dir)

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(
        quantize == "gptq.int4"
    ):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    max_returned_tokens = get_max_length(data, tokenizer, max_new_tokens, fabric)

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    merge_lora_weights(model)
    model = fabric.setup(model)

    return model, tokenizer, fabric, max_returned_tokens


def get_max_length(
    data: List[dict], tokenizer: Tokenizer, max_new_tokens: int, fabric: L.Fabric
) -> int:
    max_seq_length = 0
    for sample in data:
        instruction = sample["instruction"]
        user_input = sample["input"]
        prompt_dict = {
            "instruction": instruction,
            "input": user_input,
        }
        prompt = generate_prompt(prompt_dict)
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        max_seq_length = max(max_seq_length, max_returned_tokens)
    return max_seq_length


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    CLI(infer)
