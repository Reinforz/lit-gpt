import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import wandb
from huggingface_hub import HfApi, login
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning.pytorch.loggers import WandbLogger

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.adapter_v2 import (
    GPT,
    Block,
    Config,
    adapter_filter,
    mark_only_adapter_v2_as_trainable,
)
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt
from utils.discord import send_embedded_message

# Training settings
eval_interval = 10
save_interval = 10
eval_iters = 100
eval_max_new_tokens = 350
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 2e-4
batch_size = 128 / devices
micro_batch_size = 1  # set to 2 because this is fit into 12GB Vram
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_seq_length = None  # assign value to truncate
max_iters = 10000
weight_decay = 0.01
warmup_steps = 100

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}

# Hugging Face Hub login
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
# set up wandb... ensure WANDB_API_KEY env variable is set
wandb.login()

api = HfApi()


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/adapter_v2/alpaca"),
    precision: Optional[str] = None,
    repo_id: str = "models/model",
) -> None:
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        **{"config": hparams},
        name=f"{repo_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    precision = precision or get_default_supported_precision(training=True)

    fabric_devices = devices
    if fabric_devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(
        out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval
    )
    fabric = L.Fabric(
        devices=fabric_devices,
        strategy=strategy,
        precision=precision,
        loggers=[logger, wandb_logger],
    )
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, repo_id)


def main(
    fabric: L.Fabric,
    data_dir: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    repo_id: str,
) -> None:
    try:
        check_valid_checkpoint_dir(checkpoint_dir)

        fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

        if fabric.global_rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        train_data = torch.load(data_dir / "train.pt")
        val_data = torch.load(data_dir / "test.pt")

        config = Config.from_name(name=checkpoint_dir.name)
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
        with fabric.init_module(empty_init=False):
            model = GPT(config)
        checkpoint = lazy_load(checkpoint_path)
        # strict=False because missing keys due to adapter weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

        mark_only_adapter_v2_as_trainable(model)

        fabric.print(
            f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
        )
        fabric.print(
            f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}"
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )
        model, optimizer = fabric.setup(model, optimizer)

        fabric.seed_everything(1337 + fabric.global_rank)

        train_time = time.perf_counter()
        train(
            fabric,
            model,
            optimizer,
            train_data,
            val_data,
            checkpoint_dir,
            out_dir,
            repo_id,
        )
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if fabric.device.type == "cuda":
            fabric.print(
                f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
            )

        # Save the final checkpoint at the end of training
        save_path = out_dir / "lit_model_adapter_finetuned.pth"
        save_adapter_v2_checkpoint(fabric, model, save_path)
        api.upload_folder(
            folder_path=out_dir,
            path_in_repo="./",
            repo_id=str(repo_id),
            repo_type="model",
        )

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        fabric.print(e)
        fabric.print(stack_trace)
        send_embedded_message(
            f"Training Error: {repo_id}",
            {
                "error_message": f"{error_message}",
                "stack_trace": f"{stack_trace}",
            },
            mentionTeam=True,
        )
        raise e


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    repo_id: str,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    validate(fabric, model, val_data, tokenizer, iters=2)  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    loss_prev = 1
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in range(1, max_iters + 1):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_ix if iter_num == 1 else None
        )

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            # LOG to discord here
            send_embedded_message(
                f"Training Eval: {repo_id}",
                {
                    "iteration": f"{iter_num}",
                    "step": f"{step_count}",
                    "loss": f"{loss.item():.4f}",
                    "loss_diff": f"{loss.item() - loss_prev:.4f}",
                },
            )
            wandb.log({"train_loss": loss.item(), "train_step": step_count})
            loss_prev = loss.item()

        total_lengths += input_ids.numel()
        if iter_num % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * micro_batch_size,
                lengths=total_lengths,
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, iters=eval_iters)
            wandb.log({"val_loss": val_loss, "train_step": step_count})
            t1 = time.perf_counter() - t0
            fabric.print(
                f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms"
            )
            send_embedded_message(
                f"Training Eval: {repo_id}",
                {
                    "step": f"{iter_num}",
                    "val loss": f"{val_loss.item():.4f}",
                    "val time": f"{t1 * 1000:.2f}ms",
                },
            )
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"step-{step_count:03d}-ckpt.pth"
            save_adapter_v2_checkpoint(fabric, model, checkpoint_path)
            api.upload_folder(
                folder_path=out_dir,
                path_in_repo="./",
                repo_id=repo_id,
                repo_type="model",
            )
    send_embedded_message(
        f"Training Complete: {repo_id}", "Eval training", mentionTeam=True
    )


# the adapter "kv cache" cannot be initialized under `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_data: List[Dict],
    tokenizer: Tokenizer,
    iters: int,
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )
    val_loss = losses.mean()

    # produce an example:
    instruction = (
        "Recommend a movie for me to watch during the weekend and explain the reason."
    )
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model,
        encoded,
        max_returned_tokens=len(encoded) + eval_max_new_tokens,
        temperature=0.8,
        eos_id=tokenizer.eos_id,
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_adapter_v2_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving adapter v2 weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    if not HUGGINGFACE_TOKEN:
        raise ValueError(
            "Please set the HUGGINGFACE_TOKEN environment variable to your Hugging Face API token."
        )

    if not WANDB_API_KEY:
        raise ValueError(
            "Please set the WANDB_API_KEY environment variable to your Weights & Biases API key."
        )

    if not WANDB_PROJECT:
        raise ValueError(
            "Please set the WANDB_PROJECT environment variable to your Weights & Biases project name."
        )

    CLI(setup)
