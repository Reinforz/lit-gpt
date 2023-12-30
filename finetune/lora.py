import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
import wandb
from huggingface_hub import HfApi, login
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning.pytorch.loggers import WandbLogger

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt
from utils.discord import send_embedded_message

# set up wandb... ensure WANDB_API_KEY env variable is set
wandb.login()

# Training settings
eval_interval = 10
save_interval = 10
eval_iters = 100
eval_max_new_tokens = 350
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 2e-4
batch_size = 128
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_seq_length = None  # assign value to truncate
max_iters = 10000  # train dataset size
weight_decay = 0.01
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = True
lora_head = True
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


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/alpaca"),
    precision: Optional[str] = None,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]
    ] = None,
    repo_id: str = "models/model",
) -> None:
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        **{"config": hparams},
        name=f"{repo_id}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    )

    precision = precision or get_default_supported_precision(training=True)

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

    if devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
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
        devices=devices,
        strategy=strategy,
        precision=precision,
        loggers=[logger, wandb_logger],
        plugins=plugins,
    )
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, repo_id, wandb_logger)


def main(
    fabric: L.Fabric,
    data_dir: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    repo_id: str,
    wandb_logger: WandbLogger,
) -> None:
    try:
        check_valid_checkpoint_dir(checkpoint_dir)

        fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

        if fabric.global_rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        train_data = torch.load(data_dir / "train.pt")
        val_data = torch.load(data_dir / "test.pt")

        if not any(
            (lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)
        ):
            fabric.print("Warning: all LoRA layers are disabled!")
        config = Config.from_name(
            name=checkpoint_dir.name,
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
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
        with fabric.init_module(empty_init=(devices > 1)):
            model = GPT(config)
        mark_only_lora_as_trainable(model)

        fabric.print(
            f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
        )
        fabric.print(
            f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}"
        )

        model = fabric.setup_module(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdamW(
                trainable_params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params, lr=learning_rate, weight_decay=weight_decay
            )
        optimizer = fabric.setup_optimizers(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iters // batch_size
        )

        # strict=False because missing keys due to LoRA weights not contained in state dict
        load_checkpoint(fabric, model, checkpoint_path, strict=False)

        api = HfApi()

        fabric.seed_everything(1337 + fabric.global_rank)

        train_time = time.perf_counter()
        train(
            fabric,
            model,
            optimizer,
            scheduler,
            train_data,
            val_data,
            checkpoint_dir,
            out_dir,
            repo_id,
            api,
            wandb_logger,
        )
        fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if fabric.device.type == "cuda":
            fabric.print(
                f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
            )

        # Save the final LoRA checkpoint at the end of training
        save_path = out_dir / "lit_model_lora_finetuned.pth"
        save_lora_checkpoint(fabric, model, save_path)
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
                "error_message": error_message,
                "stack_trace": stack_trace,
            },
            mentionTeam=True,
        )


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    repo_id: str,
    api: HfApi,
    wandb_logger: WandbLogger,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    val_loss, prompts, outputs = validate(
        fabric, model, val_data, tokenizer, iters=2
    )  # sanity check

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    loss_prev = 1
    total_lengths = 0
    total_t0 = time.perf_counter()
    columns = [
        "step_num",
        "prompt (poor)",
        "output (poor)",
        "prompt (average)",
        "output (average)",
        "prompt (good)",
        "output (good)",
        "prompt (excellent)",
        "output (excellent)",
    ]
    output_logged_text = []

    # save instruction and output to wandb table
    output_logged_text.append(
        [
            0,
            outputs[0],
            prompts[1],
            outputs[1],
            prompts[2],
            outputs[2],
            prompts[3],
            outputs[3],
        ]
    )

    wandb_logger.log_text(key="val_examples", columns=columns, data=output_logged_text)

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
            if step_count > warmup_steps:
                scheduler.step()
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
            val_loss, prompts, outputs = validate(
                fabric, model, val_data, tokenizer, iters=eval_iters
            )
            wandb.log({"val_loss": val_loss, "train_step": step_count})
            # save instruction and output to wandb table
            output_logged_text.append(
                [
                    step_count,
                    prompts[0],
                    outputs[0],
                    prompts[1],
                    outputs[1],
                    prompts[2],
                    outputs[2],
                    prompts[3],
                    outputs[3],
                ]
            )
            wandb_logger.log_text(
                key="val_examples", columns=columns, data=output_logged_text
            )
            t1 = time.perf_counter() - t0
            fabric.print(
                f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms"
            )
            # LOG to discord validation loss
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
            save_lora_checkpoint(fabric, model, checkpoint_path)
            api.upload_folder(
                folder_path=out_dir,
                path_in_repo="./",
                repo_id=repo_id,
                repo_type="model",
            )

    send_embedded_message(
        f"Training Complete: {repo_id}", "Eval training", mentionTeam=True
    )


sample_inputs = [
    # Poor
    "Question: What are the key principles and challenges in conducting clinical trials for new drugs in the field of clinical pharmacology?\nEvaluation Criteria: Identification and discussion of key principles in clinical trials for new drugs\nAnalysis of challenges faced in conducting clinical trials\nDemonstration of understanding of the field of clinical pharmacology\nCoherence and clarity of argumentation\nAccuracy of information\nGrammar and spelling\nAnswer: Clinical pharmacology is when you study drugs and how they work in people. There are challenges like finding enough people for the trials and making sure the drug is safe. The principles are like making sure the drug works and is better than what's already out there. It's important to test on different kinds of people and record the results. Sometimes people get side effects.",
    # Average
    "Question: Explain how the integumentary system plays a crucial role in maintaining the body's homeostasis. Include the specific mechanisms and processes involved.\nEvaluation Criteria: The answer should clearly explain the concept of homeostasis and the integumentary system.\nThe answer should accurately describe the different roles played by the integumentary system in maintaining homeostasis, including temperature regulation, protection against harmful microbes, and sensation.\nThe answer should elucidate the mechanisms involved in these roles, such as how sweat glands help regulate body temperature, how the skin acts as a barrier, and how nerve endings allow for sensation.\nThe answer should be coherent and logically structured, with appropriate use of anatomical vocabulary.\nThe answer should be grammatically correct, without spelling or punctuation errors.\nAnswer: The integumentary system is really important for our body to keep everything balanced. It is made up of our skin, hair, and nails. When we are hot, we sweat and the sweat makes us cool down. The skin also keeps bad things like bacteria out of our body. And our skin also feels things like heat and cold.",
    # Good
    "Question: Explain how the process of gluconeogenesis differs from glycolysis and discuss its significance in human metabolism.\nEvaluation Criteria: The response should correctly define both gluconeogenesis and glycolysis and highlight the key differences between these two metabolic pathways.\nThe answer should discuss the crucial enzymes involved in each process, highlighting those specific to gluconeogenesis and glycolysis.\nIt should explain the significance of gluconeogenesis in maintaining glucose homeostasis in the body, particularly during fasting or starvation.\nThe student's reasoning should be logically structured, with a coherent flow from one point to the next.\nThe language should be scientifically accurate, employing appropriate biochemistry terminology.\nGrammatical correctness and spelling accuracy are also essential.\nAnswer: Gluconeogenesis and glycolysis are processes that our body uses to manage our blood sugar. Gluconeogenesis is when our body makes glucose from other things that aren't carbohydrates, like proteins or lipids. Glycolysis is when our body breaks down glucose for energy. They're important because they help our body keep our blood sugar levels stable. If we didn't have gluconeogenesis, we wouldn't be able to keep our blood sugar stable when we're not eating.",
    # Excellent
    "Question: How do nanosensors significantly contribute to the field of nanotechnology and what potential future advancements could further enhance their functionality?\nEvaluation Criteria: The answer should clearly define what nanosensors are and provide information about their role in nanotechnology.\nThe response should provide specific examples of how nanosensors have contributed to advances in nanotechnology.\nThe student should discuss potential enhancements to nanosensor functionality that could be made in the future.\nThe argument should be logically structured and well-articulated, progressing from the definition and role of nanosensors, to their contributions, and finally to future possibilities.\nThe answer should be free of spelling and grammatical errors.\nAnswer: Nanosensors are devices that use the unique properties of nanomaterials and nanoparticles to detect and measure phenomena on the nanoscale. They play a crucial role in nanotechnology by providing data about nanoscale objects and processes that are invaluable for research and development. One of the notable contributions of nanosensors in nanotechnology is in the area of medical diagnostics. For example, nanosensors can detect disease biomarkers at very early stages, allowing for prompt treatment. Furthermore, nanosensors have paved the way for advancements in environmental monitoring, where they can detect pollutants at minute concentrations, contributing to improved environmental protection efforts. Looking to the future, nanosensors could be enhanced through the integration of machine learning algorithms. These algorithms could help in predicting patterns and making sense of the vast amount of data generated. Similarly, the development of more robust and versatile nanomaterials could lead to nanosensors with increased sensitivity and specificity. This would further expand their potential applications across various fields, from healthcare to environmental protection, and beyond.",
]


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_data: List[Dict],
    tokenizer: Tokenizer,
    iters: int,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        input_ids, targets = get_batch(fabric, val_data)
        if input_ids.shape[1] > model.max_seq_length:
            input_ids = input_ids[:, : model.max_seq_length]
        if targets.shape[1] > model.max_seq_length:
            targets = targets[:, : model.max_seq_length]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )
    val_loss = losses.mean()

    # produce an example:
    instruction = "Instructions: Evaluate the answer of the following question. Give a score in terms of relevance, coherence and grammar and explanation of for the evaluation. Please structure your response as follows:\n1. Begin with the 'Answer Evaluation' section, offering an in-depth review and analysis of the student's answer with respect to the given evaluation criteria.\n2. Follow this with a whole number numerical score for relevance (out of 6), coherence (out of 2), and grammar & spelling (out of 2) for the student's answer. Each score should be listed on a new line, preceded by its respective category."
    fabric.print(instruction)

    prompts = []
    outputs = []

    for sample_input in sample_inputs:
        sample = {
            "instruction": instruction,
            "input": sample_input,
        }
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
        prompts.append(prompt)
        outputs.append(output)

    model.train()
    return val_loss, prompts, outputs


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


def save_lora_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


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
