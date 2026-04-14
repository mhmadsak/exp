import math
import os
import random
from dataclasses import asdict
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import SFTConfig
from exp.src.datasets import (
    PromptDataCollator,
    ensure_dirs,
    extract_messages,
    load_tokenized_datasets,
    save_json,
    write_jsonl,
)


def load_tokenizer(cfg: SFTConfig) -> AutoTokenizer:
    """
    Loads the tokenizer for the base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


def load_model(cfg: SFTConfig) -> AutoModelForCausalLM:
    """
    Loads the base model and optionally applies LoRA.
    """
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32,
    }

    if cfg.use_4bit:
        model_kwargs.update(
            {
                "load_in_4bit": True,
                "device_map": "auto",
            }
        )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.lora_target_modules),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def build_training_arguments(cfg: SFTConfig) -> TrainingArguments:
    """
    Creates the Hugging Face Trainer arguments.
    """
    return TrainingArguments(
        output_dir=cfg.output_dir,
        logging_dir=cfg.logging_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        report_to=["tensorboard"],
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        group_by_length=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


@torch.no_grad()
def generate_rewrite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_content: str,
    user_content: str,
    cfg: SFTConfig,
) -> str:
    """
    Generates a rewritten stronger prompt using the same
    system+user format used in the dataset.
    """
    model.eval()

    messages = [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    generated = model.generate(
        **enc,
        max_new_tokens=cfg.generation_max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.generation_temperature,
        top_p=cfg.generation_top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = generated[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def save_validation_generations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: SFTConfig,
) -> None:
    """
    Generates model outputs for a few validation examples
    and saves them for manual inspection.
    """
    from datasets import load_dataset

    raw_valid = load_dataset("json", data_files={"validation": cfg.valid_file})["validation"]

    n = min(cfg.num_validation_generations, len(raw_valid))
    indices = list(range(len(raw_valid)))
    random.shuffle(indices)
    indices = indices[:n]

    rows = []
    for idx in indices:
        ex = raw_valid[idx]
        task_id, system_content, user_content, assistant_content = extract_messages(ex)

        pred = generate_rewrite(
            model=model,
            tokenizer=tokenizer,
            system_content=system_content,
            user_content=user_content,
            cfg=cfg,
        )

        rows.append(
            {
                "task_id": task_id,
                "system_content": system_content,
                "user_content": user_content,
                "reference_output": assistant_content,
                "generated_output": pred,
            }
        )

    write_jsonl(cfg.generated_samples_file, rows)


def run_test_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: SFTConfig,
) -> None:
    """
    Small sanity-check generation after training.
    """
    system_content = "You are an expert prompt rewriter for coding tasks."
    user_content = """Coding task:
Write a Python function that checks whether a string is a palindrome, ignoring spaces and letter case.

Basic prompt:
Write Python code to check if a string is a palindrome.

Rewrite this basic prompt into a stronger prompt for a coding LLM. Keep the task meaning unchanged."""

    output = generate_rewrite(
        model=model,
        tokenizer=tokenizer,
        system_content=system_content,
        user_content=user_content,
        cfg=cfg,
    )

    print("\n=== Test inference ===")
    print("System:")
    print(system_content)
    print("\nUser:")
    print(user_content)
    print("\n=== Generated rewrite ===")
    print(output)


def run_sft_training(cfg: SFTConfig) -> None:
    """
    Runs the full SFT pipeline.

    This is intentionally not called 'main' so it can be imported
    inside a larger project.
    """
    ensure_dirs(cfg)
    save_json(cfg.config_save_file, asdict(cfg))
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg)
    train_ds, valid_ds = load_tokenized_datasets(tokenizer, cfg)
    model = load_model(cfg)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(cfg),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=PromptDataCollator(tokenizer),
    )

    train_result = trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.state.save_to_json(os.path.join(cfg.output_dir, "trainer_state.json"))

    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")

    final_metrics = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    save_json(cfg.final_metrics_file, final_metrics)

    save_validation_generations(model, tokenizer, cfg)
    run_test_inference(model, tokenizer, cfg) 