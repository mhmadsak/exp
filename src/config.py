from dataclasses import dataclass
from typing import Tuple


@dataclass
class SFTConfig:
    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    trust_remote_code: bool = True

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------
    train_file: str = "data/processed/train.jsonl"
    valid_file: str = "data/processed/valid.jsonl"

    # ------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------
    output_dir: str = "outputs/qwen25_prompt_rewriter"
    logging_dir: str = "outputs/qwen25_prompt_rewriter/logs"
    generated_samples_file: str = "outputs/qwen25_prompt_rewriter/validation_generations.jsonl"
    final_metrics_file: str = "outputs/qwen25_prompt_rewriter/final_metrics.json"
    config_save_file: str = "outputs/qwen25_prompt_rewriter/run_config.json"

    # ------------------------------------------------------------
    # Tokenization / sequence length
    # ------------------------------------------------------------
    max_seq_length: int = 2048

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2

    # ------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    # ------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )

    # ------------------------------------------------------------
    # Optional quantized loading
    # ------------------------------------------------------------
    use_4bit: bool = False

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------
    # Validation generation
    # ------------------------------------------------------------
    num_validation_generations: int = 5
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.2
    generation_top_p: float = 0.9
    do_sample: bool = True