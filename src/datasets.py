import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from config import SFTConfig


def ensure_dirs(cfg: SFTConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging_dir).mkdir(parents=True, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_messages(example: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Extracts:
      - task_id
      - system_content
      - user_content
      - assistant_content

    from one dataset example.
    """
    task_id = str(example.get("task_id", ""))

    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("Example is missing a valid 'messages' list.")

    system_content = None
    user_content = None
    assistant_content = None

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()

        if role == "system" and system_content is None:
            system_content = content
        elif role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and assistant_content is None:
            assistant_content = content

    if not system_content:
        raise ValueError("Could not find system message in example.")
    if not user_content:
        raise ValueError("Could not find user message in example.")
    if not assistant_content:
        raise ValueError("Could not find assistant message in example.")

    return task_id, system_content, user_content, assistant_content


def build_training_messages(
    system_content: str,
    user_content: str,
    assistant_content: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Builds the message list exactly in the same structure as the dataset.

    During training:
      system + user + assistant

    During inference:
      system + user
    """
    messages = [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()},
    ]

    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content.strip()})

    return messages


def build_training_texts(
    tokenizer: AutoTokenizer,
    system_content: str,
    user_content: str,
    assistant_content: str,
) -> Tuple[str, str]:
    """
    Returns:
      - full_text:   system + user + assistant
      - prompt_text: system + user + assistant generation prefix

    full_text is used as the full token sequence.
    prompt_text is used to know where the assistant answer starts,
    so the loss can be masked before that point.
    """
    full_messages = build_training_messages(
        system_content=system_content,
        user_content=user_content,
        assistant_content=assistant_content,
    )

    prompt_messages = build_training_messages(
        system_content=system_content,
        user_content=user_content,
        assistant_content=None,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return full_text, prompt_text


def tokenize_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    cfg: SFTConfig,
) -> Dict[str, Any]:
    """
    Converts one raw dataset example into tokenized training features.

    Important:
    - system and user tokens are masked with -100
    - loss is computed only on the assistant response
    """
    task_id, system_content, user_content, assistant_content = extract_messages(example)

    full_text, prompt_text = build_training_texts(
        tokenizer=tokenizer,
        system_content=system_content,
        user_content=user_content,
        assistant_content=assistant_content,
    )

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding=False,
        add_special_tokens=False,
    )

    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    prefix_len = min(len(prompt_enc["input_ids"]), len(input_ids))

    labels = input_ids.copy()
    labels[:prefix_len] = [-100] * prefix_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task_id": task_id,
        "system_content": system_content,
        "user_content": user_content,
        "target_assistant_content": assistant_content,
    }


class PromptDataCollator:
    """
    Pads variable-length examples inside a batch.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


def load_tokenized_datasets(
    tokenizer: AutoTokenizer,
    cfg: SFTConfig,
) -> Tuple[Dataset, Dataset]:
    """
    Loads train and validation JSONL files,
    then tokenizes all examples.
    """
    raw = load_dataset(
        "json",
        data_files={
            "train": cfg.train_file,
            "validation": cfg.valid_file,
        },
    )

    train_ds = raw["train"].map(
        lambda ex: tokenize_example(ex, tokenizer, cfg),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing train dataset",
    )

    valid_ds = raw["validation"].map(
        lambda ex: tokenize_example(ex, tokenizer, cfg),
        remove_columns=raw["validation"].column_names,
        desc="Tokenizing validation dataset",
    )

    return train_ds, valid_ds