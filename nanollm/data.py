"""Data loading and batching for pretraining and LoRA fine-tuning."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------- #
# pretraining data
# --------------------------------------------------------------------- #


def tokenize_corpus(corpus_path: str | Path, tokenizer, out_dir: str | Path, val_fraction: float = 0.01):
    """Tokenize a plain-text corpus into uint16 numpy arrays train.bin/val.bin.

    The last ``val_fraction`` of the corpus is held out for validation so the
    validation loss measures generalization to unseen (later) text.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(corpus_path).read_text(encoding="utf-8")
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    n_val = max(1, int(len(ids) * val_fraction))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    return {"train_tokens": int(len(train_ids)), "val_tokens": int(len(val_ids))}


class CorpusBatcher:
    """Random-offset batching over a token array (nanoGPT-style memory map)."""

    def __init__(self, data_dir: str | Path, batch_size: int, block_size: int, device: str = "cuda"):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.data_dir = Path(data_dir)
        self.split_files = {
            "train": self.data_dir / "train.bin",
            "val": self.data_dir / "val.bin",
        }
        self._memmaps: dict[str, np.memmap] = {}

    def _tokens(self, split: str) -> np.ndarray:
        if split not in self._memmaps:
            arr = np.memmap(self.split_files[split], dtype=np.uint16, mode="r")
            self._memmaps[split] = arr
        return self._memmaps[split]

    def get_batch(self, split: str, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._tokens(split)
        total = len(data)
        ix = torch.randint(0, total - self.block_size, (self.batch_size,), generator=None)
        x = torch.stack([torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self._tokens("train"))


# --------------------------------------------------------------------- #
# instruction fine-tuning data
# --------------------------------------------------------------------- #

USER_TOK, ASSISTANT_TOK, END_TOK = "<|user|>", "<|assistant|>", "<|end|>"


def build_instruction_sequence(question: str, answer: str, tokenizer) -> list[int]:
    """Encode a QA pair into the instruction format with loss masking.

    Returns (input_ids, target_ids) where target_ids are -100 everywhere
    except the assistant portion (only the answer is supervised).
    """
    q_ids = tokenizer.encode(f"{USER_TOK}{question}{ASSISTANT_TOK}")
    a_ids = tokenizer.encode(f"{answer}{END_TOK}")
    input_ids = q_ids + a_ids
    assistant_pos = len(q_ids)
    targets = [-100] * len(input_ids)
    for i in range(assistant_pos, len(input_ids) - 1):
        targets[i] = input_ids[i + 1]
    return input_ids, targets


def load_qa_pairs(jsonl_path: str | Path) -> list[dict]:
    pairs = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append({"question": obj["question"], "answer": obj["answer"]})
    return pairs


def qa_pad_collate(batch: list[tuple[list[int], list[int]]], block_size: int, pad_id: int = 0):
    """Collate variable-length instruction sequences into padded batches.

    Sequences longer than block_size are truncated (with targets masked);
    shorter ones are right-padded with target -100 so padding never trains.
    """
    xs, ys = [], []
    for x, y in batch:
        x = x[:block_size]
        y = y[:block_size]
        pad = block_size - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
    )


class InstructionDataset(torch.utils.data.Dataset):
    """Packed instruction sequences for LoRA fine-tuning."""

    def __init__(self, jsonl_path: str | Path, tokenizer, block_size: int):
        self.block_size = block_size
        pairs = load_qa_pairs(jsonl_path)
        self.samples = [build_instruction_sequence(p["question"], p["answer"], tokenizer) for p in pairs]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        if len(x) > self.block_size:
            x, y = x[: self.block_size], y[: self.block_size]
        return x, y
