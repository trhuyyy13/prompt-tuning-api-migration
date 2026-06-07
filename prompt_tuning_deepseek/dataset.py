"""Dataset for global Prompt Tuning: [P_global] + probing_input -> y_pos.

Important: `y_pos` is the *target/label*. It is never fed into the model as
part of the prompt — it only contributes the `labels` (and the teacher-forced
half of `input_ids` for causal-LM training). The mapping the model learns is

    [P_global] + probing_input  ->  y_pos

`probing_input` tokens are masked with -100 in `labels`, so the CE loss is
only computed on `y_pos` tokens (the soft prompt is masked too, by the model
wrapper, since it never produces `input_ids`).
"""
import torch
from torch.utils.data import Dataset

from utils import load_json_or_jsonl

# The released JSON uses "probing input" / "deprecated api" / ... (with spaces).
# We also accept underscored variants in case the data gets re-exported.
INPUT_FIELD_CANDIDATES = ("probing_input", "probing input")
TARGET_FIELD_CANDIDATES = ("y_pos",)
DEPRECATED_API_CANDIDATES = ("deprecated_api", "deprecated api", "old_api")
REPLACEMENT_API_CANDIDATES = ("replacement_api", "replacement api", "new_api", "expected call")
ALIAS_DICT_CANDIDATES = ("alias_dict", "alias dict")


def first_present(sample, candidates, default=None):
    for key in candidates:
        if key in sample and sample[key] is not None:
            return sample[key]
    return default


def build_metadata(sample):
    deprecated_api = first_present(sample, DEPRECATED_API_CANDIDATES, default=[])
    if isinstance(deprecated_api, str):
        deprecated_api = [deprecated_api] if deprecated_api else []

    return {
        "library": sample.get("library"),
        "category": sample.get("category"),
        "sample_index": sample.get("sample_index"),
        "deprecated_api": deprecated_api,
        "replacement_api": first_present(sample, REPLACEMENT_API_CANDIDATES),
        "alias_dict": first_present(sample, ALIAS_DICT_CANDIDATES, default={}) or {},
    }


class PromptTuningDataset(Dataset):
    """Tokenizes (probing_input, y_pos) pairs for teacher-forced causal-LM training."""

    def __init__(self, data_path, tokenizer, max_input_length=512,
                 max_target_length=128, max_seq_length=640):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_seq_length

        raw_samples = load_json_or_jsonl(data_path)
        self.examples = []
        for sample in raw_samples:
            input_text = first_present(sample, INPUT_FIELD_CANDIDATES)
            target_text = first_present(sample, TARGET_FIELD_CANDIDATES)
            if input_text is None or target_text is None:
                continue
            self.examples.append(self._encode(sample, input_text, target_text))

    def _encode(self, sample, input_text, target_text):
        tok = self.tokenizer

        input_ids_input = tok(
            input_text, add_special_tokens=False,
            truncation=True, max_length=self.max_input_length,
        ).input_ids
        input_ids_target = tok(
            target_text, add_special_tokens=False,
            truncation=True, max_length=self.max_target_length,
        ).input_ids

        if tok.eos_token_id is not None:
            input_ids_target = list(input_ids_target) + [tok.eos_token_id]

        input_ids = list(input_ids_input) + list(input_ids_target)
        labels = [-100] * len(input_ids_input) + list(input_ids_target)

        # If the concatenation overflows max_seq_length, trim from the LEFT of
        # the context only -- the target half (where the loss lives) is never
        # touched, so `labels != -100` always equals len(input_ids_target).
        overflow = len(input_ids) - self.max_seq_length
        if overflow > 0:
            cut = min(overflow, len(input_ids_input))
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            input_ids_input = input_ids_input[cut:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "input_len": len(input_ids_input),
            "target_len": len(input_ids_target),
            "probing_input": input_text,
            "y_pos": target_text,
            "metadata": build_metadata(sample),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DataCollatorForPromptTuning:
    """Dynamic padding: input_ids -> pad_token_id, attention_mask -> 0, labels -> -100."""

    def __init__(self, tokenizer):
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set before building the collator")
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "input_len": torch.tensor([f["input_len"] for f in features], dtype=torch.long),
            "target_len": torch.tensor([f["target_len"] for f in features], dtype=torch.long),
        }
