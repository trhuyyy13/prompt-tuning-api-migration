"""Small shared helpers for the global Prompt Tuning baseline."""
import json
import os
import random
import re

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_or_jsonl(path):
    """Load either a JSON list/dict file or a JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]

    examples = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def save_json(obj, path, indent=2):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def normalize_code_text(s):
    return " ".join(str(s).strip().split())


def contains_api(text, api, alias_dict=None):
    """Check whether `text` references `api`.

    `api` is the canonical/replacement form (e.g. "numpy.prod"), but generated
    code usually uses a shorter alias (e.g. "np.prod"). `alias_dict` maps the
    alias seen in code -> canonical dotted path, so we also match any alias
    whose canonical value equals `api`.
    """
    if not api:
        return False

    norm_text = normalize_code_text(text)
    norm_api = normalize_code_text(api)

    candidates = {norm_api}
    if alias_dict:
        for alias, canonical in alias_dict.items():
            if normalize_code_text(canonical) == norm_api:
                candidates.add(normalize_code_text(alias))

    for cand in candidates:
        if not cand:
            continue
        pattern = r"(?<![\w.])" + re.escape(cand) + r"(?![\w])"
        if re.search(pattern, norm_text):
            return True
    return False


def save_checkpoint(output_dir, model, tokenizer, optimizer=None, scheduler=None,
                     global_step=0, epoch=0, extra=None):
    """Persist only what's needed to resume / reuse the soft prompt.

    The frozen 1.3B base model is intentionally NOT saved here.
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.soft_prompt.detach().to(torch.float32).cpu(),
               os.path.join(output_dir, "soft_prompt.pt"))

    tokenizer.save_pretrained(output_dir)

    state = {"global_step": global_step, "epoch": epoch}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, os.path.join(output_dir, "training_state.pt"))


def load_soft_prompt_checkpoint(checkpoint_dir, map_location="cpu"):
    """Load a saved soft prompt tensor from a checkpoint directory or file path."""
    path = checkpoint_dir
    if os.path.isdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, "soft_prompt.pt")
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict):
        obj = obj.get("soft_prompt", obj)
    return obj
