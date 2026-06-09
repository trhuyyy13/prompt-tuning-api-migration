"""Generate with the trained global soft prompt and score against y_pos.

Generation input is STRICTLY  [P_global] + probing_input.
`y_pos` is only used afterwards, as the reference for scoring -- it is never
fed into the model.
"""
import argparse
import json
import os
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import build_metadata, first_present, INPUT_FIELD_CANDIDATES, TARGET_FIELD_CANDIDATES
from model_prompt_tuning import SoftPromptCausalLM
from utils import load_json_or_jsonl, load_soft_prompt_checkpoint, normalize_code_text, save_json, set_seed

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def parse_args():
    p = argparse.ArgumentParser(description="Generate & score with the trained P_global soft prompt")
    p.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-coder-1.3b-base")
    p.add_argument("--data_file", required=True)
    p.add_argument("--checkpoint_dir", required=True, help="dir containing soft_prompt.pt + prompt_config.json")
    p.add_argument("--output_file", required=True, help="where to write predictions.json")
    p.add_argument("--metrics_file", default=None, help="defaults to <output_dir>/metrics.json")
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8,
                   help="number of samples per generation batch (default 8; use 1 for debugging)")
    p.add_argument("--limit", type=int, default=None, help="evaluate on only the first N samples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """Return a local directory path for the checkpoint.

    If *checkpoint_dir* is an existing local directory, return it as-is.
    Otherwise treat it as a Hugging Face Hub repo ID and download the repo
    to a local cache directory via ``huggingface_hub.snapshot_download``.
    """
    if os.path.isdir(checkpoint_dir):
        return checkpoint_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to load checkpoints from the Hub. "
            "Install it with: pip install huggingface_hub"
        ) from e
    print(f"[*] {checkpoint_dir!r} is not a local directory — downloading from Hugging Face Hub...")
    local_dir = snapshot_download(repo_id=checkpoint_dir)
    print(f"[*] downloaded to {local_dir}")
    return local_dir


def load_prompt_config(checkpoint_dir):
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    path = os.path.join(checkpoint_dir, "prompt_config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_base_model(model_name_or_path, tokenizer, device):
    """Wrap the untouched base model with a zero-length soft prompt.

    This reuses the exact same generation path as `build_model_from_checkpoint`
    (`generate_with_soft_prompt` -> `inputs_embeds=...`) with a no-op prompt, so
    "before tuning" and "after tuning" numbers are produced by identical code
    and are directly comparable.
    """
    model = SoftPromptCausalLM(
        model_name_or_path=model_name_or_path,
        num_virtual_tokens=0,
        prompt_init="random",
        tokenizer=tokenizer,
    )
    model.to(device)
    model.eval()
    print(f"[*] loaded base model {model_name_or_path} as-is (no soft prompt -- baseline)")
    return model


def build_model_from_checkpoint(model_name_or_path, checkpoint_dir, tokenizer, device):
    local_dir = _resolve_checkpoint_dir(checkpoint_dir)
    cfg = load_prompt_config(local_dir)
    # Prefer the base model recorded at training time; fall back to the CLI arg.
    resolved_base = cfg.get("model_name_or_path") or model_name_or_path
    if resolved_base != model_name_or_path:
        print(f"[*] using base model from checkpoint config: {resolved_base}")
    model = SoftPromptCausalLM(
        model_name_or_path=resolved_base,
        num_virtual_tokens=cfg["num_virtual_tokens"],
        prompt_init="random",  # overwritten by the trained checkpoint right below
        tokenizer=tokenizer,
    )
    soft_prompt = load_soft_prompt_checkpoint(local_dir)
    with torch.no_grad():
        model.soft_prompt.copy_(soft_prompt.to(device=model.soft_prompt.device, dtype=model.soft_prompt.dtype))
    model.to(device)
    model.eval()
    print(f"[*] loaded soft prompt ({cfg['num_virtual_tokens']} virtual tokens) from {checkpoint_dir}")
    return model


@torch.no_grad()
def generate_predictions(model, tokenizer, samples, device, max_input_length, max_new_tokens,
                          do_sample=False, num_beams=1, batch_size=8, desc="Generating"):
    """Run [P_global] + probing_input -> generation for each sample (y_pos is NOT used as input).

    Uses left-padded batched generation for higher GPU throughput.
    """
    # Collect valid (sample, input_text, target_text) tuples first.
    valid = []
    for s in samples:
        inp = first_present(s, INPUT_FIELD_CANDIDATES)
        tgt = first_present(s, TARGET_FIELD_CANDIDATES)
        if inp is not None and tgt is not None:
            valid.append((s, inp, tgt))

    # Left-pad so the model generates from the rightmost real token.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    predictions = []
    for i in tqdm(range(0, len(valid), batch_size), desc=desc):
        chunk = valid[i : i + batch_size]
        batch_samples, batch_inputs, batch_targets = zip(*chunk)

        encoded = tokenizer(
            list(batch_inputs),
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids    = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        output_ids = model.generate_with_soft_prompt(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams,
        )

        for j, (sample, input_text, target_text) in enumerate(chunk):
            row_ids = output_ids[j] if output_ids.ndim > 1 else output_ids
            prediction = tokenizer.decode(row_ids, skip_special_tokens=True)
            predictions.append({
                "probing_input": input_text,
                "y_pos": target_text,
                "prediction": prediction,
                "metadata": build_metadata(sample),
            })

    tokenizer.padding_side = orig_padding_side
    return predictions


# ----------------------------------------------------------------------
# Metrics: Exact Match, BLEU / ROUGE-L, token-level F1
# ----------------------------------------------------------------------
def token_f1(pred, ref):
    pred_tokens, ref_tokens = pred.split(), ref.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(predictions):
    total = len(predictions)
    if total == 0:
        return {"total": 0}

    preds = [normalize_code_text(r["prediction"]) for r in predictions]
    refs = [normalize_code_text(r["y_pos"]) for r in predictions]

    exact_match = sum(1 for p, r in zip(preds, refs) if p == r)
    f1_scores = [token_f1(p, r) for p, r in zip(preds, refs)]

    metrics = {
        "total": total,
        "exact_match_count": exact_match,
        "exact_match_rate": exact_match / total,
        "token_level_f1": sum(f1_scores) / total,
    }

    if sacrebleu is not None:
        metrics["bleu"] = sacrebleu.corpus_bleu(preds, [refs]).score
    else:
        print("[warn] sacrebleu not installed -- skipping BLEU")

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
        metrics["rouge_l"] = sum(rouge_l) / total
    else:
        print("[warn] rouge_score not installed -- skipping ROUGE-L")

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")

    # Resolve base model from checkpoint config so --checkpoint_dir <hf-repo-id>
    # works without also specifying --model_name_or_path.
    local_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    cfg = load_prompt_config(local_dir)
    base_model = cfg.get("model_name_or_path") or args.model_name_or_path
    if base_model != args.model_name_or_path:
        print(f"[*] using base model from checkpoint config: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model_from_checkpoint(base_model, local_dir, tokenizer, device)

    samples = load_json_or_jsonl(args.data_file)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[*] generating for {len(samples)} samples (max_new_tokens={args.max_new_tokens}, greedy)")

    predictions = generate_predictions(model, tokenizer, samples, device,
                                        args.max_input_length, args.max_new_tokens,
                                        batch_size=args.batch_size)
    save_json(predictions, args.output_file)
    print(f"[*] wrote predictions -> {args.output_file}")

    metrics = compute_metrics(predictions)
    metrics_file = args.metrics_file or os.path.join(os.path.dirname(os.path.abspath(args.output_file)), "metrics.json")
    save_json(metrics, metrics_file)
    print(f"[*] metrics -> {metrics_file}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
