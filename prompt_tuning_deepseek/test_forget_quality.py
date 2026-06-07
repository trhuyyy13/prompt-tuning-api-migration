"""API-migration quality test (style follows Thamkhao/forget_quality.py).

This is NOT a training-loss check. It measures whether, after Prompt Tuning,
[P_global] + probing_input actually generates the *replacement* API rather
than the deprecated one. `y_pos` is used purely as a reference for exact-match
-- it is never fed into the model.

Two ways to get predictions:
  1. `--predictions_file predictions.json`  (output of evaluate.py, fastest)
  2. `--checkpoint_dir ... --data_file ...` (generate on the fly, like
     forget_quality.py does -- handy for "test directly on the train set").
"""
import argparse

import torch
from transformers import AutoTokenizer

from evaluate import build_model_from_checkpoint, generate_predictions
from utils import contains_api, load_json_or_jsonl, normalize_code_text, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Test API-migration (forget) quality after Prompt Tuning")
    p.add_argument("--predictions_file", default=None,
                   help="predictions.json from evaluate.py; if omitted, generate fresh from --checkpoint_dir/--data_file")

    # Used only when --predictions_file is not given.
    p.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-coder-1.3b-base")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--data_file", default=None)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_metrics", required=True)
    p.add_argument("--output_details", required=True)
    return p.parse_args()


def get_predictions(args):
    if args.predictions_file:
        return load_json_or_jsonl(args.predictions_file)

    if not (args.checkpoint_dir and args.data_file):
        raise ValueError("Provide either --predictions_file, or both --checkpoint_dir and --data_file")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model_from_checkpoint(args.model_name_or_path, args.checkpoint_dir, tokenizer, device)
    samples = load_json_or_jsonl(args.data_file)
    if args.limit:
        samples = samples[: args.limit]
    return generate_predictions(model, tokenizer, samples, device, args.max_input_length,
                                args.max_new_tokens, desc="Generating (forget-quality test)")


def main():
    args = parse_args()
    predictions = get_predictions(args)

    details = []
    new_api_hit = old_api_present = exact_match = 0
    target_has_new = pred_has_new = 0

    for rec in predictions:
        meta = rec.get("metadata", {}) or {}
        replacement_api = meta.get("replacement_api")
        deprecated_apis = meta.get("deprecated_api") or []
        if isinstance(deprecated_apis, str):
            deprecated_apis = [deprecated_apis]
        alias_dict = meta.get("alias_dict") or {}

        prediction, target = rec["prediction"], rec["y_pos"]

        is_new_hit = contains_api(prediction, replacement_api, alias_dict)
        is_old_present = any(contains_api(prediction, api, alias_dict) for api in deprecated_apis)
        is_exact = normalize_code_text(prediction) == normalize_code_text(target)
        target_contains_new = contains_api(target, replacement_api, alias_dict)

        new_api_hit += int(is_new_hit)
        old_api_present += int(is_old_present)
        exact_match += int(is_exact)
        target_has_new += int(target_contains_new)
        pred_has_new += int(is_new_hit)

        details.append({
            "probing_input": rec["probing_input"],
            "y_pos": target,
            "prediction": prediction,
            "deprecated_api": deprecated_apis,
            "replacement_api": replacement_api,
            "new_api_hit": is_new_hit,
            "old_api_still_present": is_old_present,
            "exact_match": is_exact,
        })

    total = len(predictions)
    metrics = {
        "total": total,
        "new_api_hit_count": new_api_hit,
        "new_api_hit_rate": (new_api_hit / total) if total else 0.0,
        "old_api_still_present_count": old_api_present,
        "old_api_still_present_rate": (old_api_present / total) if total else 0.0,
        "exact_match_count": exact_match,
        "exact_match_rate": (exact_match / total) if total else 0.0,
        "target_contains_new_api_count": target_has_new,
        "prediction_contains_new_api_count": pred_has_new,
    }

    save_json(metrics, args.output_metrics)
    save_json(details, args.output_details)

    print(f"[*] forget-quality metrics -> {args.output_metrics}")
    print(f"[*] forget-quality details -> {args.output_details}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
