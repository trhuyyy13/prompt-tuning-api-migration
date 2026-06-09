"""API-migration quality test (classification logic follows Thamkhao/forget_quality.py).

This is NOT a training-loss check. It measures whether, after Prompt Tuning,
[P_global] + probing_input actually generates the *replacement* API rather
than the deprecated one. `y_pos` is used purely as a reference -- it is never
fed into the model.

Each generated sample is bucketed exactly like Thamkhao/forget_quality.py
(`check_api_usage` + alias dict, with replacement checked before deprecated):

    - "replacement" (R) : prediction contains the replacement API
    - "deprecated"  (D) : prediction still contains a deprecated API (and not R)
    - "mismatch"        : neither API appears in the prediction

Three ways to get predictions:
  1. `--predictions_file predictions.json`  (output of evaluate.py, fastest)
  2. `--checkpoint_dir ... --data_file ...` (generate on the fly, like
     forget_quality.py does -- handy for "test directly on the train set").
  3. `--baseline --data_file ...` (no checkpoint: run the *original*,
     untouched base model -- handy for measuring the R/D/Mismatch split
     BEFORE prompt tuning, as a point of comparison).

Everything is written into a single `--output_dir`:
  - forget_quality_metrics.json : total + counts/rates per type (R/D/Mismatch) + exact match
  - forget_quality_details.json : per-sample {probing_input, target, predict, type, ...}
"""
import argparse
import os

import torch
from transformers import AutoTokenizer

from evaluate import build_base_model, build_model_from_checkpoint, generate_predictions
from utils import contains_api, load_json_or_jsonl, normalize_code_text, save_json, set_seed

TYPE_REPLACEMENT = "replacement"
TYPE_DEPRECATED = "deprecated"
TYPE_MISMATCH = "mismatch"


def parse_args():
    p = argparse.ArgumentParser(description="Test API-migration (forget) quality after Prompt Tuning")
    p.add_argument("--predictions_file", default=None,
                   help="predictions.json from evaluate.py; if omitted, generate fresh from --checkpoint_dir/--data_file")

    # Used only when --predictions_file is not given.
    p.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-coder-1.3b-base")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--baseline", action="store_true",
                   help="Skip the soft prompt and run the original base model as-is "
                        "(measures R/D/Mismatch BEFORE prompt tuning). "
                        "Mutually exclusive with --checkpoint_dir.")
    p.add_argument("--data_file", default=None)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8,
                   help="number of samples per generation batch (default 8; use 1 for debugging)")
    p.add_argument("--num_beams", type=int, default=1,
                   help="beam width for generation (default 1 = greedy)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", required=True, help="folder to write forget_quality_metrics.json / forget_quality_details.json into")
    return p.parse_args()


def get_predictions(args):
    if args.predictions_file:
        return load_json_or_jsonl(args.predictions_file)

    if args.baseline and args.checkpoint_dir:
        raise ValueError("--baseline and --checkpoint_dir are mutually exclusive")
    if not args.data_file or not (args.baseline or args.checkpoint_dir):
        raise ValueError("Provide either --predictions_file, or --data_file with "
                         "--checkpoint_dir (trained soft prompt) or --baseline (original model)")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve the base model name: prefer what was recorded in prompt_config.json
    # (so passing a HF Hub repo ID as --checkpoint_dir just works without also
    # specifying --model_name_or_path).
    base_model = args.model_name_or_path
    if not args.baseline and args.checkpoint_dir:
        from evaluate import _resolve_checkpoint_dir, load_prompt_config
        try:
            local_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
            cfg = load_prompt_config(local_dir)
            base_model = cfg.get("model_name_or_path") or base_model
            if base_model != args.model_name_or_path:
                print(f"[*] using base model from checkpoint config: {base_model}")
        except Exception:
            pass  # fall back to --model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.baseline:
        model = build_base_model(base_model, tokenizer, device)
    else:
        model = build_model_from_checkpoint(base_model, args.checkpoint_dir, tokenizer, device)
    samples = load_json_or_jsonl(args.data_file)
    if args.limit:
        samples = samples[: args.limit]
    return generate_predictions(model, tokenizer, samples, device, args.max_input_length,
                                args.max_new_tokens, num_beams=args.num_beams,
                                batch_size=args.batch_size,
                                desc="Generating (forget-quality test)")


def classify(prediction, replacement_api, deprecated_apis, alias_dict):
    """Same priority as Thamkhao/forget_quality.py: replacement wins over deprecated."""
    is_replacement = contains_api(prediction, replacement_api, alias_dict)
    is_deprecated = any(contains_api(prediction, api, alias_dict) for api in deprecated_apis)
    if is_replacement:
        sample_type = TYPE_REPLACEMENT
    elif is_deprecated:
        sample_type = TYPE_DEPRECATED
    else:
        sample_type = TYPE_MISMATCH
    return sample_type, is_replacement, is_deprecated


def main():
    args = parse_args()
    predictions = get_predictions(args)

    details = []
    type_counts = {TYPE_REPLACEMENT: 0, TYPE_DEPRECATED: 0, TYPE_MISMATCH: 0}
    new_api_hit = old_api_present = exact_match = 0

    for rec in predictions:
        meta = rec.get("metadata", {}) or {}
        replacement_api = meta.get("replacement_api")
        deprecated_apis = meta.get("deprecated_api") or []
        if isinstance(deprecated_apis, str):
            deprecated_apis = [deprecated_apis]
        alias_dict = meta.get("alias_dict") or {}

        prediction, target = rec["prediction"], rec["y_pos"]

        sample_type, is_replacement, is_deprecated = classify(prediction, replacement_api, deprecated_apis, alias_dict)
        is_exact = normalize_code_text(prediction) == normalize_code_text(target)

        type_counts[sample_type] += 1
        new_api_hit += int(is_replacement)
        old_api_present += int(is_deprecated)
        exact_match += int(is_exact)

        details.append({
            "probing_input": rec["probing_input"],
            "target": target,
            "predict": prediction,
            "type": sample_type,
            "deprecated_api": deprecated_apis,
            "replacement_api": replacement_api,
            "new_api_hit": is_replacement,
            "old_api_still_present": is_deprecated,
            "exact_match": is_exact,
        })

    total = len(predictions)
    rate = lambda n: (n / total) if total else 0.0
    metrics = {
        "total": total,
        "replacement_count": type_counts[TYPE_REPLACEMENT],
        "replacement_rate": rate(type_counts[TYPE_REPLACEMENT]),
        "deprecated_count": type_counts[TYPE_DEPRECATED],
        "deprecated_rate": rate(type_counts[TYPE_DEPRECATED]),
        "mismatch_count": type_counts[TYPE_MISMATCH],
        "mismatch_rate": rate(type_counts[TYPE_MISMATCH]),
        "new_api_hit_count": new_api_hit,
        "new_api_hit_rate": rate(new_api_hit),
        "old_api_still_present_count": old_api_present,
        "old_api_still_present_rate": rate(old_api_present),
        "exact_match_count": exact_match,
        "exact_match_rate": rate(exact_match),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "forget_quality_metrics.json")
    details_path = os.path.join(args.output_dir, "forget_quality_details.json")
    save_json(metrics, metrics_path)
    save_json(details, details_path)

    print(f"[*] forget-quality metrics -> {metrics_path}")
    print(f"[*] forget-quality details -> {details_path}")
    print("================ SUMMARY ================")
    print(f"Replacement (R)  : {type_counts[TYPE_REPLACEMENT]}/{total}  ({rate(type_counts[TYPE_REPLACEMENT]):.2%})")
    print(f"Deprecated  (D)  : {type_counts[TYPE_DEPRECATED]}/{total}  ({rate(type_counts[TYPE_DEPRECATED]):.2%})")
    print(f"Mismatch         : {type_counts[TYPE_MISMATCH]}/{total}  ({rate(type_counts[TYPE_MISMATCH]):.2%})")
    print(f"Exact match      : {exact_match}/{total}  ({rate(exact_match):.2%})")
    print("==========================================")


if __name__ == "__main__":
    main()
