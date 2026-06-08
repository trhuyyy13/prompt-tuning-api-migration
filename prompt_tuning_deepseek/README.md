# Global Prompt Tuning baseline — DeepSeek-Coder-1.3B-base

A **Prompt Tuning (`P_global` only)** baseline for the deprecated-API
migration task:

```text
[P_global] + probing_input  ->  y_pos
```

- Base model `deepseek-ai/deepseek-coder-1.3b-base` is **fully frozen**.
- The only trainable parameters are a global soft prompt `P_global`
  (`num_virtual_tokens × hidden_size` virtual-token embeddings), prepended to
  the embeddings of `probing_input`.
- `y_pos` is the **target/label only**. It is concatenated after
  `probing_input` purely so the causal LM can be teacher-forced; the
  cross-entropy loss is computed **only on `y_pos` tokens** (both
  `probing_input` and the soft prompt are masked with `-100`).
- No `P_library`, `P_migration_type`, API-specific prompts, LoRA, Prefix
  Tuning, or full fine-tuning — this is the global-prompt-only baseline.

The released dataset stores the relevant fields as `"probing input"` /
`"y_pos"` (note the space, not underscore); `dataset.py` accepts both forms.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python prompt_tuning_deepseek/train_prompt_tuning.py \
  --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
  --train_file data_raw/outdated_y+_FINAL.json \
  --valid_file data_raw/outdated_y+_FINAL.json \
  --output_dir outputs/prompt_tuning_deepseek_global \
  --num_virtual_tokens 20 \
  --prompt_init random \
  --prompt_init_text "Generate the next line:" \
  --max_input_length 512 \
  --max_target_length 128 \
  --max_seq_length 640 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-3 \
  --num_train_epochs 10 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --seed 42 \
  --bf16
```

To resume an interrupted run:

```bash
... --resume_from_checkpoint                       # resumes from --output_dir
... --resume_from_checkpoint outputs/some/other_dir # resumes from a given dir
```

What gets written to `output_dir` (the 1.3B base model is **never** saved):

- `soft_prompt.pt` — the trained `P_global` tensor `[num_virtual_tokens, hidden_size]`
- `training_state.pt` — optimizer / scheduler state + `global_step`/`epoch` (for resume)
- `training_args.json`, `prompt_config.json`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, ...)

## Evaluate generation

```bash
python prompt_tuning_deepseek/evaluate.py \
  --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
  --data_file data_raw/outdated_y+_FINAL.json \
  --checkpoint_dir outputs/prompt_tuning_deepseek_global \
  --output_file outputs/prompt_tuning_deepseek_global/predictions.json \
  --max_input_length 512 \
  --max_new_tokens 128
```

Generation input is **strictly** `[P_global] + probing_input` (greedy,
`do_sample=False`, `num_beams=1`). `y_pos` is only read afterwards, as the
reference for scoring (Exact Match, BLEU, ROUGE-L, token-level F1). Add
`--limit 200` to try a quick subset first.

Outputs:
- `outputs/.../predictions.json` — `{probing_input, y_pos, prediction, metadata}` per sample
- `outputs/.../metrics.json`

## Test forget quality / API-migration quality

```bash
python prompt_tuning_deepseek/test_forget_quality.py \
  --predictions_file outputs/prompt_tuning_deepseek_global/predictions.json \
  --output_dir outputs/prompt_tuning_deepseek_global
```

Or generate on the fly (style of `Thamkhao/forget_quality.py`, e.g. to test
straight on the training set without a separate `evaluate.py` pass):

```bash
python prompt_tuning_deepseek/test_forget_quality.py \
  --checkpoint_dir outputs/prompt_tuning_deepseek_global \
  --data_file data_raw/outdated_y+_FINAL.json \
  --limit 500 \
  --output_dir outputs/prompt_tuning_deepseek_global
```

For every sample, the prediction is bucketed exactly like
`Thamkhao/forget_quality.py` (`check_api_usage` + each sample's own `alias
dict`, e.g. `np.prod` ↔ `numpy.prod`, replacement checked before deprecated —
see `classify()` / `contains_api()`):

- **`replacement` (R)** — prediction contains the replacement API
- **`deprecated` (D)** — prediction still contains a deprecated API (and not R)
- **`mismatch`** — neither API appears

Everything is written into `--output_dir`:

- `forget_quality_metrics.json` — `total`, `replacement_count/_rate`,
  `deprecated_count/_rate`, `mismatch_count/_rate`, `exact_match_count/_rate`, ...
- `forget_quality_details.json` — per-sample `{probing_input, target, predict,
  type, deprecated_api, replacement_api, ...}`

## Training flow (short version)

1. `dataset.py` tokenizes `probing_input` and `y_pos` **separately**, appends
   EOS to the target, concatenates them as `input_ids = input_ids_input +
   input_ids_target`, and builds `labels = [-100]*len(input_ids_input) +
   input_ids_target` — so the loss only ever sees `y_pos` tokens.
2. `model_prompt_tuning.SoftPromptCausalLM` embeds `input_ids` with the
   frozen base model's embedding layer, prepends `soft_prompt.expand(batch, -1, -1)`,
   prepends `1`s to `attention_mask` and `-100`s to `labels` for the soft
   prompt span, then calls `base_model(inputs_embeds=..., attention_mask=...,
   labels=...)`.
3. Only `soft_prompt` has `requires_grad=True`; `AdamW` only optimizes it;
   gradients never reach the base model.
4. `evaluate.py` / `test_forget_quality.py` build the **exact same**
   `[P_global] + probing_input` sequence via `generate_with_soft_prompt` (which
   calls `base_model.generate(inputs_embeds=...)`, with a manual KV-cached
   greedy fallback if that path misbehaves) and never feed `y_pos` in.

## Self-check results (what to look for)

1. **Only the soft prompt trains** — printed at start of training:
   `trainable params: <num_virtual_tokens * hidden_size> || all params: ~1.3B
   || trainable%: ~0.00xx` and `only soft_prompt requires grad: True`.
2. **Labels mask the context correctly** — printed for the first 2 examples:
   `input_len`, `target_len`, and `num labels != -100`, with an assertion that
   the latter equals `target_len` (i.e. exactly the `y_pos` + EOS tokens, never
   `probing_input`).
3. **Generation never sees `y_pos`** — `generate_predictions()` tokenizes only
   `probing_input`; `y_pos` is carried through purely as the `"y_pos"` field of
   the output record, used only for scoring.
4. **API-quality check uses the sample's own alias map** — `contains_api()`
   matches both the canonical dotted path (`numpy.prod`) and any alias in
   `alias dict` whose canonical value equals the target API (`np.prod`).
5. **Memory** — `per_device_train_batch_size=2` + `gradient_accumulation_steps=8`
   + `--bf16`/`--fp16` + only saving `soft_prompt.pt` (a few hundred KB, not
   the 1.3B base model) keeps things well within an A100's memory.

## Common pitfalls & fixes

- **`pad_token` is `None`** — DeepSeek-Coder's tokenizer has no pad token by
  default; both the wrapper and the scripts set `tokenizer.pad_token =
  tokenizer.eos_token` before building the collator/model.
- **`generate(inputs_embeds=...)` returns only new tokens** — when generating
  from embeddings, HF cannot map them back to `input_ids`, so the prompt is
  *not* echoed back. `generate_with_soft_prompt` decodes the raw output as the
  full prediction (no slicing by input length needed). If your transformers
  version raises on this path, the wrapper automatically falls back to a
  manual KV-cached greedy loop that still uses the soft prompt.
- **OOM** — lower `per_device_train_batch_size`, raise
  `gradient_accumulation_steps`, keep `--bf16` (A100) or `--fp16`, and don't
  remove the `torch.no_grad()` guards in evaluation/generation.
- **Loss not decreasing / soft prompt not updating** — check the printed
  `trainable params` line; if it's not roughly `num_virtual_tokens *
  hidden_size`, something is wrong with freezing (e.g. a stray
  `requires_grad_(True)` somewhere, or the optimizer was built on
  `model.parameters()` instead of `model.trainable_parameters()`).
- **Mismatched `num_virtual_tokens` at eval time** — `evaluate.py` /
  `test_forget_quality.py` read `num_virtual_tokens` from
  `prompt_config.json` in `--checkpoint_dir`, so the wrapper shape always
  matches the saved `soft_prompt.pt`.
- **Batched generation with a prepended soft prompt** is intentionally *not*
  used — left-padding would insert pad tokens between `P_global` and
  `probing_input`, breaking the intended `[P_global] + probing_input`
  adjacency. Generation runs one sample at a time; use `--limit` for quick
  checks on a subset.
