"""Train a global soft prompt P_global for DeepSeek-Coder-1.3B-base.

Mapping learned:           [P_global] + probing_input  ->  y_pos
Loss:                      standard causal-LM cross entropy, computed ONLY on
                           y_pos tokens (probing_input and P_global are masked
                           with -100 in `labels`).
Trainable parameters:      ONLY `model.soft_prompt`. The 1.3B base model is
                           frozen end to end; nothing of it is saved either.
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import DataCollatorForPromptTuning, PromptTuningDataset
from model_prompt_tuning import PROMPT_INIT_CHOICES, SoftPromptCausalLM
from utils import load_soft_prompt_checkpoint, save_checkpoint, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Global Prompt Tuning for DeepSeek-Coder-1.3B")

    p.add_argument("--model_name_or_path", default="deepseek-ai/deepseek-coder-1.3b-base")
    p.add_argument("--train_file", required=True)
    p.add_argument("--valid_file", default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--num_virtual_tokens", type=int, default=20)
    p.add_argument("--prompt_init", choices=PROMPT_INIT_CHOICES, default="random")
    p.add_argument("--prompt_init_text", default="Generate the migrated API line:")

    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=640)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-3)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)

    # Either `--resume_from_checkpoint` (resume from --output_dir) or
    # `--resume_from_checkpoint /path/to/ckpt`.
    p.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# Debug: prove that probing_input is context and y_pos is the only target
# ----------------------------------------------------------------------
def debug_first_examples(dataset, tokenizer, num_examples=2):
    print("=" * 70)
    print("[DEBUG] sanity check on the first examples (labels masking)")
    for i in range(min(num_examples, len(dataset))):
        ex = dataset[i]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        input_len, target_len = ex["input_len"], ex["target_len"]
        num_label_tokens = sum(1 for l in labels if l != -100)

        context_ids = input_ids[:input_len]
        target_ids = input_ids[input_len:]

        print(f"--- example {i} ---")
        print(f"input_len (probing_input tokens)         = {input_len}")
        print(f"target_len (y_pos tokens incl. EOS)       = {target_len}")
        print(f"num labels != -100 (should == target_len) = {num_label_tokens}")
        assert num_label_tokens == target_len, "labels must mask exactly the probing_input part!"
        print(f"[context decoded] ...{tokenizer.decode(context_ids, skip_special_tokens=False)[-300:]!r}")
        print(f"[target  decoded]     {tokenizer.decode(target_ids, skip_special_tokens=False)!r}")
    print("=" * 70)


@torch.no_grad()
def evaluate_loss(model, dataloader, device, autocast_dtype):
    model.eval()
    total_loss, num_batches = 0.0, 0
    use_amp = autocast_dtype is not None and device.type == "cuda"
    for batch in dataloader:
        batch = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device = {device}")

    if args.fp16 and args.bf16:
        raise ValueError("Pass only one of --fp16 / --bf16")
    autocast_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    use_amp = autocast_dtype is not None and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # ---- tokenizer & model -------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SoftPromptCausalLM(
        model_name_or_path=args.model_name_or_path,
        num_virtual_tokens=args.num_virtual_tokens,
        prompt_init=args.prompt_init,
        prompt_init_text=args.prompt_init_text,
        tokenizer=tokenizer,
    )
    model.to(device)
    model.report_trainable_parameters()

    # ---- data ---------------------------------------------------------------
    collator = DataCollatorForPromptTuning(tokenizer)
    train_dataset = PromptTuningDataset(args.train_file, tokenizer, args.max_input_length,
                                        args.max_target_length, args.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                              shuffle=True, collate_fn=collator, num_workers=args.num_workers,
                              drop_last=False)
    print(f"[*] train examples = {len(train_dataset)}, steps/epoch = {len(train_loader)}")

    eval_loader = None
    if args.valid_file:
        valid_dataset = PromptTuningDataset(args.valid_file, tokenizer, args.max_input_length,
                                            args.max_target_length, args.max_seq_length)
        eval_loader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                 shuffle=False, collate_fn=collator, num_workers=args.num_workers)
        print(f"[*] valid examples = {len(valid_dataset)}")

    debug_first_examples(train_dataset, tokenizer)

    # ---- optimizer / scheduler ----------------------------------------------
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    total_steps = updates_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)
    print(f"[*] total optimizer steps = {total_steps} (warmup = {warmup_steps})")

    # ---- optionally resume ---------------------------------------------------
    start_epoch, global_step = 0, 0
    if args.resume_from_checkpoint:
        ckpt_dir = args.output_dir if args.resume_from_checkpoint is True else args.resume_from_checkpoint
        soft_prompt_path = os.path.join(ckpt_dir, "soft_prompt.pt")
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.isfile(soft_prompt_path):
            tensor = load_soft_prompt_checkpoint(ckpt_dir)
            with torch.no_grad():
                model.soft_prompt.copy_(tensor.to(device=model.soft_prompt.device, dtype=model.soft_prompt.dtype))
            print(f"[resume] loaded soft prompt from {soft_prompt_path}")
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu")
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            global_step = state.get("global_step", 0)
            start_epoch = state.get("epoch", 0)
            print(f"[resume] resuming at epoch={start_epoch}, global_step={global_step}")
        else:
            print(f"[resume] no training_state.pt under {ckpt_dir}; starting optimizer from scratch")

    # ---- persist configs (so evaluate.py / test_forget_quality.py can rebuild the wrapper) ----
    save_json(vars(args), os.path.join(args.output_dir, "training_args.json"))
    save_json({
        "model_name_or_path": args.model_name_or_path,
        "num_virtual_tokens": args.num_virtual_tokens,
        "prompt_init": args.prompt_init,
        "prompt_init_text": args.prompt_init_text,
    }, os.path.join(args.output_dir, "prompt_config.json"))

    # ---- training loop --------------------------------------------------------
    model.train()
    running_loss, running_steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.gradient_accumulation_steps
            running_steps += 1

            is_last_in_epoch = (step + 1) == len(train_loader)
            if (step + 1) % args.gradient_accumulation_steps == 0 or is_last_in_epoch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(running_steps, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"[train] epoch={epoch} step={global_step} loss={avg_loss:.4f} lr={lr:.6g}")
                    running_loss, running_steps = 0.0, 0

                if eval_loader is not None and global_step % args.eval_steps == 0:
                    eval_loss = evaluate_loss(model, eval_loader, device, autocast_dtype)
                    print(f"[eval ] step={global_step} eval_loss={eval_loss:.4f} ppl={math.exp(min(eval_loss, 20)):.4f}")

                if global_step % args.save_steps == 0:
                    save_checkpoint(args.output_dir, model, tokenizer, optimizer, scheduler,
                                    global_step=global_step, epoch=epoch)
                    print(f"[save ] checkpoint @ step={global_step} -> {args.output_dir}")

    save_checkpoint(args.output_dir, model, tokenizer, optimizer, scheduler,
                    global_step=global_step, epoch=args.num_train_epochs)
    print(f"[done ] final soft prompt saved to {os.path.join(args.output_dir, 'soft_prompt.pt')}")


if __name__ == "__main__":
    main()
