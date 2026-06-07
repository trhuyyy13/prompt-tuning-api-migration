import json
import torch
import csv
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def check_api_usage(generated_line, target_api, alias_dict):
    for key, value in alias_dict.items():
        if value == target_api:
            pattern = r'\b' + re.escape(key) + r'\b'
            
            if re.search(pattern, generated_line):
                return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dep_json = os.path.join(args.output_dir, "deprecated_api_cases.json")
    rep_json = os.path.join(args.output_dir, "replacement_api_cases.json")
    mis_json = os.path.join(args.output_dir, "mismatch_api_cases.json")
    dep_csv = os.path.join(args.output_dir, "deprecated_api_cases.csv")
    rep_csv = os.path.join(args.output_dir, "replacement_api_cases.csv")
    mis_csv = os.path.join(args.output_dir, "mismatch_api_cases.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device}")

    print(f"[*] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[*] Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"[*] Loading data from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_samples = len(data)
    mis_api_cases = []
    dep_api_cases = []
    rep_api_cases = []

    pbar = tqdm(range(0, num_samples, args.batch_size), desc="Processing Batches")

    for i in pbar:
        batch_data = data[i : i + args.batch_size]
        
        batch_prompts = []
        batch_meta = []

        for item in batch_data:
            code_context = item.get("probing input new", "")

            encoded_context = tokenizer.encode(code_context, add_special_tokens=False)
            if len(encoded_context) > args.max_prompt_length - 200:
                encoded_context = encoded_context[-(args.max_prompt_length - 200):]
                code_context = tokenizer.decode(encoded_context, skip_special_tokens=True).replace("\u0120", " ").replace("\u010a", "\n")
                
            prompt_text = code_context
            
            batch_prompts.append(prompt_text)
            batch_meta.append({
                "probing input"  : code_context,
                "deprecated api" : item.get("deprecated api", []),
                "replacement api": item.get("replacement api", ""),
                "alias dict"     : item.get("alias dict", {})
            })

        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(args.temperature > 0),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0 else 1.0, 
                top_p=args.top_p if args.temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for idx, generated_text in enumerate(generated_texts):
            meta = batch_meta[idx]
            global_id = i + idx

            generated_text = generated_text.replace("\u0120", " ").replace("\u010a", "\n")
            
            clean_text = generated_text.strip()
            first_line_generated = clean_text.split('\n')[0]

            last_line_of_prompt = meta["probing input"].split('\n')[-1]
            full_line_to_check = last_line_of_prompt + first_line_generated

            is_replacement = check_api_usage(full_line_to_check, meta["replacement api"], meta["alias dict"])
            is_deprecated = any(
                check_api_usage(full_line_to_check, api, meta["alias dict"]) 
                for api in meta["deprecated api"]
            )

            record = {
                "id"               : global_id,
                "probing input"    : meta["probing input"],
                "deprecated api"   : meta["deprecated api"],
                "replacement api"  : meta["replacement api"],
                "generated content": full_line_to_check
            }

            if is_replacement:
                rep_api_cases.append(record)
            elif is_deprecated:
                dep_api_cases.append(record)
            else:
                mis_api_cases.append(record)

        pbar.set_postfix({
            "good"    : len(rep_api_cases),
            "bad"     : len(dep_api_cases),
            "mismatch": len(mis_api_cases)
        })

    print("\n[*] Saving result files...")

    with open(mis_json, "w", encoding="utf-8") as f:
        json.dump(mis_api_cases, f, ensure_ascii=False, indent=4)
    
    with open(rep_json, "w", encoding="utf-8") as f:
        json.dump(rep_api_cases, f, ensure_ascii=False, indent=4)
        
    with open(dep_json, "w", encoding="utf-8") as f:
        json.dump(dep_api_cases, f, ensure_ascii=False, indent=4)

    csv_headers = ["id", "probing input", "deprecated api", "replacement api", "generated content"]

    with open(mis_csv, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(mis_api_cases)

    with open(rep_csv, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(rep_api_cases)
        
    with open(dep_csv, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(dep_api_cases)

    print("\n================ SUMMARY ================")
    print(f"Mismatched API cases       : {len(mis_api_cases)}/{num_samples}")
    print(f"Replacement API cases      : {len(rep_api_cases)}/{num_samples}")
    print(f"Deprecated API cases       : {len(dep_api_cases)}/{num_samples}")
    print("==========================================")

if __name__ == "__main__":
    main()
