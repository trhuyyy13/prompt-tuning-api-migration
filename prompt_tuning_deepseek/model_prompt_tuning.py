"""Global Prompt Tuning wrapper around a frozen DeepSeek-Coder causal LM.

Only `self.soft_prompt` (a [num_virtual_tokens, hidden_size] matrix of
trainable virtual-token embeddings, P_global) receives gradients. It is
prepended to the token embeddings of `probing_input` -- never to `y_pos`,
which only exists in `labels` for teacher forcing.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import count_trainable_parameters

PROMPT_INIT_CHOICES = ("random", "vocab", "text")


class SoftPromptCausalLM(nn.Module):
    def __init__(self, model_name_or_path, num_virtual_tokens=20, prompt_init="random",
                 prompt_init_text=None, tokenizer=None, torch_dtype=None):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True,
        )
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze the entire base model -- only the soft prompt is trainable.
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.eval()

        hidden_size = self.base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.empty(num_virtual_tokens, hidden_size))
        self._init_soft_prompt(prompt_init, prompt_init_text)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_soft_prompt(self, prompt_init, prompt_init_text):
        if prompt_init not in PROMPT_INIT_CHOICES:
            raise ValueError(f"Unknown prompt_init={prompt_init!r}, expected one of {PROMPT_INIT_CHOICES}")

        embedding = self.base_model.get_input_embeddings()
        with torch.no_grad():
            if prompt_init == "random":
                nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
                return

            if prompt_init == "text":
                if not prompt_init_text:
                    raise ValueError('prompt_init="text" requires --prompt_init_text')
                ids = self.tokenizer(prompt_init_text, add_special_tokens=False).input_ids
                if not ids:
                    raise ValueError("prompt_init_text tokenized to an empty sequence")
                if len(ids) >= self.num_virtual_tokens:
                    ids = ids[: self.num_virtual_tokens]
                else:
                    reps = -(-self.num_virtual_tokens // len(ids))  # ceil division
                    ids = (ids * reps)[: self.num_virtual_tokens]
            else:  # "vocab": sample random real-token embeddings
                vocab_size = embedding.weight.size(0)
                ids = torch.randint(low=0, high=vocab_size, size=(self.num_virtual_tokens,)).tolist()

            init_ids = torch.tensor(ids, device=embedding.weight.device, dtype=torch.long)
            init_embeds = embedding(init_ids).detach().to(self.soft_prompt.dtype)
            self.soft_prompt.copy_(init_embeds)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def report_trainable_parameters(self):
        trainable, total = count_trainable_parameters(self)
        print(f"trainable params: {trainable:,} || all params: {total:,} "
              f"|| trainable%: {100 * trainable / total:.6f}")
        only_soft_prompt = all(
            p.requires_grad == (name == "soft_prompt")
            for name, p in self.named_parameters()
        )
        print(f"only soft_prompt requires grad: {only_soft_prompt}")
        return trainable, total

    # ------------------------------------------------------------------
    # Core: prepend P_global to the embeddings of probing_input (+ y_pos for TF)
    # ------------------------------------------------------------------
    def _prepend_soft_prompt(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        token_embeds = self.base_model.get_input_embeddings()(input_ids)

        soft_prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        soft_prompt_embeds = soft_prompt_embeds.to(token_embeds.dtype)
        inputs_embeds = torch.cat([soft_prompt_embeds, token_embeds], dim=1)

        soft_attention = torch.ones(
            batch_size, self.num_virtual_tokens,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        attention_mask = torch.cat([soft_attention, attention_mask], dim=1)
        return inputs_embeds, attention_mask

    def forward(self, input_ids, attention_mask, labels=None):
        inputs_embeds, attention_mask = self._prepend_soft_prompt(input_ids, attention_mask)

        if labels is not None:
            batch_size = input_ids.size(0)
            soft_labels = torch.full(
                (batch_size, self.num_virtual_tokens), -100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([soft_labels, labels], dim=1)

        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    # ------------------------------------------------------------------
    # Generation: [P_global] + probing_input -> generated continuation.
    # `y_pos` must NEVER be part of `input_ids` here.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_with_soft_prompt(self, input_ids, attention_mask, max_new_tokens=128,
                                  do_sample=False, num_beams=1, **gen_kwargs):
        inputs_embeds, attention_mask = self._prepend_soft_prompt(input_ids, attention_mask)

        try:
            # `generate(inputs_embeds=...)` returns ONLY the newly generated
            # token ids (HF cannot map embeddings back to the original prompt
            # ids, so the prompt is not echoed back) -- decode as-is.
            return self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
        except (NotImplementedError, RuntimeError, TypeError) as err:
            print(f"[generate_with_soft_prompt] base_model.generate(inputs_embeds=...) "
                  f"failed ({err!r}); falling back to manual greedy decoding "
                  f"(soft prompt is still used).")
            if num_beams != 1 or do_sample:
                raise RuntimeError(
                    "Fallback decoding only supports greedy search "
                    "(num_beams=1, do_sample=False)."
                ) from err
            return self._greedy_decode_fallback(inputs_embeds, attention_mask, max_new_tokens)

    @torch.no_grad()
    def _greedy_decode_fallback(self, inputs_embeds, attention_mask, max_new_tokens):
        """Manual KV-cached greedy decode that keeps using the soft-prompted embeddings."""
        embedding_layer = self.base_model.get_input_embeddings()
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device

        generated = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        cur_embeds, cur_mask, past_key_values = inputs_embeds, attention_mask, None
        for _ in range(max_new_tokens):
            outputs = self.base_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_ids = outputs.logits[:, -1, :].argmax(dim=-1)
            if eos_token_id is not None:
                next_ids = torch.where(finished, torch.full_like(next_ids, eos_token_id), next_ids)
                finished = finished | (next_ids == eos_token_id)

            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            if eos_token_id is not None and finished.all():
                break

            past_key_values = outputs.past_key_values
            cur_embeds = embedding_layer(next_ids).unsqueeze(1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((batch_size, 1), dtype=cur_mask.dtype, device=device)], dim=1,
            )
        return generated
