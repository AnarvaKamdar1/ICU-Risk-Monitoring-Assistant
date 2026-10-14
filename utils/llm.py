"""LLM summariser: MedGemma (thinking model) or any general instruct model, 4-bit, with template fallback."""
import os
import re

import torch

from prompting import SYSTEM_PROMPT, build_prompt, template_summary

THOUGHT_BLOCKS = [(r"<unused94>", r"<unused95>"),       # MedGemma 1.5 / Gemma thinking
                  (r"<think>", r"</think>")]           # Qwen3 / DeepSeek-style
SPECIAL_TOKENS = re.compile(r"<(?:unused\d+|end_of_turn|start_of_turn|eos|bos|pad)>|<\|[^|>]*\|>")


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def strip_thinking(raw):
    """Return (final_answer, had_thinking). final_answer is None if the model never finished thinking."""
    text, had = raw, False
    for start, end in THOUGHT_BLOCKS:
        if re.search(start, text):
            had = True
            parts = re.split(end, text, maxsplit=1)
            if len(parts) < 2:
                return None, True                      # ran out of tokens while still thinking
            text = parts[1]
    text = SPECIAL_TOKENS.sub("", text)
    text = re.sub(r"^\s*(thought|final answer:?)\s*", "", text, flags=re.I).strip()
    return text, had


class Summarizer:
    """mode: 'llm' (any HF model) or 'template'. Gemma-3/MedGemma checkpoints load as image-text-to-text."""

    def __init__(self, mode="llm", model_id="Qwen/Qwen3-4B-Instruct-2507", max_new_tokens=600, load_4bit=True):
        self.mode = "llm" if mode in ("llm", "medgemma") else mode
        self.model_id, self.max_new_tokens, self.load_4bit = model_id, max_new_tokens, load_4bit
        self.model = self.proc = None
        self.load_error = None
        self.is_gemma = "gemma" in model_id.lower()
        if self.mode == "llm":
            try:
                self._load()
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"
                self.mode = "template"

    def _load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("LLM summariser needs a GPU")
        import transformers as tf
        tok = hf_token()
        # Gemma 3 overflows in fp16 -> fp32 compute; other models are fine in fp16 on a T4
        dtype = torch.float32 if self.is_gemma else torch.float16
        kw = dict(device_map={"": 0}, token=tok)
        if self.load_4bit:
            kw["quantization_config"] = tf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        cls = tf.AutoModelForImageTextToText if self.is_gemma else tf.AutoModelForCausalLM
        try:
            self.model = cls.from_pretrained(self.model_id, dtype=dtype, **kw)
        except TypeError:
            self.model = cls.from_pretrained(self.model_id, torch_dtype=dtype, **kw)
        self.proc = (tf.AutoProcessor if self.is_gemma else tf.AutoTokenizer).from_pretrained(self.model_id, token=tok)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, system, user, skip_thinking=True):
        if self.is_gemma and skip_thinking:
            # MedGemma 1.5 thinks before answering. Pre-fill an EMPTY thought block so it answers directly.
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": user}]}]
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            text += "<unused94>thought\n<unused95>"
            inputs = self.proc.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            n_in = inputs["input_ids"].shape[-1]
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.05)
            return self.proc.tokenizer.decode(out[0][n_in:], skip_special_tokens=False)
        if self.is_gemma:
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": user}]}]
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra = {} if self.is_gemma else {"enable_thinking": False}     # ignored by templates without the switch
        inputs = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", **extra).to(self.model.device)
        n_in = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  repetition_penalty=1.05)
        dec = self.proc.tokenizer if hasattr(self.proc, "tokenizer") else self.proc
        return dec.decode(out[0][n_in:], skip_special_tokens=False)

    def summarize(self, summary):
        if self.mode != "llm":
            return template_summary(summary), "template"
        system = SYSTEM_PROMPT + "\nOutput ONLY the final summary with the six section headings. Do not show your reasoning."
        note = "empty output"
        for skip in ([True, False] if self.is_gemma else [False]):     # Gemma: try no-thinking first, then normal
            try:
                raw = self._generate(system, build_prompt(summary), skip_thinking=skip)
                text, had_thinking = strip_thinking(raw)
                if text and len(text) > 40:
                    return text, self.model_id.split("/")[-1] + (" (thinking removed)" if had_thinking else "")
                note = "model ran out of tokens while thinking" if had_thinking else "empty output"
            except Exception as e:
                note = f"{type(e).__name__}: {e}"
        return template_summary(summary) + f"\n[template fallback - {note}]", "template-fallback"
