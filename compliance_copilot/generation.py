"""Local Transformers backend. No API key or hosted inference is required."""

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class LocalGenerator:
    def __init__(self, model_name=DEFAULT_MODEL, *, revision=None, max_new_tokens=768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device).eval()
        self.revision = self.model.config._commit_hash

    def __call__(self, messages):
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        if inputs.input_ids.shape[1] + self.max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Retrieved context exceeds model context window")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
