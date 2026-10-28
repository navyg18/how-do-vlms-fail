import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def _default_cache_dir() -> Optional[str]:
    """Return a model cache directory if configured, else None."""
    env_cache = os.environ.get("HF_MODEL_CACHE")
    if env_cache:
        return env_cache

    local_models = Path(__file__).resolve().parent.parent / "models"
    return str(local_models) if local_models.exists() else None


@contextmanager
def session_hook(model: torch.nn.Module, hook: Callable):
    """Temporarily register a forward hook."""
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class BlockAttentionHook:
    """Mask attention between selected token pairs."""

    def __init__(self, indices_list):
        self.indices_list = indices_list

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs["hidden_states"]
        bsz, seq_len, _ = hidden_states.shape
        attention_mask = kwargs.get("attention_mask")

        # Build a causal mask if the layer did not receive one.
        if attention_mask is None:
            attention_mask = torch.ones(
                bsz,
                1,
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            ).tril(diagonal=0)
        else:
            attention_mask = attention_mask.clone()

        for i, j in self.indices_list:
            if attention_mask.dtype == torch.bool:
                attention_mask[:, :, i, j] = False
            else:
                attention_mask[:, :, i, j] = -1e9

        kwargs["attention_mask"] = attention_mask
        return args, kwargs


class HookedQwenVL:
    """Qwen-VL wrapper with hooks for interpretability experiments."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        hook_loc: str = "text_model_in",
        device: str = "cuda:0",
        quantize: bool = False,
        quantize_type: str = "fp16",
        attn_implementation: str = "sdpa",
        cache_dir: Optional[str] = None,
    ):
        # Attention-mask interventions only work reliably with eager attention.
        self.device = device
        self.hook_loc = hook_loc
        self.data = None
        self._text_layers = None
        self._text_layers_name = None
        self.cache_dir = cache_dir if cache_dir is not None else _default_cache_dir()

        self.model = self._load_model(
            model_id=model_id,
            quantize=quantize,
            quantize_type=quantize_type,
            attn_implementation=attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Useful Qwen-VL config fields.
        self.vision_token_id = self.model.config.video_token_id
        self.patch_size = self.model.config.vision_config.patch_size
        self.spatial_merge_size = self.model.config.vision_config.spatial_merge_size

    def _load_model(
        self,
        model_id: str,
        quantize: bool,
        quantize_type: str,
        attn_implementation: str,
    ):
        """Load the model with the requested precision/quantization."""
        common_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": self.device,
            "cache_dir": self.cache_dir,
        }

        if not quantize:
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                **common_kwargs,
            )

        if quantize_type == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                **common_kwargs,
            )

        if quantize_type == "fp16":
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                attn_implementation=attn_implementation,
                **common_kwargs,
            )

        if quantize_type == "int8":
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.int8,
                **common_kwargs,
            )

        raise ValueError(f"Unsupported quantize_type: {quantize_type}")

    def _prepare_image(self, image_path_or_image: Union[str, Image.Image]) -> Image.Image:
        """Load an image from path or return the provided PIL image."""
        if isinstance(image_path_or_image, str):
            return Image.open(image_path_or_image).convert("RGB")
        return image_path_or_image

    def _build_inputs(self, image_path_or_image: Union[str, Image.Image], prompt: str):
        """Build model inputs from an image and user prompt."""
        image = self._prepare_image(image_path_or_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        return image, inputs.to(self.device)

    @contextmanager
    def ablate_inputs(self, indices, replacement_tensor):
        """Replace selected input embeddings before the language model forward pass."""

        def ablation_hook(module, args, kwargs):
            input_embeds = kwargs.get("inputs_embeds")
            if input_embeds is None or input_embeds.shape[-2] == 1:
                return args, kwargs

            modified_input = input_embeds.clone()
            local_replacement = replacement_tensor.to(
                modified_input.device,
                dtype=modified_input.dtype,
            )

            if local_replacement.dim() == 1:
                local_replacement = local_replacement.unsqueeze(0)
            local_replacement = local_replacement.expand(len(indices), -1)

            modified_input[:, indices, :] = local_replacement
            kwargs["inputs_embeds"] = modified_input
            return args, kwargs

        handle = self.model.model.register_forward_pre_hook(
            ablation_hook,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def patch_at_layer(self, layer_idx: int, indices: list, replacement: torch.Tensor):
        """Replace selected token states at one transformer layer."""

        def transplant_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None or hidden.dim() != 3 or hidden.shape[1] == 1:
                return output

            rep = replacement.to(hidden.device, dtype=hidden.dtype)
            modified = hidden.clone()
            for pos in indices:
                if pos < modified.shape[1] and pos < rep.shape[0]:
                    modified[:, pos, :] = rep[pos]

            return (modified,) + output[1:] if isinstance(output, tuple) else modified

        layers = self._get_text_layers()
        handle = layers[layer_idx].register_forward_hook(transplant_hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def patch_at_layer_batched(self, layer_idx: int, positions_list: list, replacement: torch.Tensor):
        """Batched version of patch_at_layer for per-example token positions."""

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None or hidden.dim() != 3 or hidden.shape[1] == 1:
                return output

            rep = replacement.to(hidden.device, dtype=hidden.dtype)
            modified = hidden.clone()
            for batch_idx, positions in enumerate(positions_list):
                if batch_idx >= modified.shape[0]:
                    break
                for pos in positions:
                    if pos < modified.shape[1] and pos < rep.shape[0]:
                        modified[batch_idx, pos, :] = rep[pos]

            return (modified,) + output[1:] if isinstance(output, tuple) else modified

        layers = self._get_text_layers()
        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def debug_print_submodules(self):
        """Print candidate transformer stacks to help locate language layers."""
        print("=== self.model ===", type(self.model))
        print("=== self.model.model ===", type(self.model.model))
        hits = 0
        for name, mod in self.model.named_modules():
            if name.endswith(("layers", "h", "blocks")):
                try:
                    length = len(mod)
                except Exception:
                    length = None
                print(f"CANDIDATE: {name} | type={type(mod)} | len={length}")
                hits += 1
        print("total candidates:", hits)

    def _get_text_layers(self):
        """Locate and cache the language-model transformer blocks."""
        if self._text_layers is not None:
            return self._text_layers

        target_n = self.model.config.num_hidden_layers
        candidates = []

        for name, mod in self.model.named_modules():
            if name.endswith(("layers", "h", "blocks")):
                try:
                    length = len(mod)
                except Exception:
                    continue
                if length == target_n:
                    candidates.append((name, mod))

        if len(candidates) == 1:
            name, mod = candidates[0]
        elif len(candidates) > 1:
            candidates.sort(
                key=lambda x: (("language" not in x[0] and "text" not in x[0]), x[0])
            )
            name, mod = candidates[0]
        else:
            raise AttributeError(
                "Could not locate language transformer layers. "
                f"Expected a ModuleList with len={target_n}."
            )

        self._text_layers = mod
        self._text_layers_name = name
        print(f"[HookedQwenVL] Using language layers at: {name} (len={len(mod)})")
        return mod

    @contextmanager
    def ablate_at_layer(self, layer_idx, indices, replacement_tensor):
        """Replace selected hidden states before a given layer runs."""

        def layer_ablation_hook(module, args, kwargs):
            if not args:
                return args, kwargs

            hidden_states = args[0]
            if hidden_states is None or hidden_states.dim() != 3 or hidden_states.shape[1] == 1:
                return args, kwargs

            modified_hidden = hidden_states.clone()
            rep = replacement_tensor.to(
                modified_hidden.device,
                dtype=modified_hidden.dtype,
            )
            if rep.dim() == 1:
                rep = rep.unsqueeze(0)
            rep = rep.expand(len(indices), -1)

            modified_hidden[:, indices, :] = rep
            return (modified_hidden,) + args[1:], kwargs

        layers = self._get_text_layers()
        handle = layers[layer_idx].register_forward_pre_hook(
            layer_ablation_hook,
            with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def block_attention(self, attn_block_dict):
        """Block attention edges for specific layers and token pairs."""
        handles = []
        layers = self._get_text_layers()

        try:
            for layer, indices_list in attn_block_dict.items():
                hook = BlockAttentionHook(indices_list)
                handle = layers[layer].self_attn.register_forward_pre_hook(
                    hook,
                    with_kwargs=True,
                )
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                handle.remove()

    def prompt_hook(self, module, args, kwargs, output):
        """Store input embeddings seen by the hooked module."""
        self.data = kwargs.get("inputs_embeds")
        return output

    def forward(
        self,
        image_path_or_image: Union[str, Image.Image],
        prompt: str,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ):
        """Run a forward pass."""
        _, inputs = self._build_inputs(image_path_or_image, prompt)
        with torch.no_grad():
            return self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

    def generate(
        self,
        image_path_or_image: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: int = 100,
        output_hidden_states: bool = False,
        do_sample: bool = True,
    ):
        """Generate a response for an image-prompt pair."""
        _, inputs = self._build_inputs(image_path_or_image, prompt)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=True,
                do_sample=do_sample,
            )

        response = self.processor.batch_decode(
            output.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return (response, output.hidden_states) if output_hidden_states else response

    def get_text_model_in(self, image_path_or_image, prompt):
        """Capture the model input embeddings for a given example."""
        _, inputs = self._build_inputs(image_path_or_image, prompt)
        captured = {}

        def embed_hook(module, args, kwargs, output):
            if args:
                captured["inputs_embeds"] = args[0]
            return output

        embedding_layer = self.model.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(embed_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()

        return captured["inputs_embeds"]

    def get_layer_output(
        self,
        image_path_or_image: Union[str, Image.Image],
        prompt: str,
        layer_idx: int,
    ):
        """Return hidden states for one layer."""
        outputs = self.forward(
            image_path_or_image,
            prompt,
            output_hidden_states=True,
        )
        return outputs.hidden_states[layer_idx]

    def get_vision_token_positions(self, inputs):
        """Return token indices corresponding to <|image_pad|>."""
        image_pad_token = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        input_ids = inputs["input_ids"][0]
        vision_indices = torch.where(input_ids == image_pad_token)[0]
        return vision_indices.tolist()
