import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from contextlib import contextmanager
from typing import Callable, Union
import os

# Optional model cache folder; None uses the Hugging Face default ($HF_HOME).
model_cache_dir = os.environ.get("HF_MODEL_CACHE")


@contextmanager
def session_hook(model: torch.nn.Module, hook: Callable):
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class BlockAttentionHook:
    def __init__(self, indices_list):
        self.indices_list = indices_list  # [(i, j), ]

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs['hidden_states']
        bsz, seq_len, _ = hidden_states.shape
        attention_mask = kwargs.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones(
                bsz, 1, seq_len, seq_len,
                dtype=torch.bool,
                device=hidden_states.device
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


class HookedLLaVA:
    """
    Hooked LLaVA-1.5 for interpretability analysis.

    Supports: llava-hf/llava-1.5-7b-hf (and other llava-hf variants)

    Architecture notes:
    - model.vision_tower          — CLIP vision encoder
    - model.multi_modal_projector — linear projection into LLM space
    - model.language_model.model.layers — LLaMA transformer blocks
    - Image token id: config.image_token_index (32000 for LLaVA-1.5)
    """

    def __init__(self,
                 model_id: str = "llava-hf/llava-1.5-7b-hf",
                 hook_loc: str = "text_model_in",
                 device: str = "cuda:0",
                 quantize: bool = False,
                 quantize_type: str = "fp16",
                 attn_implementation: str = "eager"):
        # NOTE: block_attention / ablate_at_layer hooks modify kwargs["attention_mask"]
        # inside self_attn pre-hooks. This only works with attn_implementation="eager".
        if quantize:
            if quantize_type == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
            elif quantize_type == "fp16":
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_implementation,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
            elif quantize_type == "int8":
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    dtype=torch.int8,
                    low_cpu_mem_usage=True,
                    device_map=device,
                    cache_dir=model_cache_dir
                )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map=device,
                cache_dir=model_cache_dir,
            )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.hook_loc = hook_loc
        self.data = None
        self.device = device

        # LLaVA-specific config
        self.image_token_index = self.model.config.image_token_index  # 32000
        vision_cfg = self.model.config.vision_config
        self.patch_size = vision_cfg.patch_size  # 14 for CLIP ViT-L/14

    def _prepare_inputs(self, image: Image.Image, prompt: str):
        """Build processor inputs for LLaVA using chat template."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
        )
        return inputs

    @contextmanager
    def ablate_inputs(self, indices, replacement_tensor):
        """
        Ablation context manager for LLaVA.
        Replaces vision token embeddings at specified indices.
        Hook is registered on the language model's embedding entry point.
        """
        def ablation_hook(module, args, kwargs):
            input_embeds = kwargs.get("inputs_embeds")
            if input_embeds is None:
                return args, kwargs

            # Skip decode steps (single token)
            if input_embeds.shape[-2] == 1:
                return args, kwargs

            modified_input = input_embeds.clone()

            local_replacement_tensor = replacement_tensor.to(
                modified_input.dtype
            ).to(modified_input.device)

            if local_replacement_tensor.dim() == 1:
                local_replacement_tensor = local_replacement_tensor.unsqueeze(0)

            local_replacement_tensor = local_replacement_tensor.expand(
                len(indices), -1
            )

            modified_input[:, indices, :] = local_replacement_tensor
            kwargs["inputs_embeds"] = modified_input

            return args, kwargs

        # LLaVA: language model is at model.language_model
        hook = self.model.language_model.register_forward_pre_hook(
            ablation_hook, with_kwargs=True
        )
        try:
            yield
        finally:
            hook.remove()

    @contextmanager
    def patch_at_layer(self, layer_idx: int, indices: list, replacement: torch.Tensor):
        """
        Transplant hook: replaces hidden_states[:, indices[i], :] with
        replacement[indices[i], :] at a specific transformer layer.
        """
        def transplant_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is None or h.dim() != 3 or h.shape[1] == 1:
                return
            rep = replacement.to(h.device, dtype=h.dtype)
            modified = h.clone()
            for pos in indices:
                if pos < modified.shape[1] and pos < rep.shape[0]:
                    modified[:, pos, :] = rep[pos]
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        layers = self._get_text_layers()
        hook = layers[layer_idx].register_forward_hook(transplant_hook)
        try:
            yield
        finally:
            hook.remove()

    @contextmanager
    def patch_at_layer_batched(self, layer_idx: int, positions_list: list, replacement: torch.Tensor):
        """
        Batched variant of patch_at_layer.
        positions_list[i] = token positions to ablate for batch item i.
        replacement: [seq, hidden] mean activations — same for all batch items.
        """
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if h is None or h.dim() != 3 or h.shape[1] == 1:
                return
            rep = replacement.to(device=h.device, dtype=h.dtype)
            modified = h.clone()
            for i, positions in enumerate(positions_list):
                if i >= modified.shape[0]:
                    break
                for pos in positions:
                    if pos < modified.shape[1] and pos < rep.shape[0]:
                        modified[i, pos, :] = rep[pos]
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        layers = self._get_text_layers()
        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def debug_print_submodules(self):
        print("=== self.model ===", type(self.model))
        print("=== self.model.language_model ===", type(self.model.language_model))
        hits = 0
        for name, mod in self.model.named_modules():
            if name.endswith(("layers", "h", "blocks")):
                try:
                    n = len(mod)
                except Exception:
                    n = None
                print(f"CANDIDATE: {name} | type={type(mod)} | len={n}")
                hits += 1
        print("total candidates:", hits)

    def _get_text_layers(self):
        """
        Find the transformer block list for the LLaVA language model.
        LLaVA-1.5: model.language_model.model.layers
        """
        if getattr(self, "_text_layers", None) is not None:
            return self._text_layers

        target_n = self.model.config.text_config.num_hidden_layers
        candidates = []

        for name, mod in self.model.named_modules():
            if name.endswith(("layers", "h", "blocks")):
                try:
                    n = len(mod)
                except Exception:
                    continue
                if n == target_n:
                    candidates.append((name, mod))

        if len(candidates) == 1:
            name, mod = candidates[0]
            self._text_layers = mod
            self._text_layers_name = name
            print(f"[HookedLLaVA] Using language layers at: {name} (len={len(mod)})")
            return mod

        if len(candidates) > 1:
            candidates.sort(key=lambda x: (("language" not in x[0] and "text" not in x[0]), x[0]))
            name, mod = candidates[0]
            self._text_layers = mod
            self._text_layers_name = name
            print(f"[HookedLLaVA] Multiple candidates; using: {name} (len={len(mod)})")
            return mod

        raise AttributeError(
            f"Could not locate language transformer layers. "
            f"(expected ModuleList len={target_n}). "
            f"Run debug_print_submodules() to inspect."
        )

    @contextmanager
    def ablate_at_layer(self, layer_idx, indices, replacement_tensor):
        def layer_ablation_hook(module, args, kwargs):
            if len(args) == 0:
                return

            hidden_states = args[0]
            if hidden_states is None or hidden_states.dim() != 3:
                return

            if hidden_states.shape[1] == 1:
                return

            modified_hidden = hidden_states.clone()

            rep = replacement_tensor.to(modified_hidden.device, dtype=modified_hidden.dtype)
            if rep.dim() == 1:
                rep = rep.unsqueeze(0)
            rep = rep.expand(len(indices), -1)

            modified_hidden[:, indices, :] = rep
            new_args = (modified_hidden,) + args[1:]
            return new_args, kwargs

        layers = self._get_text_layers()
        hook = layers[layer_idx].register_forward_pre_hook(layer_ablation_hook, with_kwargs=True)
        try:
            yield
        finally:
            hook.remove()

    @contextmanager
    def block_attention(self, attn_block_dict):
        """
        Block attention between specific token pairs.
        attn_block_dict: {layer: [(i, j), ...]}
        """
        hooks = []

        for layer, indices_list in attn_block_dict.items():
            hook = BlockAttentionHook(indices_list)
            layers = self._get_text_layers()
            h = layers[layer].self_attn.register_forward_pre_hook(
                hook, with_kwargs=True
            )
            hooks.append(h)

        has_error = False
        try:
            yield
        except Exception as e:
            import traceback
            traceback.print_exc()
            has_error = True
        finally:
            for h in hooks:
                h.remove()
            if has_error:
                raise Exception("Error during block_attention context")

    def prompt_hook(self, module, args, kwargs, output):
        """Hook to capture input embeddings"""
        self.data = kwargs.get('inputs_embeds')
        return output

    def forward(self,
                image_path_or_image: Union[str, Image.Image],
                prompt: str,
                output_hidden_states=False,
                output_attentions=False):
        """Forward pass through LLaVA."""
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image).convert("RGB")
        else:
            image = image_path_or_image

        inputs = self._prepare_inputs(image, prompt).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions
            )

        return outputs

    def generate(self,
                 image_path_or_image: Union[str, Image.Image],
                 prompt: str,
                 max_new_tokens: int = 100,
                 output_hidden_states: bool = False,
                 do_sample=True):
        """Generate text response from LLaVA."""
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image).convert("RGB")
        else:
            image = image_path_or_image

        inputs = self._prepare_inputs(image, prompt).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=True,
                do_sample=do_sample
            )

        response_str = self.processor.batch_decode(
            output.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        if output_hidden_states:
            return response_str, output.hidden_states

        return response_str

    def get_text_model_in(self, image_path_or_image, prompt):
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image).convert("RGB")
        else:
            image = image_path_or_image

        inputs = self._prepare_inputs(image, prompt).to(self.device)

        captured = {}

        def embed_hook(module, args, kwargs):
            captured["inputs_embeds"] = args[0]
            return None

        embedding_layer = self.model.get_input_embeddings()
        handle = embedding_layer.register_forward_hook(embed_hook)
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()

        return captured["inputs_embeds"]

    def get_layer_output(self,
                         image_path_or_image: Union[str, Image.Image],
                         prompt: str,
                         layer_idx: int):
        """Get hidden states at a specific layer."""
        outputs = self.forward(
            image_path_or_image,
            prompt,
            output_hidden_states=True
        )
        return outputs.hidden_states[layer_idx]

    def get_vision_token_positions(self, inputs):
        """
        Extract vision token positions from processed inputs.
        LLaVA-1.5: image tokens are config.image_token_index (32000).
        """
        input_ids = inputs["input_ids"][0]
        vision_mask = (input_ids == self.image_token_index)
        vision_indices = torch.where(vision_mask)[0]

        if len(vision_indices) == 0:
            return []

        return vision_indices.tolist()
