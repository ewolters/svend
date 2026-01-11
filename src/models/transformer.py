"""
Complete transformer model for reasoning tasks.

This is the core model that can be instantiated at any scale
(500M to 7B+) using the configuration system.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TransformerConfig
from .layers import RMSNorm, TransformerBlock, create_causal_mask


class ReasoningTransformer(nn.Module):
    """
    Custom transformer model optimized for reasoning tasks.

    Features:
    - Pre-norm architecture (stable training)
    - RoPE positional embeddings (length generalization)
    - Grouped Query Attention (memory efficiency)
    - SwiGLU activation (gradient flow)
    - Optional gradient checkpointing (memory/compute tradeoff)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
        return_logits: bool = True,
    ) -> dict:
        """
        Forward pass through the transformer.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask for padding tokens [batch_size, seq_length]
            position_ids: Position indices [batch_size, seq_length]
            past_key_values: Cached KV for generation
            use_cache: Whether to return new KV cache
            return_hidden_states: Whether to return all hidden states
            return_logits: Whether to compute logits (False for distillation hidden states)

        Returns:
            Dictionary with logits, hidden_states, past_key_values as requested
        """
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create position IDs if not provided
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, past_length + seq_length,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        causal_mask = create_causal_mask(
            seq_length,
            past_key_values_length=past_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Combine with attention mask (for padding)
        if attention_mask is not None:
            # Expand attention mask: [batch, seq] -> [batch, 1, 1, seq]
            expanded_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + expanded_mask

        # Storage for outputs
        all_hidden_states = [hidden_states] if return_hidden_states else None
        new_past_key_values = [] if use_cache else None

        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing for memory efficiency
                hidden_states, present_kv = checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            if use_cache:
                new_past_key_values.append(present_kv)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        # Compute logits
        logits = None
        if return_logits:
            if self.lm_head is not None:
                logits = self.lm_head(hidden_states)
            else:
                # Tied embeddings
                logits = F.linear(hidden_states, self.embed_tokens.weight)

        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "past_key_values": new_past_key_values,
            "last_hidden_state": hidden_states,
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            input_ids: Input token IDs
            labels: Target token IDs (shifted internally)
            attention_mask: Padding mask

        Returns:
            (loss, logits) tuple
        """
        outputs = self.forward(input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs["logits"]

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for loss computation
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,  # Ignore padding
        )

        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs [batch_size, seq_length]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Whether to sample (False = greedy)
            eos_token_id: Stop generation at this token
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs [batch_size, seq_length + new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            # Forward pass (only new tokens if we have cache)
            if past_key_values is not None:
                curr_input_ids = generated[:, -1:]
            else:
                curr_input_ids = generated

            outputs = self.forward(
                curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]  # Last token logits
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return generated

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for faster training."""
        self.gradient_checkpointing = False


class ReasoningTransformerForCausalLM(nn.Module):
    """
    Wrapper providing HuggingFace-compatible interface.

    This makes the model compatible with HuggingFace Trainer
    and other ecosystem tools.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model = ReasoningTransformer(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """HuggingFace-compatible forward pass."""
        if labels is not None:
            loss, logits = self.model.compute_loss(
                input_ids, labels, attention_mask=attention_mask, **kwargs
            )
            return {"loss": loss, "logits": logits}
        else:
            outputs = self.model.forward(
                input_ids, attention_mask=attention_mask, **kwargs
            )
            return {"logits": outputs["logits"]}

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def num_parameters(self, *args, **kwargs):
        return self.model.num_parameters(*args, **kwargs)


def create_model(config_or_size: Union[TransformerConfig, str]) -> ReasoningTransformer:
    """
    Create a model from config or size string.

    Args:
        config_or_size: Either a TransformerConfig or size string ("500m", "7b", etc.)

    Returns:
        Initialized ReasoningTransformer
    """
    from .config import get_config

    if isinstance(config_or_size, str):
        config = get_config(config_or_size)
    else:
        config = config_or_size

    return ReasoningTransformer(config)
