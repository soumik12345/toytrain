import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float

from toytrain.models.attention.utils import safely_compute_attention_weights

MHLA_KV_CACHE_TYPE = Float[Array, "B P {self.kv_lora_rank}"]

MHLA_OUT_CACHE_TYPE = Float[Array, "B total_keys {self.kv_lora_rank}"]

MHLA_CALL_RETURN_TYPE = tuple[Float[Array, "B T {self.model_dim}"], MHLA_OUT_CACHE_TYPE]


class MultiHeadLatentAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        kv_lora_rank: int,
        attention_weight_dropout_proba: float,
        residual_dropout_proba: float,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive.")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.kv_lora_rank = kv_lora_rank
        self.k_dim = model_dim
        self.v_dim = model_dim

        self.q_projection = nnx.Linear(
            self.model_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.kv_down_projection = nnx.Linear(
            self.model_dim, self.kv_lora_rank, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_up_projection = nnx.Linear(
            self.kv_lora_rank, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.v_up_projection = nnx.Linear(
            self.kv_lora_rank, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.out_projection = nnx.Linear(
            self.model_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.attention_weights_dropout = nnx.Dropout(
            attention_weight_dropout_proba, rngs=rngs
        )
        self.residual_dropout = nnx.Dropout(residual_dropout_proba, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B T {self.model_dim}"],
        cache: MHLA_KV_CACHE_TYPE | None = None,
        key_padding_mask: Bool[Array, "B total_keys"] | None = None,
        attention_mask: Bool[Array, "T total_keys"] | None = None,
        is_causal: bool = True,
        train: bool = True,
    ) -> MHLA_CALL_RETURN_TYPE:
        query_length = x.shape[1]

        # B -> Batch size
        # T -> Query length
        # H -> Num query heads
        # D -> Per head model_dim, i.e, model_dim // num_heads
        # C -> model_dim
        # R -> KV LoRA rank

        # (B, T, C) -> (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        q = (
            self.q_projection(x)
            .reshape(-1, query_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # (B, T, C) -> (B, T, R)
        compressed_kv = self.kv_down_projection(x)

        past_length = 0
        if cache is not None:
            past_length = cache.shape[1]
            compressed_kv = jnp.concatenate([cache, compressed_kv], axis=1)

        new_cache = compressed_kv
        total_key_length = compressed_kv.shape[1]

        # K -> total key/value length = past_length + query_length

        # (B, K, R) -> (B, K, C) -> (B, K, H, D) -> (B, H, K, D)
        k = (
            self.k_up_projection(compressed_kv)
            .reshape(-1, total_key_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_up_projection(compressed_kv)
            .reshape(-1, total_key_length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # (B, H, K, D) -> (B, H, D, K)
        k_transpose = k.astype(jnp.float32).swapaxes(-1, -2)
        # (B, H, T, D) @ (B, H, D, K) -> (B, H, T, K)
        attention_scores = q.astype(jnp.float32) @ k_transpose
        attention_scores = attention_scores * (self.head_dim**-0.5)

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.reshape(x.shape[0], 1, 1, total_key_length)
            attention_scores = jnp.where(padding_mask, -jnp.inf, attention_scores)

        if attention_mask is not None:
            attention_scores = jnp.where(attention_mask, -jnp.inf, attention_scores)

        if is_causal:
            causal_mask = jnp.triu(
                jnp.full(
                    (query_length, total_key_length),
                    -jnp.inf,
                    dtype=attention_scores.dtype,
                ),
                k=past_length + 1,
            )
            attention_scores = attention_scores + causal_mask

        attention_score_weights = safely_compute_attention_weights(attention_scores)
        attention_score_weights = self.attention_weights_dropout(
            attention_score_weights, deterministic=not train
        )

        # (B, H, T, K) @ (B, H, K, D) -> (B, H, T, D)
        context = attention_score_weights @ v.astype(jnp.float32)
        context = context.astype(q.dtype)

        # (B, H, T, D) -> (B, T, model_dim)
        context = context.transpose(0, 2, 1, 3).reshape(
            -1, query_length, self.model_dim
        )
        output = self.out_projection(context)
        output = self.residual_dropout(output, deterministic=not train)

        return output, new_cache
