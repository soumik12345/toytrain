import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Bool, Float, jaxtyped

from toytrain.models.layers import RoPE
from toytrain.models.attention.utils import safely_compute_attention_weights

GQA_KV_CACHE_TYPE = tuple[
    Float[Array, "B {self.num_groups} P {self.head_dim}"],
    Float[Array, "B {self.num_groups} P {self.head_dim}"],
]

GQA_OUT_CACHE_TYPE = tuple[
    Float[Array, "B {self.num_groups} total_keys {self.head_dim}"],
    Float[Array, "B {self.num_groups} total_keys {self.head_dim}"],
]

GQA_CALL_RETURN_TYPE = tuple[Float[Array, "B T {self.model_dim}"], GQA_OUT_CACHE_TYPE]


class GroupedQueryAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_groups: int,
        attention_weight_dropout_proba: float,
        residual_dropout_proba: float,
        dtype: jnp.dtype = jnp.float32,
        *,
        rope_module: RoPE | nnx.Module | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive.")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        if num_heads % num_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups})."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = model_dim // num_heads
        self.heads_per_dim = num_heads // num_groups
        self.kv_dim = num_groups * self.head_dim
        self.rope_module = rope_module

        self.qkv_projection = nnx.Linear(
            self.model_dim,
            self.model_dim + 2 * self.kv_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.out_projection = nnx.Linear(
            self.model_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.attention_weights_dropout = nnx.Dropout(
            attention_weight_dropout_proba, rngs=rngs
        )
        self.residual_dropout = nnx.Dropout(residual_dropout_proba, rngs=rngs)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, "B T {self.model_dim}"],
        cache: GQA_KV_CACHE_TYPE | None = None,
        key_padding_mask: Bool[Array, "B total_keys"] | None = None,
        attention_mask: Bool[Array, "T total_keys"] | None = None,
        is_causal: bool = True,
        train: bool = True,
    ) -> GQA_CALL_RETURN_TYPE:
        query_length = x.shape[1]

        # B -> Batch size
        # T -> Query length
        # H -> Num query heads
        # G -> Number of key/value heads
        # C -> model_dim
        # D -> Per head model_dim, i.e, model_dim // num_heads

        # qkv -> (B, T, C + 2 * G * D)
        qkv = self.qkv_projection(x)

        # q -> (B, T, C)
        # k -> (B, T, G * D)
        # v -> (B, T, G * D)
        q, k, v = jnp.split(
            qkv, (self.model_dim, self.model_dim + self.kv_dim), axis=-1
        )

        # (B, T, C) -> (B, H, T, D)
        q = q.reshape(-1, query_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        past_length = 0 if cache is None else cache[0].shape[2]

        # (B, T, G * D) -> (B, T, G, D) -> (B, G, T, D)
        k = k.reshape(-1, query_length, self.num_groups, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # (B, T, G * D) -> (B, T, G, D) -> (B, G, T, D)
        v = v.reshape(-1, query_length, self.num_groups, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.rope_module is not None:
            q = self.rope_module(q, position_offset=past_length)
            k = self.rope_module(k, position_offset=past_length)

        if cache is not None:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        cache = (k, v)
        total_key_length = k.shape[2]

        # K -> total key/value length = past_length + query_length

        # (B, G, K, D) -> (B, H, K, D)
        k = k.repeat(self.heads_per_dim, axis=1)
        v = v.repeat(self.heads_per_dim, axis=1)

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

        return output, cache
