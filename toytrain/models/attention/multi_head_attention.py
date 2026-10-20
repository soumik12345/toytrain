import math

import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Bool, Float, jaxtyped

from toytrain.models.attention.utils import (
    apply_l2_norm,
    safely_compute_attention_weights,
)

MHA_KV_CACHE_TYPE = tuple[
    Float[Array, "B {self.num_heads} P {self.head_dim}"],
    Float[Array, "B {self.num_heads} P {self.head_dim}"],
]

MHA_OUT_CACHE_TYPE = tuple[
    Float[Array, "B {self.num_heads} total_keys {self.head_dim}"],
    Float[Array, "B {self.num_heads} total_keys {self.head_dim}"],
]

MHA_CALL_RETURN_TYPE = tuple[Float[Array, "B T {self.model_dim}"], MHA_OUT_CACHE_TYPE]


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        attention_weight_dropout_proba: float,
        residual_dropout_proba: float,
        k_dim: int | None = None,
        v_dim: int | None = None,
        qk_norm_init_length: float | None = None,
        is_cross_attention: bool = False,
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
        self.k_dim = k_dim if k_dim is not None else model_dim
        self.v_dim: int = v_dim if v_dim is not None else model_dim
        self.is_cross_attention = is_cross_attention
        self.qk_norm_init_length = qk_norm_init_length

        self.use_separate_projections = (
            is_cross_attention
            or (self.k_dim != self.model_dim)
            or (self.v_dim != self.model_dim)
        )

        if not self.use_separate_projections:
            self.qkv_projection = nnx.Linear(
                self.model_dim,
                3 * self.model_dim,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.q_projection = nnx.Linear(
                self.model_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
            )
            self.k_projection = nnx.Linear(
                self.k_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
            )
            self.v_projection = nnx.Linear(
                self.v_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
            )

        if self.qk_norm_init_length is not None:
            if (
                not math.isfinite(self.qk_norm_init_length)
                or self.qk_norm_init_length <= 1
            ):
                raise ValueError("qk_norm_init_length must be finite and > 1.")
            initial_scale = math.log2(self.qk_norm_init_length) + math.log2(
                self.qk_norm_init_length - 1
            )
            self.qk_scale = nnx.Param(jnp.asarray(initial_scale, dtype=jnp.float32))

        self.out_projection = nnx.Linear(
            self.model_dim, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.attention_weights_dropout = nnx.Dropout(
            attention_weight_dropout_proba, rngs=rngs
        )
        self.residual_dropout = nnx.Dropout(residual_dropout_proba, rngs=rngs)

    def split_projection_across_heads(self, array: Array) -> Array:
        sequence_length = array.shape[1]
        array = array.reshape(-1, sequence_length, self.num_heads, self.head_dim)
        return array.transpose(0, 2, 1, 3)

    def self_attention_prepare_qkv(self, x: Array, cache: MHA_KV_CACHE_TYPE | None):
        if not self.use_separate_projections:
            qkv = self.qkv_projection(x)
            q, k, v = jnp.split(qkv, 3, axis=-1)
        else:
            q = self.q_projection(x)
            k = self.k_projection(x)
            v = self.v_projection(x)

        # (B, T, C) -> (B, H, T, D)
        q = self.split_projection_across_heads(q)
        k = self.split_projection_across_heads(k)
        v = self.split_projection_across_heads(v)

        # Normalize new keys once; cached keys are already normalized.
        if self.qk_norm_init_length is not None:
            k = apply_l2_norm(k)

        past_length = 0
        if cache is not None:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        return q, k, v, past_length

    def cross_attention_prepare_qkv(
        self,
        x: Array,
        key: Float[Array, "B S {self.k_dim}"] | None,
        value: Float[Array, "B S {self.v_dim}"] | None,
        cache: MHA_KV_CACHE_TYPE | None,
    ):
        q = self.split_projection_across_heads(self.q_projection(x))

        if cache is not None:
            k, v = cache
        else:
            if key is None:
                raise ValueError("Cross-attention requires `key` when `cache` is None.")
            key_input = key
            value_input = value if value is not None else key_input
            k = self.split_projection_across_heads(self.k_projection(key_input))
            v = self.split_projection_across_heads(self.v_projection(value_input))
            if self.qk_norm_init_length is not None:
                k = apply_l2_norm(k)

        return q, k, v

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        x: Float[Array, "B T {self.model_dim}"],
        key: Float[Array, "B S {self.k_dim}"] | None = None,
        value: Float[Array, "B S {self.v_dim}"] | None = None,
        key_padding_mask: Bool[Array, "B total_keys"] | None = None,
        attention_mask: Bool[Array, "T total_keys"] | None = None,
        cache: MHA_KV_CACHE_TYPE | None = None,
        is_causal: bool | None = None,
        train: bool = True,
    ) -> MHA_CALL_RETURN_TYPE:
        query_length = x.shape[1]
        is_cross_attention_call = (
            self.is_cross_attention or key is not None or value is not None
        )

        # B -> Batch size
        # T -> Query length
        # H -> Num heads
        # C -> model_dim
        # D -> Per head model_dim, i.e, model_dim // num_head

        if is_causal is None:
            is_causal = not is_cross_attention_call

        if not is_cross_attention_call:
            q, k, v, past_length = self.self_attention_prepare_qkv(x, cache)
        else:
            if not self.use_separate_projections:
                raise ValueError(
                    "Cross-attention inputs (`key`/`value`) were passed, but module was "
                    "initialized with a fused qkv_projection. Initialize with "
                    "is_cross_attention=True or custom k_dim/v_dim."
                )
            if is_causal:
                raise ValueError("Causal mask is not applicable for cross attention")
            if value is not None and key is None:
                raise ValueError(
                    "Cannot pass `value` without `key` in cross-attention."
                )
            q, k, v = self.cross_attention_prepare_qkv(x, key, value, cache)
            past_length = 0

        if self.qk_norm_init_length is not None:
            q = apply_l2_norm(q)

        cache = (k, v)
        total_key_length = k.shape[2]

        # (B, H, total_keys, D)^T -> (B, H, D, total_keys)
        k_transpose = k.astype(jnp.float32).swapaxes(-1, -2)
        # (B, H, T, D) @ (B, H, D, total_keys) -> (B, H, T, total_keys)
        attention_scores = q.astype(jnp.float32) @ k_transpose
        attention_scores = (
            attention_scores * self.qk_scale[...]
            if self.qk_norm_init_length is not None
            else attention_scores * (self.head_dim**-0.5)
        )

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

        # (B, H, T, total_keys) @ (B, H, total_keys, D) -> (B, H, T, D)
        context = attention_score_weights @ v.astype(jnp.float32)
        context = context.astype(q.dtype)

        # (B, H, T, D) -> (B, T, C)
        context = context.transpose(0, 2, 1, 3).reshape(
            -1, query_length, self.model_dim
        )
        output = self.out_projection(context)
        output = self.residual_dropout(output, deterministic=not train)

        return output, cache
