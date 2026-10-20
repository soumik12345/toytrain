import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float

from toytrain.models.attention.utils import safely_compute_attention_weights
from toytrain.models.layers import RMSNorm, RoPE

MHLA_KV_CACHE_TYPE = tuple[
    Float[Array, "B P {self.kv_lora_rank}"],
    Float[Array, "B P {self.rope_head_dim}"] | None,
]

MHLA_OUT_CACHE_TYPE = tuple[
    Float[Array, "B total_keys {self.kv_lora_rank}"],
    Float[Array, "B total_keys {self.rope_head_dim}"] | None,
]

MHLA_CALL_RETURN_TYPE = tuple[Float[Array, "B T {self.model_dim}"], MHLA_OUT_CACHE_TYPE]


class MultiHeadLatentAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        attention_weight_dropout_proba: float,
        residual_dropout_proba: float,
        use_qk_norm: bool = False,
        use_rmsnorm_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        *,
        rope_module: RoPE | None = None,
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
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.k_dim = model_dim
        self.v_dim = model_dim
        self.use_qk_norm = use_qk_norm

        self.rope_module = rope_module
        if rope_module is not None:
            self.rope_head_dim = rope_module.model_dim
            self.qk_head_dim = self.head_dim + self.rope_head_dim
            if self.rope_head_dim <= 0 or self.rope_head_dim % 2 != 0:
                raise ValueError("RoPE head dimension must be positive and even.")

        self.q_down_projection = nnx.Linear(
            self.model_dim, self.q_lora_rank, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.q_up_projection = nnx.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim
            if self.rope_module is not None
            else self.model_dim,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.kv_down_projection = nnx.Linear(
            self.model_dim, self.kv_lora_rank, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_up_projection = nnx.Linear(
            self.kv_lora_rank, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        if self.rope_module is not None:
            self.k_rope_projection = nnx.Linear(
                self.model_dim,
                self.rope_head_dim,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )

        self.v_up_projection = nnx.Linear(
            self.kv_lora_rank, self.model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.q_lora_rank, use_rmsnorm_bias)
            self.kv_norm = RMSNorm(self.kv_lora_rank, use_rmsnorm_bias)

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
        past_length = 0 if cache is None else cache[0].shape[1]

        # B -> Batch size
        # T -> Query length
        # P -> Cached sequence length (past_length)
        # K -> Total key/value length, i.e., P + T
        # H -> Num query heads
        # D -> Content/value head dimension, i.e., model_dim // num_heads
        # D_R -> RoPE head dimension
        # D_QK -> Q/K head dimension, it's D + D_R when RoPE is enabled, D otherwise
        # C -> model_dim
        # R_Q -> Q LoRA rank
        # R_KV -> KV LoRA rank

        # (B, T, C) -> (B, T, R_Q)
        compressed_q = self.q_down_projection(x)
        if self.use_qk_norm:
            compressed_q = self.q_norm(compressed_q)
        # (B, T, R_Q) -> (B, T, H * D_QK)
        q = self.q_up_projection(compressed_q)
        # (B, T, H * D_QK) -> (B, T, H, D_QK)
        q = (
            q.reshape(-1, query_length, self.num_heads, self.qk_head_dim)
            if self.rope_module is not None
            else q.reshape(-1, query_length, self.num_heads, self.head_dim)
        )
        # (B, T, H, D_QK) -> (B, H, T, D_QK)
        q = q.transpose(0, 2, 1, 3)

        if self.rope_module is not None:
            # (B, H, T, D_QK) -> (B, H, T, D), (B, H, T, D_R)
            q_content, q_position = jnp.split(q, [self.head_dim], axis=-1)
            # RoPE preserves (B, H, T, D_R).
            q_position = self.rope_module(q_position, past_length)
        else:
            q_content = q

        # (B, T, C) -> (B, T, R_KV)
        compressed_kv = self.kv_down_projection(x)
        if self.use_qk_norm:
            compressed_kv = self.kv_norm(compressed_kv)

        if self.rope_module is not None:
            # (B, T, C) -> (B, T, D_R); positional keys are shared across heads.
            k_position = self.k_rope_projection(x)
            # RoPE preserves (B, T, D_R).
            k_position = self.rope_module(k_position, past_length)
        else:
            k_position = None

        if cache is not None:
            past_kv, past_k_position = cache
            # (B, P, R_KV) + (B, T, R_KV) -> (B, K, R_KV), concatenating time.
            compressed_kv = jnp.concatenate([past_kv, compressed_kv], axis=1)
            if self.rope_module is not None:
                assert k_position is not None and past_k_position is not None
                # (B, P, D_R) + (B, T, D_R) -> (B, K, D_R), concatenating time.
                k_position = jnp.concatenate([past_k_position, k_position], axis=1)

        # Cache shapes: (B, K, R_KV), (B, K, D_R); the second entry is None without RoPE.
        new_cache = (compressed_kv, k_position)
        total_key_length = compressed_kv.shape[1]

        # (B, K, R_KV) -> (B, K, C) -> (B, K, H, D) -> (B, H, K, D)
        k_content = (
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
        k_content_transpose = k_content.astype(jnp.float32).swapaxes(-1, -2)
        # (B, H, T, D) @ (B, H, D, K) -> (B, H, T, K)
        attention_scores = q_content.astype(jnp.float32) @ k_content_transpose
        if self.rope_module is not None:
            assert k_position is not None
            # (B, K, D_R) -> (B, 1, K, D_R) -> (B, 1, D_R, K)
            k_position_transpose = k_position.astype(jnp.float32)[
                :, None, :, :
            ].swapaxes(-1, -2)
            # (B, H, T, D_R) @ (B, 1, D_R, K) -> (B, H, T, K), broadcasting heads.
            position_attention_scores = (
                q_position.astype(jnp.float32) @ k_position_transpose
            )
            attention_scores = attention_scores + position_attention_scores
        attention_scores = (
            attention_scores * (self.head_dim**-0.5)
            if self.rope_module is None
            else attention_scores * (self.qk_head_dim**-0.5)
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
