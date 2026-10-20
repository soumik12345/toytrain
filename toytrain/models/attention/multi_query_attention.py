import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Bool, Float, jaxtyped

from toytrain.models.attention.utils import safely_compute_attention_weights

MQA_KV_CACHE_TYPE = tuple[
    Float[Array, "B 1 P {self.head_dim}"],
    Float[Array, "B 1 P {self.head_dim}"],
]

MQA_OUT_CACHE_TYPE = tuple[
    Float[Array, "B 1 total_keys {self.head_dim}"],
    Float[Array, "B 1 total_keys {self.head_dim}"],
]

MQA_CALL_RETURN_TYPE = tuple[Float[Array, "B T {self.model_dim}"], MQA_OUT_CACHE_TYPE]


class MultiQueryAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
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

        self.qkv_projection = nnx.Linear(
            self.model_dim,
            self.model_dim + 2 * self.head_dim,
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
        key_padding_mask: Bool[Array, "B total_keys"] | None = None,
        attention_mask: Bool[Array, "T total_keys"] | None = None,
        cache: MQA_KV_CACHE_TYPE | None = None,
        is_causal: bool = True,
        train: bool = True,
    ) -> MQA_CALL_RETURN_TYPE:
        query_length = x.shape[1]

        # B -> Batch size
        # T -> Query length
        # K -> Total key/value length, including cached tokens
        # H -> Number of query heads (num_heads)
        # C -> model_dim
        # D -> Dimension per head (model_dim // num_heads)

        # (B, T, C + 2D)
        qkv = self.qkv_projection(x)

        # q -> (B, T, C)
        # k -> (B, T, D)
        # v -> (B, T, D)
        q, k, v = jnp.split(
            qkv, (self.model_dim, self.model_dim + self.head_dim), axis=-1
        )

        # (B, T, C) -> (B, H, T, D)
        q = q.reshape(-1, query_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        # (B, T, D) -> (B, 1, T, D)
        k = jnp.expand_dims(k, axis=1)
        v = jnp.expand_dims(v, axis=1)

        past_length = 0
        if cache is not None:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        cache = (k, v)
        total_key_length = k.shape[2]

        # (B, 1, K, D) -> (B, 1, D, K)
        k_transpose = k.astype(jnp.float32).swapaxes(-1, -2)
        # (B, H, T, D) @ (B, 1, D, K) -> (B, H, T, K)
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

        # Attention weights -> (B, H, T, K); softmax is over keys.
        attention_score_weights = safely_compute_attention_weights(attention_scores)
        attention_score_weights = self.attention_weights_dropout(
            attention_score_weights, deterministic=not train
        )

        # (B, H, T, K) @ (B, 1, K, D) -> (B, H, T, D)
        context = attention_score_weights @ v.astype(jnp.float32)
        context = context.astype(q.dtype)

        # (B, H, T, D) -> (B, T, model_dim)
        context = context.transpose(0, 2, 1, 3).reshape(
            -1, query_length, self.model_dim
        )
        output = self.out_projection(context)
        output = self.residual_dropout(output, deterministic=not train)

        return output, cache
