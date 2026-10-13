import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout_proba: float,
        k_dim: int | None = None,
        v_dim: int | None = None,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.k_dim = k_dim if k_dim is not None else model_dim
        self.v_dim: float = v_dim if v_dim is not None else model_dim

        self.q_projection = nnx.Linear(
            model_dim, model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_projection = nnx.Linear(
            self.k_dim, model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.v_projection = nnx.Linear(
            self.v_dim, model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.out_projection = nnx.Linear(
            model_dim, model_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_proba, rngs=rngs)

    def split_heads(self, array: Array, query_length: int) -> Array:
        return array.reshape(-1, query_length, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        x: Array,
        key_padding_mask: Array | None = None,
        attention_mask: Array | None = None,
        cache: tuple[Array, Array] | None = None,
        is_causal: bool = True,
        train: bool = True,
    ) -> tuple[Array, tuple[Array, Array] | None]:
        query_length = x.shape[1]

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        # (B, T, C) -> (B, H, T, D)
        q = self.split_heads(q, query_length)
        k = self.split_heads(k, query_length)
        v = self.split_heads(v, query_length)

        past_length = 0
        if cache is not None:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)

        cache = (k, v)

        # (B, H, T, D)^T -> (B, H, D, T)
        k_transpose = k.astype(jnp.float32).swapaxes(-1, -2)
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attention_scores = q.astype(jnp.float32) @ k_transpose
        attention_scores = attention_scores * (self.head_dim**-0.5)

        total_key_length = k.shape[2]

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.reshape(x.shape[0], 1, 1, total_key_length)
            attention_scores = jnp.where(padding_mask, -jnp.inf, attention_scores)

        if attention_mask is not None:
            attention_scores = jnp.where(attention_mask, -jnp.inf, attention_scores)

        # upper triangular mask with inf in the upper triangle and zero everywhere else
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

        attention_score_weights = nnx.softmax(attention_scores, axis=-1)

        # (B, H, T, T) @ ((B, H, T, D) -> (B, H, T, D)
        context = attention_score_weights @ v.astype(jnp.float32)
        context = context.astype(q.dtype)

        # (B, H, T, D) -> (B, T, C)
        context = context.transpose(0, 2, 1, 3).reshape(
            -1, query_length, self.model_dim
        )
        output = self.out_projection(context)
        output = self.dropout(output, deterministic=not train)

        return output, cache
