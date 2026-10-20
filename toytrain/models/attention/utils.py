import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


def safely_compute_attention_weights(attention_scores: Array):
    fully_masked = jnp.all(jnp.isneginf(attention_scores), axis=-1, keepdims=True)
    safe_scores = jnp.where(fully_masked, 0.0, attention_scores)
    attention_score_weights = nnx.softmax(safe_scores, axis=-1)
    return jnp.where(fully_masked, 0.0, attention_score_weights)


def apply_l2_norm(x: Array, epsilon: float = 1e-6) -> Array:
    x = x.astype(jnp.float32)
    squared_norm = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    return x / jnp.sqrt(jnp.maximum(squared_norm, epsilon**2))
