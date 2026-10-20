import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from toytrain.models.layers import RMSNorm


class Embedding(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.model_dtype = dtype
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=model_dim,
            dtype=self.model_dtype,
            rngs=rngs,
        )
        self.norm = RMSNorm(model_dim, dtype=self.model_dtype)

    def __call__(self, x: Array, out: bool = False) -> Array:
        return self.embedding.attend(self.norm(x)) if out else self.embedding(x)


class FeedForwardNetwork(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        dropout_proba: float,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.up_projection = nnx.Linear(
            model_dim, feed_forward_dim, dtype=dtype, rngs=rngs
        )
        self.down_projection = nnx.Linear(
            feed_forward_dim, model_dim, dtype=dtype, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout_proba, rngs=rngs)

    def __call__(self, x: Array, train: bool = True) -> Array:
        x = self.up_projection(x)
        x = nnx.gelu(x)
        x = self.down_projection(x)
        x = self.dropout(x, deterministic=not train)
        return x
