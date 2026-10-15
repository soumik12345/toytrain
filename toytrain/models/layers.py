import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class RMSNorm(nnx.Module):
    def __init__(
        self, model_dim: int, epsilon: float = 1e-6, dtype: jnp.dtype = jnp.float32
    ) -> None:
        self.epsilon = epsilon
        self.model_dtype = dtype
        self.gamma = nnx.Param(jnp.ones((1, 1, model_dim), dtype=self.model_dtype))
        self.beta = nnx.Param(jnp.zeros((1, 1, model_dim), dtype=self.model_dtype))

    def __call__(self, x: Array) -> Array:
        x_square = jnp.square(x)
        mean_x_square = jnp.mean(x_square, axis=-1, keepdims=True)
        root_mean_x_square = jnp.sqrt(mean_x_square + self.epsilon)
        norm_x = x / root_mean_x_square
        return norm_x * self.gamma + self.beta


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


class RoPE(nnx.Module):
    def __init__(
        self, sequence_len: int, model_dim: int, theta_base: float = 1e4
    ) -> None:
        frequencies = jnp.arange(sequence_len, dtype=jnp.float32)[:, None] + 1
        positions = jnp.arange(model_dim // 2, dtype=jnp.float32)[:, None]
        positions = positions.repeat(2, axis=-1).reshape(1, -1)
        theta = jnp.exp(-2 * positions / model_dim * jnp.log(theta_base))
        self.cos = jnp.cos(frequencies * theta)
        self.sin = jnp.sin(frequencies * theta)
        self.model_dim = model_dim

    def __call__(self, x: Array, position_offset: int) -> Array:
        original_dtype = x.dtype
        x = x.astype(jnp.float32)

        current_sequence_length = x.shape[1]

        rope_cosine_part = (
            x * self.cos[position_offset : position_offset + current_sequence_length, :]
        )

        x_inter = x.reshape((-1, current_sequence_length, self.model_dim // 2, 2))
        x_inter_one = x_inter[..., 0]
        x_inter_two = -1 * x_inter[..., 1]
        x_inter = jnp.stack([x_inter_two, x_inter_one], axis=-1)
        x_inter = x_inter.reshape((-1, current_sequence_length, self.model_dim))

        rope_sine_part = (
            x_inter
            * self.sin[position_offset : position_offset + current_sequence_length, :]
        )

        x = rope_cosine_part + rope_sine_part
        x = x.astype(original_dtype)
        return x
