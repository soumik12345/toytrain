import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Float, jaxtyped


class RMSNorm(nnx.Module):
    def __init__(
        self,
        model_dim: int,
        use_bias: bool = True,
        epsilon: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.model_dtype = dtype
        self.gamma = nnx.Param(jnp.ones((1, 1, model_dim), dtype=self.model_dtype))
        if self.use_bias:
            self.beta = nnx.Param(jnp.zeros((1, 1, model_dim), dtype=self.model_dtype))

    def __call__(self, x: Array) -> Array:
        x = x.astype(jnp.float32)
        x_square = jnp.square(x)
        mean_x_square = jnp.mean(x_square, axis=-1, keepdims=True)
        root_mean_x_square = jnp.sqrt(mean_x_square + self.epsilon)
        norm_x = x / root_mean_x_square
        norm_x = norm_x * self.gamma
        return norm_x + self.beta if self.use_bias else norm_x


class RoPE(nnx.Module):
    def __init__(
        self, sequence_len: int, model_dim: int, theta_base: float = 1e4
    ) -> None:
        # (sequence_len, 1) -> [[1], [2], ... [sequence_len]]
        frequencies = jnp.arange(sequence_len, dtype=jnp.float32)[:, None] + 1

        # (model_dim // 2, 1) -> [[0], [1], ..., [model_dim // 2 - 1]]
        positions = jnp.arange(model_dim // 2, dtype=jnp.float32)[:, None]

        # (model_dim // 2, 1) -> (model_dim // 2, 2) -> (1, model_dim)
        positions = positions.repeat(2, axis=-1).reshape(1, -1)

        # (1, model_dim)
        theta = jnp.exp(-2 * positions / model_dim * jnp.log(theta_base))

        # (sequence_len, model_dim)
        self.cos = jnp.cos(frequencies * theta)
        self.sin = jnp.sin(frequencies * theta)

        self.model_dim = model_dim

    def validate(self, position_offset: int, position_end: int):
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative.")
        if position_end > self.cos.shape[0]:
            raise ValueError(
                f"RoPE position range [{position_offset}, {position_end}) exceeds "
                f"the precomputed sequence length ({self.cos.shape[0]}). "
                "Initialize RoPE with a larger sequence_len."
            )

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, x: Float[Array, "*leading T {self.model_dim}"], position_offset: int
    ) -> Float[Array, "*leading T {self.model_dim}"]:
        original_dtype = x.dtype
        x = x.astype(jnp.float32)

        current_sequence_length = x.shape[-2]
        position_end = position_offset + current_sequence_length
        self.validate(position_offset, position_end)

        # (sequence_len, model_dim) -> (T, model_dim)
        cosine_part = self.cos[position_offset:position_end]
        sine_part = self.sin[position_offset:position_end]

        # (..., T, model_dim // 2, 2)
        pairs = x.reshape((*x.shape[:-1], self.model_dim // 2, 2))

        # stack((..., T, model_dim // 2), (..., T, model_dim // 2)) -> (..., T, model_dim // 2, 2) -> (..., T, model_dim)
        rotated = jnp.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)

        return (x * cosine_part + rotated * sine_part).astype(original_dtype)
