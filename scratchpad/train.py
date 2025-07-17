from time import time

import flax.nnx as nnx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, jaxtyped
from typeguard import typechecked as typechecker

from asr import ASR


@jaxtyped(typechecker=typechecker)
def loss_fn(
    model: ASR,
    signal: Float[Array, "1 num_timesteps"],
    expected_bytes: Int[Array, "num_timesteps//128"],
) -> tuple[Float[Array, ""], Float[Array, "256 num_timesteps//128"]]:
    # Create masks for separate loss calculations on silence (null bytes in transcript)
    # and non silence (non-null bytes in transcript).
    mask_non_zero = jnp.array(expected_bytes != 0, dtype=int)
    mask_zeros = 1 - mask_non_zero

    # Run the model to get byte estimates.
    logits = model(signal)

    # Compute loss.
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.T, labels=expected_bytes
    )

    # Compute separate aggregated losses for silence and speaking.
    loss_zeros = (loss * mask_zeros).sum() / mask_zeros.sum()
    loss_non_zeros = (loss * mask_non_zero).sum() / mask_non_zero.sum()

    # Weigh silence and speaking equally.
    loss = loss_non_zeros + loss_zeros

    return loss, logits


@jaxtyped(typechecker=typechecker)
def train_step(
    model: ASR,
    optimizer: nnx.Optimizer,
    signal: Float[Array, "1 num_timesteps"],
    expected_bytes: Int[Array, "num_timesteps//128"],
) -> Array:
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(model, signal, expected_bytes)
    optimizer.update(grads=grads)
    return loss


# Build our ASR model.
model = ASR(rngs=nnx.Rngs(0), unroll=32)

# Load training data.
data_path = "data/LibriSpeech/train-clean-100/19/198"

# Train the model.
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.001))
train_step_jit = nnx.jit(train_step)

for step in range(100):
    start = time()
    loss = train_step_jit(
        model, optimizer, example_signal, expected_bytes
    ).block_until_ready()
    step_time_ms = (time() - start) * 1000
    print(f"{step=:06}: {loss=:#.5g} {step_time_ms=:#.5g}")
