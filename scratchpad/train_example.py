# %%
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, jaxtyped
from typeguard import typechecked as typechecker
from Levenshtein import distance

from asr import ASR


@jaxtyped(typechecker=typechecker)
def loss_fn(
    model: ASR,
    signal: Float[Array, "1 num_timesteps"],
    expected_bytes: Array,
) -> tuple[Float[Array, ""], Array]:
    # Run the model to get byte estimates.
    logits = model(signal)

    # Add a batch axis of size 1.
    logits = jnp.expand_dims(logits, axis=0)
    expected_bytes = jnp.expand_dims(expected_bytes, axis=0)

    # Transpose to shape (batch, timesteps, channels).
    logits = logits.transpose(0, 2, 1)

    # Expected bytes should have shape (batch, num_bytes).
    # Pad it to the size of timesteps.
    padding = logits.shape[1] - expected_bytes.shape[1]
    label_paddings = jnp.concat(
        (jnp.zeros(expected_bytes.shape), jnp.ones((expected_bytes.shape[0], padding))),
        axis=1,
    )
    labels = jnp.concat(
        (expected_bytes, jnp.zeros((expected_bytes.shape[0], padding), dtype=int)),
        axis=1,
    )

    # Compute loss.
    loss = optax.losses.ctc_loss(
        logits=logits,
        logit_paddings=jax.numpy.zeros((logits.shape[0], logits.shape[1])),
        labels=labels,
        label_paddings=label_paddings,
        blank_id=0,
    ).mean()

    return loss, logits


@jaxtyped(typechecker=typechecker)
def train_step(
    model: ASR,
    optimizer: nnx.Optimizer,
    signal: Float[Array, "1 num_timesteps"],
    expected_bytes: Int[Array, "num_expected_bytes"],
) -> tuple:
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, signal, expected_bytes)
    optimizer.update(grads=grads)
    return loss, logits


# model = ASR(rngs=nnx.Rngs(0), unroll=32)
# example_signal = jax.random.uniform(nnx.Rngs(0xBEEF).signal(), (1, 560_000))

# expected_bytes_pytree = [0] * (example_signal.shape[1] // 128)
# transcript = "hello world"
# for index, char in enumerate(bytes(transcript, encoding="utf-8")):
#     expected_bytes_pytree[-len(transcript) + index] = char
# expected_bytes: Array = jnp.array(expected_bytes_pytree, dtype=int)

# optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.001))

# train_step_jit = nnx.jit(train_step)

# # %%
# from time import time

# for step in range(100):
#     start = time()
#     loss, logits = train_step_jit(model, optimizer, example_signal, expected_bytes)
#     loss.block_until_ready()
#     predicted_bytes = [x for x in jnp.argmax(logits, axis=2)[0].tolist() if x != 0]
#     actual_bytes = list(map(int, bytes("hello world", encoding="utf-8")))
#     character_error_rate = distance(predicted_bytes, actual_bytes) / len(actual_bytes)
#     step_time_ms = (time() - start) * 1000
#     print(f"{step=:06}: {loss=:#.5g} {character_error_rate=:#.3g} {step_time_ms=:#.5g}")
#     print(f"  {predicted_bytes}")
#     print(f"  {actual_bytes}")
# # %%
# import treescope

# treescope.basic_interactive_setup(autovisualize_arrays=True)

# jnp.argmax(model(example_signal), axis=0)

# # %%
# print(list(map(int, bytes("hello world", encoding="utf-8"))))

# %%
# PCM

import miniaudio

stream = miniaudio.flac_read_file_f32(
    "../data/LibriSpeech/train-clean-100/6880/216547/6880-216547-0000.flac"
)

signal = jnp.array(stream.samples)

# %%
with open(
    "../data/LibriSpeech/train-clean-100/6880/216547/6880-216547.trans.txt",
    encoding="utf-8",
) as file:
    transcript = file.readline()
    foo = bytes(transcript[len("6880-216547-0000 ") :], encoding="utf-8")
    foo = jnp.array([int(x) for x in foo], dtype=int)
    print(foo)

signal = jnp.expand_dims(signal, 0)

padding_needed = (128 - signal.shape[1] % 128) % 128
signal = jnp.concat((signal, jnp.zeros((1, padding_needed))), axis=1)

signal.shape, foo.shape

# %%
model = ASR(rngs=nnx.Rngs(0), unroll=32)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.00001))

train_step_jit = nnx.jit(train_step)

from time import time

# %%
start = time()
for step in range(100_000_000):
    loss, logits = train_step_jit(model, optimizer, signal, foo)
    if step % 250 == 0:
        predicted_bytes = [x for x in jnp.argmax(logits, axis=2)[0].tolist() if x != 0]
        character_error_rate = (
            distance(predicted_bytes, foo.tolist()) / foo.shape[0] * 100
        )
        step_time_ms = (time() - start) * 1000 / 100
        start = time()
        print(
            f"{step=:06}: {loss=:#.5g} {character_error_rate=:#.3g} {step_time_ms=:#.5g}"
        )
        print(
            f"  predicted = {bytes(predicted_bytes).decode("utf-8", errors="ignore")}"
        )
        print(f"  actual    = {bytes(foo.tolist()).decode("utf-8")}")
