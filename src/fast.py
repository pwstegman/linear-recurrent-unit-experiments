# %%
import os
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import datasets
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import optax
import treescope
from jaxtyping import Array, Float, Int, jaxtyped
from Levenshtein import distance
from tqdm.auto import trange
from typeguard import typechecked as typechecker
import aim

from asr import ASR
import tiktoken


try:
    treescope.basic_interactive_setup()
except AttributeError:
    print("Likely not running as iPython; Treescope is disabled.")

# jax.config.update("jax_explain_cache_misses", True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)


def get_next_batch(dataloader: jdl.DataLoader, tokenizer: tiktoken.Encoding) -> tuple:
    while True:
        batch = next(iter(dataloader))

        # Pad the signals to produce an input X of shape (batch_size, max_num_timesteps,
        # num_channels).
        # Note that num_channels will be 1.
        max_length = max(waveform["array"].shape[0] for waveform in batch["audio"])

        # Skip anything longer than 60 seconds.
        if max_length <= 16_000 * 60:
            break
    # Group into 15 second blocks for compilation efficiency.
    block_size = 16_000 * 15
    # Make sure it's divisible by 4096 (the frame size)
    block_size += (4096 - (block_size % 4096)) % 4096
    block_padding_amount = (block_size - (max_length % block_size)) % block_size
    batched_waveforms = jnp.expand_dims(
        a=jnp.stack(
            arrays=[
                jnp.pad(
                    array=waveform["array"],
                    pad_width=(
                        (
                            0,
                            max_length
                            - waveform["array"].shape[0]
                            + block_padding_amount,
                        )
                    ),
                )
                for waveform in batch["audio"]
            ]
        ),
        axis=2,
    )

    # Pad the transcripts to produce an expected output Y of shape (batch_size,
    # max_num_bytes, num_channels).
    texts = [
        tokenizer.encode(text, disallowed_special=[])
        for text in batch["normalized_text"]
    ]

    padding_mask = []

    max_text_length = max([len(text) for text in texts])
    text_block_size = 128
    text_block_padding_amount = (
        text_block_size - (max_text_length % text_block_size)
    ) % text_block_size
    for text in texts:
        padding_amount = max_text_length - len(text) + text_block_padding_amount
        padding_mask.append([0.0] * len(text) + [1.0] * padding_amount)
        text.extend([tokenizer.max_token_value + 1] * padding_amount)

    batched_transcripts = jnp.array(texts)
    batched_transcript_paddings = jnp.array(padding_mask)

    # Make dummy values for performance profiling.
    # batch_size = 16
    # batched_waveforms = jnp.ones(shape=(batch_size, 16_000 * 120, 1))
    # batched_transcripts = jnp.ones(shape=(batch_size, 512), dtype=jnp.int32)
    # batched_transcript_paddings = jnp.zeros(shape=(batch_size, 512))

    return batched_waveforms, batched_transcripts, batched_transcript_paddings


@jaxtyped(typechecker=typechecker)
def loss_fn(
    model: ASR,
    batched_waveforms: Float[Array, "batch_size num_timesteps 1"],
    expected_bytes: Int[Array, "batch_size num_bytes"],
    expected_bytes_paddings: Float[Array, "batch_size num_bytes"],
) -> tuple[Float[Array, ""], Array]:
    # Run the model to get byte estimates.
    batched_logits = model(batched_waveforms)

    # Compute loss.
    loss = optax.losses.ctc_loss(
        logits=batched_logits,
        logit_paddings=jax.numpy.zeros(batched_logits.shape[:2]),
        labels=expected_bytes,
        label_paddings=expected_bytes_paddings,
        blank_id=tokenizer.max_token_value + 1,
    ).mean()

    return loss, batched_logits


@jaxtyped(typechecker=typechecker)
def train_step(
    model: ASR,
    optimizer: nnx.Optimizer,
    batched_waveforms: Float[Array, "batch_size num_timesteps 1"],
    expected_bytes: Int[Array, "batch_size num_bytes"],
    expected_bytes_paddings: Float[Array, "batch_size num_bytes"],
) -> tuple:
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(
        model, batched_waveforms, expected_bytes, expected_bytes_paddings
    )
    optimizer.update(grads=grads)
    return loss, logits


# You can pre-download the dataset by running the following in bash:
#   poetry run python -c "import datasets; datasets.load_dataset(path='facebook/voxpopuli', name='en', num_proc=8);"
voxpopuli = datasets.load_dataset(
    path="facebook/voxpopuli",
    cache_dir="/mnt/workspace/src/data/",
    name="en",
    split="train",
    num_proc=8,
)

jdl.manual_seed(seed=0)
dataloader = jdl.DataLoader(
    dataset=voxpopuli,
    backend="jax",
    batch_size=1,
    shuffle=True,
    drop_last=False,
)

# %%
tokenizer = tiktoken.get_encoding("o200k_base")

model = ASR(
    rngs=nnx.Rngs(default=0),
    audio_channels=1,
    predicted_classes=tokenizer.n_vocab + 1,  # "+ 1" for a final "silence" token.
    unroll=8,
)
# optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.0001))
optimizer: nnx.Optimizer = nnx.Optimizer(
    model=model,
    tx=optax.chain(
        optax.clip_by_global_norm(max_norm=1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            step_size_fn=optax.schedules.warmup_constant_schedule(
                init_value=0.000001, peak_value=0.0001, warmup_steps=10_000
            )
        ),
        optax.scale(step_size=-1.0),
    ),
)


def gradient_report(state: nnx.State):
    means = []
    max_grad_abs = 0
    sizes = []
    for item in state.values():
        if isinstance(item, nnx.State):
            mean, max_grad_abs_c, size = gradient_report(item)
            max_grad_abs = max(max_grad_abs, max_grad_abs_c)
        else:
            item: nnx.VariableState
            mean = jnp.abs(item.value).mean()
            max_grad_abs = max(max_grad_abs, float(jnp.abs(item.value).max()))
            size = item.value.size

        means.append(mean)
        sizes.append(size)
    total_size = sum(sizes)
    weights = [size / total_size for size in sizes]
    mean = jnp.mean(jnp.array(means) * jnp.array(weights))

    return mean, max_grad_abs, total_size


train_step_jit = nnx.jit(train_step)

# %%
run = aim.Run()

for step in trange(1, 100_000):
    batched_waveforms, batched_transcripts, batched_transcript_paddings = (
        get_next_batch(dataloader=dataloader, tokenizer=tokenizer)
    )

    train_loss, logits = train_step_jit(
        model=model,
        optimizer=optimizer,
        batched_waveforms=batched_waveforms,
        expected_bytes=batched_transcripts,
        expected_bytes_paddings=batched_transcript_paddings,
    )

    jit_cache_size = train_step_jit.inner._cache_size()

    run.track(train_loss, name="loss", step=step, context={"subset": "train"})
    run.track(
        jit_cache_size,
        name="jit_cache_size",
        step=step,
        context={"subset": "performance"},
    )
