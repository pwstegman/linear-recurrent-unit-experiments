# %%
import time
import traceback

import datasets
import datasets.config
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import treescope
from jaxtyping import Array, Float, Int, jaxtyped
from typeguard import typechecked as typechecker

import aim
import optimizer as opt
from asr import ASR

# Configure the notebook.
try:
    treescope.basic_interactive_setup()
except AttributeError:
    print("Likely not running as iPython; Treescope interactive setup is disabled.")

# jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)


# %%
# Define training helper methods.
def get_batch(
    dataset,
    start_index: int,
    batch_size: int,
) -> tuple[Array, Array, Array, int]:
    audio_len = 16_000 * 5
    text_len = 128

    waveforms = np.zeros(shape=(batch_size, audio_len, 1), dtype=np.float32)
    transcripts = np.zeros(
        shape=(batch_size, text_len),
        dtype=np.int32,
    )

    for batch_index in range(batch_size):
        entry_index = (start_index + batch_index) % len(dataset)

        audio = np.frombuffer(buffer=dataset[entry_index]["audio"], dtype=np.float32)
        text = np.frombuffer(buffer=dataset[entry_index]["tokens"], dtype=np.int32)

        waveforms[batch_index, :, 0] = audio
        transcripts[batch_index, :] = text

    transcripts_padding = (transcripts == 0).astype(np.float32)

    return (
        jax.numpy.array(waveforms),
        jax.numpy.array(transcripts),
        jax.numpy.array(transcripts_padding),
        start_index + batch_index,
    )


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
        # blank_id=tokenizer.max_token_value + 1,
        blank_id=0,
    ).mean()

    return loss, batched_logits


@jaxtyped(typechecker=typechecker)
def train_step(
    model: ASR,
    optimizer: opt.Optimizer,
    validation_loss,
    batched_waveforms: Float[Array, "batch_size num_timesteps 1"],
    expected_bytes: Int[Array, "batch_size num_bytes"],
    expected_bytes_paddings: Float[Array, "batch_size num_bytes"],
) -> tuple[Float[Array, ""], Array]:
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(
        model, batched_waveforms, expected_bytes, expected_bytes_paddings
    )
    optimizer.update(grads=grads, value=validation_loss)
    return loss, logits


@jaxtyped(typechecker=typechecker)
def eval_step(
    model: ASR,
    batched_waveforms: Float[Array, "batch_size num_timesteps 1"],
    expected_bytes: Int[Array, "batch_size num_bytes"],
    expected_bytes_paddings: Float[Array, "batch_size num_bytes"],
) -> tuple[Float[Array, ""], Array]:
    batched_logits = model(batched_waveforms)
    loss = optax.losses.ctc_loss(
        logits=batched_logits,
        logit_paddings=jax.numpy.zeros(batched_logits.shape[:2]),
        labels=expected_bytes,
        label_paddings=expected_bytes_paddings,
        blank_id=0,
    ).mean()
    return loss, batched_logits


# %%
# Create the model.
model = ASR(
    rngs=nnx.Rngs(default=0),
    audio_channels=1,
    predicted_classes=256,
)

# %%
# Setup compiled versions of the helper methods.
train_step_jit = nnx.jit(train_step)
eval_step_jit = nnx.jit(eval_step)

# %%
# Load train and validation data and produce the validation batches.
dataset_validation: datasets.arrow_dataset.Dataset = datasets.Dataset.from_file(
    filename="/mnt/workspace/voxpopuli_byte_level/5s_128t_validation.arrow"
)

val_batches = []
batch_size_validation = 4
dataset_sample_index_validation = 0
for i in range(min(len(dataset_validation) // batch_size_validation, 80)):
    (
        batched_waveforms_validation,
        batched_transcripts_validation,
        batched_transcript_paddings_validation,
        dataset_sample_index_validation,
    ) = get_batch(
        dataset=dataset_validation,
        start_index=dataset_sample_index_validation,
        batch_size=batch_size_validation,
    )
    val_batches.append(
        (
            batched_waveforms_validation,
            batched_transcripts_validation,
            batched_transcript_paddings_validation,
        )
    )

# Load the train data after we've already loaded the validation data.
# Safer to load after to prevent accidental bleeding of train into validation via
# bugs/typos.
dataset_train: datasets.arrow_dataset.Dataset = datasets.Dataset.from_file(
    filename="/mnt/workspace/voxpopuli_byte_level/5s_128t_train.arrow"
)

# %%
# Configure a new training run.
run = aim.Run(log_system_params=True)

print(f"{run.hash=}")
print(model)

run.add_tag(value="voxpopuli")

run.set_artifacts_uri(uri="file:///mnt/workspace/aim/artifacts/")
run.log_artifact(path="/mnt/workspace/src/asr_trainer.py", name="asr_trainer.py")
run.log_artifact(path="/mnt/workspace/src/asr.py", name="asr.py")
run.log_artifact(path="/mnt/workspace/src/rnn.py", name="rnn.py")

batch_size = 96
epochs = 500

run["train_dataset_size"] = len(dataset_train)
run["batch_size"] = batch_size
run["epochs"] = epochs

# %%
# Store the audio from the last validation set entry, for reference.
run.track(
    value=[
        aim.Audio(
            np.array(val_batches[-1][0][index, :, 0]),
            format="wav",
            rate=16_000,
            caption=f"{index}: {bytes(
                [x for x in val_batches[-1][1][index].tolist() if x != 0]
            ).decode("utf-8", errors="backslashreplace")}",
        )
        for index in range(val_batches[-1][0].shape[0])
    ],
    name="validation_audio_subset",
    step=0,
    epoch=0,
)
# %%
# Compute initial validation loss.
model.eval()
eval_losses = []
for val_batch in val_batches:
    (
        batched_waveforms_eval,
        batched_transcripts_eval,
        batched_transcript_paddings_eval,
    ) = val_batch
    eval_loss, eval_logits = eval_step_jit(
        model=model,
        batched_waveforms=batched_waveforms_eval,
        expected_bytes=batched_transcripts_eval,
        expected_bytes_paddings=batched_transcript_paddings_eval,
    )
    eval_losses.append(eval_loss)
model.train()

eval_loss = sum(eval_losses) / len(eval_losses)

# %%
# Actually train.
steps_per_epoch = len(dataset_train) / batch_size

# Create the optimizer.
optimizer: opt.Optimizer = opt.Optimizer(
    model=model,
    tx=optax.chain(
        # Adam optimizer.
        optax.scale_by_adam(),
        # Warm up over the first epoch.
        optax.scale_by_schedule(
            step_size_fn=optax.schedules.warmup_constant_schedule(
                init_value=0.00001,
                peak_value=0.001,
                warmup_steps=int(steps_per_epoch),
            )
        ),
        # Warm down as we start to plateau.
        optax.contrib.reduce_on_plateau(
            factor=0.5,
            patience=25,
            cooldown=50,
            accumulation_size=250,
        ),
        optax.scale(step_size=-1.0),
    ),
)

# Allow resuming in the middle of a previous train.
start_step = locals()["step"] if "step" in locals() else 0
dataset_sample_index = (
    locals()["dataset_sample_index"] if "dataset_sample_index" in locals() else 0
)
end_step = int(steps_per_epoch * epochs)

model.train()
for step in tqdm.tqdm(
    iterable=range(start_step, end_step), initial=start_step, total=end_step
):
    try:
        (
            batched_waveforms,
            batched_transcripts,
            batched_transcript_paddings,
            dataset_sample_index,
        ) = get_batch(
            dataset=dataset_train,
            start_index=dataset_sample_index,
            batch_size=batch_size,
        )

        train_loss, logits = train_step_jit(
            model=model,
            optimizer=optimizer,
            validation_loss=eval_loss,
            batched_waveforms=batched_waveforms,
            expected_bytes=batched_transcripts,
            expected_bytes_paddings=batched_transcript_paddings,
        )

        if jnp.any(jnp.isnan(train_loss)):
            print("NaN loss detected.")
            break

        if step % 250 == 0 or step == end_step - 1:
            try:
                model.eval()
                eval_losses = []
                for val_batch in val_batches:
                    (
                        batched_waveforms_eval,
                        batched_transcripts_eval,
                        batched_transcript_paddings_eval,
                    ) = val_batch
                    eval_loss, eval_logits = eval_step_jit(
                        model=model,
                        batched_waveforms=batched_waveforms_eval,
                        expected_bytes=batched_transcripts_eval,
                        expected_bytes_paddings=batched_transcript_paddings_eval,
                    )
                    eval_losses.append(eval_loss)
                model.train()

                eval_loss = sum(eval_losses) / len(eval_losses)
                run.track(
                    value=eval_loss / batched_waveforms_eval.shape[1] * 16_000,
                    name="validation_loss_per_second_of_audio",
                    step=step,
                    epoch=int(step / steps_per_epoch),
                    context={"subset": "validation"},
                )

                token_predictions = jnp.argmax(eval_logits, axis=2).tolist()
                token_predictions = [
                    [
                        token
                        for token in tokens
                        # if token != tokenizer.max_token_value + 1
                        if token != 0
                    ]
                    for tokens in token_predictions
                ]
                token_actuals = batched_transcripts_eval.tolist()
                token_actuals = [
                    [
                        token
                        for token in tokens
                        # if token != tokenizer.max_token_value + 1
                        if token != 0
                    ]
                    for tokens in token_actuals
                ]
                string_actuals = [
                    bytes(seq).decode("utf-8", errors="backslashreplace")
                    for seq in token_actuals
                ]
                string_predictions = [
                    bytes(seq).decode("utf-8", errors="backslashreplace")
                    for seq in token_predictions
                ]

                run.track(
                    value=[
                        aim.Text(f"Predicted: {pred}\nActual   : {act}")
                        for pred, act in zip(string_predictions, string_actuals)
                    ],
                    name="validation_decoded_subset",
                    step=step,
                    epoch=int(step / steps_per_epoch),
                    context={"subset": "validation"},
                )

            except Exception as e:
                print("Error during eval:")
                print(e)

        run.track(
            value=train_loss / batched_waveforms.shape[1] * 16_000,
            name="loss_per_second_of_audio",
            step=step,
            epoch=int(step / steps_per_epoch),
            context={"subset": "train"},
        )
        run.track(
            value=optimizer.opt_state[2].scale.tolist(),
            name="plateau_lr_scale",
            step=step,
            epoch=int(step / steps_per_epoch),
            context={"subset": "train"},
        )
        run.track(
            value=optimizer.opt_state[2].best_value.tolist(),
            name="plateau_loss_best_value",
            step=step,
            epoch=int(step / steps_per_epoch),
            context={"subset": "train"},
        )
        run.track(
            value=optimizer.opt_state[2].plateau_count.tolist(),
            name="plateau_count",
            step=step,
            epoch=int(step / steps_per_epoch),
            context={"subset": "train"},
        )

        # Cleanup time
        del batched_waveforms
        del batched_transcripts
        del batched_transcript_paddings

    except Exception as e:
        print(f"Skipping step {step} due to error:")
        print(traceback.format_exc())
        time.sleep(5)

# %%
# Close out the run.
run.close()

# %%
