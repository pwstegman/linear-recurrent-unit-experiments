# %%
import typing
import datasets
import jax
import numpy as np
import tiktoken
import jax.numpy as jnp
import tqdm
import pyarrow as pa

split = "validation"

voxpopuli = datasets.load_dataset(
    path="facebook/voxpopuli",
    cache_dir="/mnt/workspace/src/data/",
    name="en",
    split=split,
    num_proc=8,
)
voxpopuli = typing.cast(datasets.Dataset, voxpopuli)

# tokenizer = tiktoken.get_encoding("r50k_base")

import time


def get_batch(
    dataset: datasets.Dataset,
    tokenizer: tiktoken.Encoding,
    start_index: int,
    batch_size: int,
):
    audio_len = 16_000 * 30
    text_len = 64

    waveforms = np.zeros(shape=(batch_size, audio_len, 1), dtype=np.float32)
    transcripts = np.full(
        shape=(batch_size, text_len),
        fill_value=tokenizer.max_token_value + 1,
        dtype=np.int32,
    )
    transcripts_padding = np.zeros(shape=(batch_size, text_len), dtype=np.float32)

    offset = 0
    batch_index = 0
    count = 0
    while batch_index < batch_size:
        entry = dataset[start_index + offset]
        offset += 1

        audio = entry["audio"]["array"]
        text = entry["normalized_text"]

        # Skip any samples longer than the max.
        if len(audio) > 16_000 * 30:
            print("Skipping audio")
            continue

        # Skip any samples with too much text.
        tokens = tokenizer.encode(text, disallowed_special=[])

        if len(tokens) > text_len:
            print("Skipping text")
            continue

        waveforms[batch_index, : len(audio), 0] = audio
        transcripts[batch_index, : len(tokens)] = tokens
        transcripts_padding[batch_index, len(tokens) :] = 1.0
        batch_index += 1

    return (
        jax.numpy.array(waveforms),
        jax.numpy.array(transcripts),
        jax.numpy.array(transcripts_padding),
        start_index + offset,
        count,
    )


schema = pa.schema(
    [
        pa.field("audio", pa.binary()),
        pa.field("tokens", pa.binary()),
    ]
)

audio_bin = 16_000 * 5
tokens_bin = 128

writers: dict[tuple[int, int], pa.RecordBatchFileWriter] = {}

for entry in tqdm.tqdm(voxpopuli):
    audio = entry["audio"]["array"]
    text = entry["normalized_text"]
    # tokens = tokenizer.encode(text, disallowed_special=[])
    tokens = [int(char) for char in bytes(text, encoding="utf-8")]

    # Skip complete silence.
    if len(tokens) == 0:
        print("Skipping complete silence")
        continue

    # Determine what group this entry falls into.
    audio_padding_len = (audio_bin - len(audio) % audio_bin) % audio_bin
    audio_padded_len = len(audio) + audio_padding_len

    tokens_padding_len = (tokens_bin - len(tokens) % tokens_bin) % tokens_bin
    tokens_padded_len = len(tokens) + tokens_padding_len

    audio_bin_index = audio_padded_len // audio_bin
    tokens_bin_index = tokens_padded_len // tokens_bin

    audio_np = np.zeros(shape=(audio_bin * audio_bin_index,), dtype=np.float32)
    audio_np[: len(audio)] = audio

    tokens_np = np.full(
        shape=(tokens_bin * tokens_bin_index,),
        # fill_value=tokenizer.max_token_value + 1,
        fill_value=0,
        dtype=np.int32,
    )
    tokens_np[: len(tokens)] = tokens

    group = (audio_bin_index, tokens_bin_index)
    if group not in writers:
        writers[group] = pa.ipc.new_stream(
            sink=pa.OSFile(
                f"/mnt/workspace/voxpopuli_byte_level/{audio_bin_index * 5}s_{tokens_bin_index * tokens_bin}t_{split}.arrow",
                "wb",
            ),
            schema=schema,
            options=pa.ipc.IpcWriteOptions(use_legacy_format=True),
        )

    audio_batch = pa.array([audio_np.tobytes(order="C")])
    tokens_batch = pa.array([tokens_np.tobytes(order="C")])

    batch = pa.RecordBatch.from_arrays(
        [audio_batch, tokens_batch], names=["audio", "tokens"]
    )

    writers[group].write_batch(batch)

for writer in writers.values():
    writer.close()

# %%
import pyarrow as pa

# reader = pa.ipc.open_file(
#     source=pa.OSFile("/mnt/workspace/voxpopuli/5s_16t_train.arrow", "rb")
# )
with pa.memory_map("/mnt/workspace/voxpopuli/5s_16t_train.arrow", "rb") as source:
    loaded_array = pa.ipc.open_file(source).read_all()

# %%
from datasets import Dataset

dataset: datasets.arrow_dataset.Dataset = Dataset.from_file(
    filename="/mnt/workspace/voxpopuli/5s_16t_train.arrow"
)
dataset.data
