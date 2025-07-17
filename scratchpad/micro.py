"""MicroSpeech"""

# %%
import abc
from typing import Any, cast

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import treescope
from jax import Array

treescope.basic_interactive_setup(autovisualize_arrays=True)


# %%
class RNNCell(nnx.Module):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[Array, Array]:
        pass

    @abc.abstractmethod
    def get_initial_state(self, *args, **kwargs) -> Array:
        pass


class SimpleRNNCell(RNNCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        weight_initializer: nnx.Initializer = nnx.initializers.uniform(),
        bias_initializer: nnx.Initializer = nnx.initializers.zeros_init(),
    ) -> None:
        """Create a new cell that processes a single timestep of an RNN.

        Args:
            input_size (int): The size of the input vector at each timestep.
            hidden_size (int): The size of the hidden state vector produced by the cell.
            output_size (int): The size of the output vector produced at this timestep.
            rngs (nnx.Rngs): The random number generator.
            weight_initializer (nnx.Initializer, optional): The initializer for the
                weights. Defaults to nnx.initializers.uniform().
            bias_initializer (nnx.Initializer, optional): The initializer for the
                biases. Defaults to nnx.initializers.zeros_init().
        """

        # Weights.
        self.input_to_hidden = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, input_size))
        )
        self.hidden_to_hidden = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, hidden_size))
        )
        self.hidden_to_output = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, hidden_size))
        )
        self.input_to_output = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, input_size))
        )

        # Biases.
        self.hidden_bias = nnx.Param(
            value=bias_initializer(rngs.biases(), (hidden_size,))
        )
        self.output_bias = nnx.Param(
            value=bias_initializer(rngs.biases(), (output_size,))
        )

        # Store properties.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(
        self,
        hidden_prev: Array,
        input: Array,
    ) -> tuple[Array, Array]:
        """Transform (previous hidden state, input vector) -> (new hidden state, output vector).

        Args:
            hidden_prev (Array): The hidden-state vector of shape
                (self.hidden_size,) from the previous RNNCell call.
            input (Array): The input vector of shape (self.input_size,).

        Returns:
            tuple[Array, Array]: Tuple of (1) the new hidden-state vector with shape
                (self.hidden_size,) and (2) the output vector of shape
                (self.output_size,).
        """
        hidden = nnx.tanh(
            self.hidden_to_hidden @ hidden_prev
            + self.input_to_hidden @ input
            + self.hidden_bias
        )
        output = (
            self.hidden_to_output @ hidden
            + self.input_to_output @ input
            + self.output_bias
        )

        return hidden, output

    def get_initial_state(self) -> Array:
        """Produce the initial hidden-state vector for a series of RNNCell calls.

        Returns:
            Array: The initial hidden state vector.
        """
        return jnp.zeros(shape=(self.hidden_size,))


class LRUCell(RNNCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        weight_initializer: nnx.Initializer = nnx.initializers.uniform(),
    ) -> None:
        # Weights.
        self.input_to_hidden_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, input_size))
        )
        self.input_to_hidden_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, input_size))
        )

        self.hidden_to_hidden_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size,))
        )
        self.hidden_to_hidden_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size,))
        )

        self.hidden_to_output_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, hidden_size))
        )
        self.hidden_to_output_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, hidden_size))
        )

        self.input_to_output = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, input_size))
        )

        # Store properties.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def __call__(
        self,
        hidden_prev: Array,
        input: Array,
    ) -> tuple[Array, Array]:
        hidden_to_hidden_stable = jnp.exp(
            -jnp.exp(self.hidden_to_hidden_real.value)  # Real
            + 1j * jnp.exp(self.hidden_to_hidden_imag.value)  # Imag
        )

        hidden_to_hidden_stable_magnitude = jnp.abs(hidden_to_hidden_stable)
        input_to_hidden_normalization = jnp.sqrt(
            1 - hidden_to_hidden_stable_magnitude * hidden_to_hidden_stable_magnitude
        )

        input_to_hidden = self.input_to_hidden_real + 1j * self.input_to_hidden_imag
        hidden = (
            hidden_to_hidden_stable * hidden_prev
        ) + input_to_hidden_normalization * (input_to_hidden @ input)

        hidden_to_output = self.hidden_to_output_real + 1j * self.hidden_to_output_imag
        output = (hidden_to_output @ hidden).real + (self.input_to_output @ input)

        return hidden, output

    def get_initial_state(self) -> Array:
        """Produce the initial hidden-state vector for a series of RNNCell calls.

        Returns:
            Array: The initial hidden state vector.
        """
        return jnp.zeros(shape=(self.hidden_size,), dtype=jnp.complex64)


class RNN(nnx.Module):
    def __init__(self, cell: RNNCell, unroll: int = 2) -> None:
        self.cell = cell
        self.unroll = unroll

    def __call__(
        self,
        inputs_timeseries: Array,
        initial_hidden_state: Array | None = None,
    ) -> tuple[Array, Array]:
        """Transform a timeseries of input vectors to an equivalently long timeseries of output vectors.

        Each output vector at time t is only affected by the vectors in range [0, t).

        Args:
            inputs_timeseries (Array): An input timeseries of vectors with shape
                (self.cell.input_size, number_of_timesteps).
            initial_hidden_state (Array | None, optional): The initial hidden state
                for RNNCell calls. Has shape (self.cell.hidden_size,). Defaults to
                self.cell.get_initial_state().

        Returns:
            tuple[Array, Array]: Tuple of (1) the final hidden-state vector with shape
                (self.cell.hidden_size,) and (2) the timeseries of output vectors of
                shape (self.cell.output_size, number_of_timesteps).
        """
        if initial_hidden_state is None:
            initial_hidden_state = self.cell.get_initial_state()

        time_axis = 1
        scanner = nnx.scan(
            f=self.cell,
            in_axes=(nnx.Carry, time_axis),
            out_axes=time_axis,
            unroll=self.unroll,
        )
        final_hidden_state, outputs_timeseries = scanner(
            initial_hidden_state, inputs_timeseries
        )

        return final_hidden_state, outputs_timeseries


class LRUCombined(RNNCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rngs: nnx.Rngs,
        unroll: int = 1,
        weight_initializer: nnx.Initializer = nnx.initializers.uniform(),
    ) -> None:
        # Weights.
        self.input_to_hidden_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, input_size))
        )
        self.input_to_hidden_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size, input_size))
        )

        self.hidden_to_hidden_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size,))
        )
        self.hidden_to_hidden_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (hidden_size,))
        )

        self.hidden_to_output_real = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, hidden_size))
        )
        self.hidden_to_output_imag = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, hidden_size))
        )

        self.input_to_output = nnx.Param(
            value=weight_initializer(rngs.weights(), (output_size, input_size))
        )

        # Store properties.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.unroll = unroll

    def cell_step(
        self,
        hidden_to_hidden_stable: Array,
        input_to_hidden_normalization: Array,
        input_to_hidden: Array,
        hidden_to_output: Array,
        hidden_prev: Array,
        input: Array,
    ) -> tuple[Array, Array]:
        hidden = (
            hidden_to_hidden_stable * hidden_prev
        ) + input_to_hidden_normalization * (input_to_hidden @ input)
        output = (hidden_to_output @ hidden).real + (self.input_to_output @ input)

        return hidden, output

    def __call__(
        self,
        input: Array,
        initial_hidden_state: Array | None = None,
    ) -> tuple[Array, Array]:
        if initial_hidden_state is None:
            initial_hidden_state = self.get_initial_state()

        hidden_to_hidden_stable = jnp.exp(
            -jnp.exp(self.hidden_to_hidden_real.value)  # Real
            + 1j * jnp.exp(self.hidden_to_hidden_imag.value)  # Imag
        )
        hidden_to_hidden_stable_magnitude = jnp.abs(hidden_to_hidden_stable)
        input_to_hidden_normalization = jnp.sqrt(
            1 - hidden_to_hidden_stable_magnitude * hidden_to_hidden_stable_magnitude
        )
        input_to_hidden = self.input_to_hidden_real + 1j * self.input_to_hidden_imag
        hidden_to_output = self.hidden_to_output_real + 1j * self.hidden_to_output_imag

        time_axis = 1
        scanner = nnx.scan(
            f=self.cell_step,
            in_axes=(None, None, None, None, nnx.Carry, time_axis),
            out_axes=time_axis,
            unroll=self.unroll,
        )
        final_hidden_state, outputs_timeseries = scanner(
            hidden_to_hidden_stable,
            input_to_hidden_normalization,
            input_to_hidden,
            hidden_to_output,
            initial_hidden_state,
            input,
        )

        return final_hidden_state, outputs_timeseries

    def get_initial_state(self) -> Array:
        """Produce the initial hidden-state vector for a series of RNNCell calls.

        Returns:
            Array: The initial hidden state vector.
        """
        return jnp.zeros(shape=(self.hidden_size,), dtype=jnp.complex64)


# %%
# from timeit import timeit


# class Trainer:
#     def __init__(self, model: RNN) -> None:
#         self.model = model

#     def step(self, sample: Array) -> Array:
#         final_hidden_state, outputs_timeseries = self.model(sample)
#         return outputs_timeseries


# for unroll_amount_log2 in range(0, 10):
#     unroll_amount = 2**unroll_amount_log2

#     sru_rnn = RNN(SimpleRNNCell(128, 512, 128, nnx.Rngs(0)), unroll=unroll_amount)
#     lru_rnn = LRUCombined(128, 512, 128, nnx.Rngs(0), unroll_amount)

#     sru_trainer = Trainer(sru_rnn)
#     lru_trainer = Trainer(lru_rnn)

#     sample = jax.random.uniform(nnx.Rngs(0).foo(), (128, 4096))

#     sru_step = jax.jit(sru_trainer.step).lower(sample).compile()
#     lru_step = jax.jit(lru_trainer.step).lower(sample).compile()

#     runtime_sru_ms = (
#         timeit(stmt=lambda: sru_step(sample).block_until_ready(), number=10) / 10 * 1000
#     )
#     runtime_lru_ms = (
#         timeit(stmt=lambda: lru_step(sample).block_until_ready(), number=10) / 10 * 1000
#     )

#     print(f"{unroll_amount=} {runtime_sru_ms=:#.5g} {runtime_lru_ms=:#.5g}")


# %%
class MicroSpeech(nnx.Module):
    def __init__(self, window_size: int, rngs: nnx.Rngs) -> None:
        self.rnn1 = RNN(
            cell=LRUCell(
                input_size=window_size, hidden_size=32, output_size=32, rngs=rngs
            ),
            unroll=32,
        )
        self.mlp1 = nnx.Linear(in_features=32, out_features=32, rngs=rngs)
        self.rnn2 = RNN(
            cell=LRUCell(input_size=32, hidden_size=32, output_size=256, rngs=rngs),
            unroll=32,
        )
        self.window_size = window_size

    def __call__(
        self,
        inputs_timeseries: Array,
    ) -> Array:
        """Predict the transcript byte at each timestep in the input audio.

        Args:
            inputs_timeseries (Array): An input signal of mono audio with shape
                (1, number_of_timesteps).

        Returns:
            Array: An output with shape (256, number_of_timesteps), where each value in
                axis 0 is the logit representing the estimated likelihood of that byte
                coming next in the transcript at the corresponding time.
        """
        padding = (
            self.window_size - (inputs_timeseries.shape[1] % self.window_size)
        ) % self.window_size
        inputs_timeseries = jnp.pad(
            inputs_timeseries, ((0, 0), (0, padding)), mode="constant"
        )
        inputs_timeseries = inputs_timeseries.reshape((self.window_size, -1), order="F")

        _, signals_out = self.rnn1(inputs_timeseries)
        signals_out: Array = nnx.selu(signals_out)
        signals_out = self.mlp1(signals_out.transpose((1, 0))).transpose((1, 0))
        signals_out: Array = nnx.selu(signals_out)
        _, signals_out = self.rnn2(signals_out)
        return signals_out


# %%
def loss_fn(
    model: MicroSpeech, signal: Array, expected_bytes: Array
) -> tuple[Array, Array]:
    # Create masks for separate loss calculations on silence (null bytes in transcript)
    # and non silence (non-null bytes in transcript).
    mask_non_zero = jnp.array(expected_bytes != 0, dtype=int)
    mask_zeros = 1 - mask_non_zero

    # Run the model to get byte estimates.
    logits = model(signal)

    # Compute loss.
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.transpose((1, 0)), labels=expected_bytes
    )

    # Compute separate aggregated losses for silence and speaking.
    loss_zeros = (loss * mask_zeros).sum() / mask_zeros.sum()
    loss_non_zeros = (loss * mask_non_zero).sum() / mask_non_zero.sum()

    # Weigh silence and speaking equally.
    loss = loss_non_zeros + loss_zeros

    return loss, logits


def train_step(
    model: MicroSpeech,
    optimizer: nnx.Optimizer,
    signal: jax.Array,
    expected_bytes: jax.Array,
) -> jax.Array:
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, signal, expected_bytes)
    optimizer.update(grads=grads)
    return loss


# %%
window_size = 128
micro_speech = MicroSpeech(window_size=window_size, rngs=nnx.Rngs(default=0xBADBEEF))

micro_speech  # type: ignore

# %%
# LibriSpeech is 16kHz with audio clips up to 35 seconds long.
# That means our max input is 35 * 16,000 = 560,000 samples.
# We'll test with orders of magnitude fewer, as the runtime can be extrapolated linearly.
example_signal = jax.random.uniform(nnx.Rngs(0xBEEF).signal(), (1, 560_000))

# Set the expected bytes (i.e., transcript).
expected_bytes_pytree = [0] * (
    (example_signal.shape[1] + window_size - 1) // window_size
)

# Randomly put "hello world" at the end.
transcript = "hello world"
for index, char in enumerate(bytes(transcript, encoding="utf-8")):
    expected_bytes_pytree[-len(transcript) + index] = char

expected_bytes: jax.Array = jnp.array(expected_bytes_pytree, dtype=int)

expected_bytes  # type: ignore

# %%
# Plot the example signal.
plt.plot(example_signal[0, :: example_signal.shape[1] // 1000])
plt.show()

# %%
# Train the model.
import optax

optimizer = nnx.Optimizer(micro_speech, optax.adamw(learning_rate=0.001))

train_step_jit = nnx.jit(train_step)

# Benchmark the JITed method
train_step_jit(
    micro_speech, optimizer, example_signal, expected_bytes
).block_until_ready()  # Triggers JIT compilation.
#!%timeit train_step_jit(micro_speech, optimizer, example_signal, expected_bytes).block_until_ready()

for i in range(10_000):
    loss = train_step_jit(micro_speech, optimizer, example_signal, expected_bytes)
    print(f"{loss=}")

# %%
signal_out = micro_speech(example_signal)
jnp.array([int(byte) for byte in jnp.argmax(signal_out, axis=0)])

# %%
signal_out = micro_speech(jax.random.uniform(nnx.Rngs(0xC0DECAFE1).wee(), (1, 5_600)))
jnp.array([int(byte) for byte in jnp.argmax(signal_out, axis=0)])

# %%
