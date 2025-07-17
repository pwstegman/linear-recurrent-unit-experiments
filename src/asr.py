# %%
import flax
import flax.nnx as nnx
import flax.typing
import jax
from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker
import jax.numpy as jnp

from rnn import LRU, BiLRU, LRUFullParallel


class ExpandContractResidual(nnx.Module):
    def __init__(
        self,
        in_features: int,
        expand_features: int,
        residual_scale: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.expand = nnx.Linear(
            in_features=in_features,
            out_features=expand_features,
            rngs=rngs,
        )
        self.contract = nnx.Linear(
            in_features=expand_features,
            out_features=in_features,
            rngs=rngs,
        )
        self.residual_scale = residual_scale

    def __call__(self, x):
        y = self.expand(x)
        y = nnx.relu(y)
        y = self.contract(y) + x * self.residual_scale
        return y


class FactoredConvolution(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        strides: int,
        feature_group_count: int,
        padding: flax.typing.PaddingLike,
        rngs: nnx.Rngs,
    ) -> None:
        self.depth_wise = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            feature_group_count=feature_group_count,
            padding=padding,
            rngs=rngs,
        )
        self.point_wise = nnx.Linear(
            in_features=out_features,
            out_features=out_features,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        x = self.depth_wise(x)
        x = nnx.relu(x)
        x = self.point_wise(x)
        return x


class ASR(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        audio_channels: int,
        predicted_classes: int,
    ) -> None:
        self.audio_channels = audio_channels
        self.predicted_classes = predicted_classes
        self.deterministic = False
        self.rngs = rngs

        unroll = 1

        self.layers = [
            nnx.Bidirectional(
                forward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
                backward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
            ),
            nnx.Dropout(0.5, rngs=rngs),
            nnx.relu,
            nnx.Bidirectional(
                forward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
                backward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
            ),
            nnx.Dropout(0.5, rngs=rngs),
            nnx.relu,
            nnx.Bidirectional(
                forward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
                backward_rnn=nnx.RNN(
                    cell=nnx.LSTMCell(
                        in_features=512,
                        hidden_features=256,
                        rngs=rngs,
                    ),
                    rngs=rngs,
                    unroll=unroll,
                ),
            ),
            nnx.Dropout(0.5, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                in_features=512,
                out_features=predicted_classes,
                rngs=rngs,
            ),
        ]

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, signals: Float[Array, "batch_size num_timesteps {self.audio_channels}"]
    ) -> Float[Array, "batch_size num_frames {self.predicted_classes}"]:
        # Normalize the signals in the time axis to mean 0 variance 1.
        signals_means = signals.mean(axis=1, keepdims=True)
        signals_stds = signals.std(axis=1, keepdims=True)
        epsilon = 1e-8
        signals = (signals - signals_means) / (signals_stds + epsilon)
        print(f"Normalized: {signals.shape=}")

        # Add gaussian noise.
        # if not self.deterministic:
        #     signals += 0.01 * jax.random.normal(
        #         key=self.rngs.noise(), shape=signals.shape, dtype=signals.dtype
        #     )
        #     signals_means = signals.mean(axis=1, keepdims=True)
        #     signals_stds = signals.std(axis=1, keepdims=True)
        #     epsilon = 1e-8
        #     signals = (signals - signals_means) / (signals_stds + epsilon)

        window_size = 512
        padding = jnp.zeros(
            shape=(
                signals.shape[0],
                (window_size - signals.shape[1] % window_size) % window_size,
                1,
            )
        )
        signals = jnp.concat((signals, padding), axis=1)
        print(f"Padded: {signals.shape=}")

        signals = jnp.reshape(signals, (signals.shape[0], -1, window_size))
        print(f"Reshaped: {signals.shape=}")

        for layer in self.layers:
            signals = layer(signals)
            print(f"{type(layer)} out: {signals.shape=}")

        return signals
