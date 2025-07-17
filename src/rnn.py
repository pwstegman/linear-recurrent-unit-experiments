# %%
from functools import partial

import flax
import flax.typing
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, jaxtyped
from typeguard import typechecked as typechecker


class LRU(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        unroll: int = 1,
        r_min: float = 0.0,
        r_max: float = 1.0,
    ) -> None:
        """Create a new recurrent neural network (RNN) built on the Linear Recurrent
        Unit (LRU).

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels.
            hidden_features: The recurrent hidden-state vector size.
            rngs: The random number generator.
            unroll: How many of the recurrent loop steps to unroll. This is purely a
                run-time optimization meant to speed up the calls. It has no effect on
                model behavior.
        """
        # Initialize the complex diagonal recurrent matrix (i.e., the hidden-state
        # transition matrix).

        uniform_initializer = nnx.initializers.uniform(scale=1.0)

        # Pick a nu that encodes a magnitude between r_min and r_max.
        nu = (-1 / 2) * jnp.log(
            uniform_initializer(key=rngs.weights(), shape=(hidden_features,))
            * (r_max * r_max - r_min * r_min)
            + r_min * r_min
        )

        self.nu_log = nnx.Param(value=jnp.log(nu))

        # Pick a theta in the range [0, pi/10).
        theta = (jnp.pi / 10) * uniform_initializer(
            key=rngs.weights(), shape=(hidden_features,)
        )

        self.theta_log = nnx.Param(value=jnp.log(theta))

        # Initialize the complex and real change-of-basis matrices for input -> hidden,
        # hidden -> output, etc. We follow the column vector convention, so the matrix
        # will be on the left-hand side of any change-of-basis formulas.
        weight_initializer = nnx.initializers.glorot_uniform(out_axis=0, in_axis=1)
        bias_initializer = nnx.initializers.zeros_init()
        self.B = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(hidden_features, in_features),
                dtype=jnp.complex64,
            )
        )
        self.C = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, hidden_features),
                dtype=jnp.complex64,
            )
        )
        self.D = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, in_features),
                dtype=jnp.float32,
            )
        )

        # Complex and real biases.
        self.bias_hidden = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(hidden_features,), dtype=jnp.complex64
            )
        )
        self.bias_output = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(out_features,), dtype=jnp.float32
            )
        )

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.unroll = unroll

    @jaxtyped(typechecker=typechecker)
    def _recurrent_unit(
        self,
        h_t_prev: Complex[Array, "{self.hidden_features}"],
        x_t: Float[Array, "{self.in_features}"],
    ) -> tuple[
        Complex[Array, "{self.hidden_features}"], Float[Array, "{self.out_features}"]
    ]:
        """Map (`h_t_prev`, `x_t`) -> (`h_t`, `y_t`).

        Args:
            h_t_prev: The complex-valued hidden-state vector at time `t-1`.
            x_t: The real-valued input vector at time `t`.

        Returns:
            Tuple of (`h_t`, `y_t`) where `h_t` is the complex-valued hidden-state
            vector at time `t` and `y_t` is the real-valued output vector at time `t`.
        """
        lambda_diag = jnp.exp(
            # Real
            -jnp.exp(self.nu_log.value)
            # Imaginary
            + 1j * jnp.exp(self.theta_log.value)
        )
        lambda_diag_magnitude = jnp.abs(lambda_diag)
        gamma = jnp.sqrt(1 - lambda_diag_magnitude * lambda_diag_magnitude)
        h_t = lambda_diag * h_t_prev
        h_t = h_t + gamma * (self.B @ x_t.astype(jnp.complex64))
        h_t = h_t + self.bias_hidden
        y_t = jnp.real(self.C @ h_t) + self.D @ x_t + self.bias_output

        return h_t, y_t

    @jaxtyped(typechecker=typechecker)
    def _get_initial_state(
        self, batch_size: int = 1
    ) -> Complex[Array, "{batch_size} {self.hidden_features}"]:
        return jnp.zeros(
            shape=(
                batch_size,
                self.hidden_features,
            ),
            dtype=jnp.complex64,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        X: Float[Array, "batch_size num_timesteps {self.in_features}"],
        reverse: bool = False,
    ) -> Float[Array, "batch_size num_timesteps {self.out_features}"]:
        """Map (`X`) -> (`Y`, `h_N`), where N is the number of timesteps.

        Each output vector at time t is only affected by the vectors in range [0, t).

        Args:
            X: The input timeseries. Follows the channels-last convention.

        Returns:
            Tuple (`Y`, `h_N`) where `h_N` is the final hidden-state vector and `Y`
            is the output timeseries of vectors.
        """

        batched_recurrent_unit = jax.vmap(
            fun=self._recurrent_unit, in_axes=0, out_axes=0
        )

        batch_size = X.shape[0]
        time_axis = 1
        h_initial = self._get_initial_state(batch_size=batch_size)
        scanner = nnx.scan(
            f=batched_recurrent_unit,
            in_axes=(nnx.Carry, time_axis),
            out_axes=(nnx.Carry, time_axis),
            unroll=self.unroll,
            reverse=reverse,
        )
        _, Y = scanner(h_t_prev=h_initial, x_t=X)

        return Y


class LRUFullParallel(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        unroll: int = 1,
        r_min: float = 0.0,
        r_max: float = 1.0,
    ) -> None:
        """Create a new recurrent neural network (RNN) built on the Linear Recurrent
        Unit (LRU).

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels.
            hidden_features: The recurrent hidden-state vector size.
            rngs: The random number generator.
            unroll: How many of the recurrent loop steps to unroll. This is purely a
                run-time optimization meant to speed up the calls. It has no effect on
                model behavior.
        """
        # Initialize the complex diagonal recurrent matrix (i.e., the hidden-state
        # transition matrix).

        uniform_initializer = nnx.initializers.uniform(scale=1.0)

        # Pick a nu that encodes a magnitude between r_min and r_max.
        nu = (-1 / 2) * jnp.log(
            uniform_initializer(key=rngs.weights(), shape=(hidden_features,))
            * (r_max * r_max - r_min * r_min)
            + r_min * r_min
        )

        self.nu_log = nnx.Param(value=jnp.log(nu))

        # Pick a theta in the range [0, pi/10).
        theta = (jnp.pi / 10) * uniform_initializer(
            key=rngs.weights(), shape=(hidden_features,)
        )

        self.theta_log = nnx.Param(value=jnp.log(theta))

        # Initialize the complex and real change-of-basis matrices for input -> hidden,
        # hidden -> output, etc. We follow the column vector convention, so the matrix
        # will be on the left-hand side of any change-of-basis formulas.
        weight_initializer = nnx.initializers.glorot_uniform(out_axis=0, in_axis=1)
        self.B = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(hidden_features, in_features),
                dtype=jnp.complex64,
            )
        )
        self.C = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, hidden_features),
                dtype=jnp.complex64,
            )
        )
        self.D = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, in_features),
                dtype=jnp.float32,
            )
        )

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.unroll = unroll

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        X: Float[Array, "batch_size num_timesteps {self.in_features}"],
        reverse: bool = False,
    ) -> Float[Array, "batch_size num_timesteps {self.out_features}"]:
        """Map (`X`) -> (`Y`, `h_N`), where N is the number of timesteps.

        Each output vector at time t is only affected by the vectors in range [0, t).

        Args:
            X: The input timeseries. Follows the channels-last convention.

        Returns:
            Tuple (`Y`, `h_N`) where `h_N` is the final hidden-state vector and `Y`
            is the output timeseries of vectors.
        """

        # Precompute the diagonal and the normalized B matrix.
        lambda_diag = jnp.exp(
            # Real
            -jnp.exp(self.nu_log.value)
            # Imaginary
            + 1j * jnp.exp(self.theta_log.value)
        )
        lambda_diag_magnitude = jnp.abs(lambda_diag)
        gamma = jnp.sqrt(1 - lambda_diag_magnitude * lambda_diag_magnitude)
        B_norm = jnp.expand_dims(gamma, -1) * self.B

        # Precompute the input X transformed by the normalized B matrix.
        print(f"{B_norm.shape=} {X.shape=}")
        B_norm_X = jax.vmap(jax.vmap(fun=lambda x_t: B_norm @ x_t))(
            X.astype(dtype=jnp.complex64)
        )

        # Define the recurrent operation.
        def scan_fn(a, b) -> tuple[Array, Array]:
            # Let 'a' be an arbitrary slice at index i within the scan axis.
            # Let 'b' be the slice immediately after 'a'.
            lambda_a, X_a = a
            lambda_b, X_b = b

            lambda_c = lambda_a * lambda_b
            X_c = lambda_b * X_a + X_b

            return lambda_c, X_c

        def scan(X, reverse: bool):
            # Perform the scan.
            # Repeat the lambda across the time axis.
            lambda_diags = jnp.repeat(
                lambda_diag[
                    None,
                    ...,
                ],
                repeats=X.shape[0],
                axis=0,
            )
            _, hiddens = jax.lax.associative_scan(
                scan_fn, (lambda_diags, X), axis=0, reverse=reverse
            )
            return hiddens

        def forward(B_norm_X, X, reverse=False):
            hiddens = scan(B_norm_X, reverse=reverse)
            outs = jax.vmap(lambda h, x: jnp.real(self.C @ h) + self.D @ x)(hiddens, X)
            return outs

        forward_batched = jax.vmap(forward, in_axes=(0, 0, None), out_axes=0)

        Y = forward_batched(B_norm_X, X, reverse)

        return Y


class BiLRU(nnx.Module):
    def __init__(
        self, lru_forward: LRU | LRUFullParallel, lru_backward: LRU | LRUFullParallel
    ):
        self.lru_forward = lru_forward
        self.lru_backward = lru_backward

    def __call__(self, X):
        return jnp.concat(
            (self.lru_forward(X), self.lru_backward(X, reverse=False)), axis=2
        )


class LRUOptimized(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        unroll: int = 1,
    ) -> None:
        """Create a new recurrent neural network (RNN) built on the Linear Recurrent
        Unit (LRU).

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels.
            hidden_features: The recurrent hidden-state vector size.
            rngs: The random number generator.
            unroll: How many of the recurrent loop steps to unroll. This is purely a
                run-time optimization meant to speed up the calls. It has no effect on
                model behavior.
        """
        # Initialize the complex diagonal recurrent matrix (i.e., the hidden-state
        # transition matrix).
        lambda_initializer = nnx.initializers.uniform(scale=1.0)
        self.nu_log = nnx.Param(
            value=lambda_initializer(key=rngs.weights(), shape=(hidden_features,))
        )
        self.theta_log = nnx.Param(
            value=lambda_initializer(key=rngs.weights(), shape=(hidden_features,))
        )

        # Initialize the complex and real change-of-basis matrices for input -> hidden,
        # hidden -> output, etc. We follow the column vector convention, so the matrix
        # will be on the left-hand side of any change-of-basis formulas.
        weight_initializer = nnx.initializers.glorot_uniform(out_axis=0, in_axis=1)
        bias_initializer = nnx.initializers.zeros_init()
        self.B = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(hidden_features, in_features),
                dtype=jnp.complex64,
            )
        )
        self.C = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, hidden_features),
                dtype=jnp.complex64,
            )
        )
        self.D = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, in_features),
                dtype=jnp.float32,
            )
        )

        # Complex and real biases.
        self.bias_hidden = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(hidden_features,), dtype=jnp.complex64
            )
        )
        self.bias_output = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(out_features,), dtype=jnp.float32
            )
        )

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.unroll = unroll

    @jaxtyped(typechecker=typechecker)
    def _recurrent_unit(
        self,
        h_t_prev: Complex[Array, "{self.hidden_features}"],
        lambda_diag: Complex[Array, "{self.hidden_features}"],
        x_hidden: Complex[Array, "{self.hidden_features}"],
    ) -> tuple[
        Complex[Array, "{self.hidden_features}"],
        Complex[Array, "{self.hidden_features}"],
    ]:
        """Map (`h_t_prev`, `x_t`) -> (`h_t`, `y_t`).

        Args:
            h_t_prev: The complex-valued hidden-state vector at time `t-1`.
            x_t: The real-valued input vector at time `t`.

        Returns:
            Tuple of (`h_t`, `y_t`) where `h_t` is the complex-valued hidden-state
            vector at time `t` and `y_t` is the real-valued output vector at time `t`.
        """
        h_t = lambda_diag * h_t_prev + x_hidden + self.bias_hidden
        return h_t, h_t

    @jaxtyped(typechecker=typechecker)
    def _get_initial_state(
        self, batch_size: int = 1
    ) -> Complex[Array, "{batch_size} {self.hidden_features}"]:
        return jnp.zeros(
            shape=(
                batch_size,
                self.hidden_features,
            ),
            dtype=jnp.complex64,
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, X: Float[Array, "batch_size num_timesteps {self.in_features}"]
    ) -> tuple[
        Float[Array, "batch_size num_timesteps {self.out_features}"],
        Complex[Array, "batch_size {self.hidden_features}"],
    ]:
        """Map (`X`) -> (`Y`, `h_N`), where N is the number of timesteps.

        Each output vector at time t is only affected by the vectors in range [0, t).

        Args:
            X: The input timeseries. Follows the channels-last convention.

        Returns:
            Tuple (`Y`, `h_N`) where `h_N` is the final hidden-state vector and `Y`
            is the output timeseries of vectors.
        """
        # Compute the complex diagonal transition matrix for the hidden-state vector.
        lambda_diag = jnp.exp(
            # Real
            -jnp.exp(self.nu_log.value)
            # Imaginary
            + 1j * jnp.exp(self.theta_log.value)
        )

        # Compute the scaling factor for the X_hidden.
        lambda_diag_magnitude = jnp.abs(lambda_diag)
        gamma = jnp.sqrt(1 - lambda_diag_magnitude * lambda_diag_magnitude)

        # Pre-transform X to the complex hidden vector space.
        X_hidden = X.astype(dtype=jnp.complex64)
        X_hidden = self.B @ jnp.expand_dims(X_hidden, axis=-1)
        X_hidden = jnp.squeeze(X_hidden, axis=-1) + self.bias_hidden
        X_hidden = gamma * X_hidden

        # Scan across the RNN.
        batched_rnn_unit = jax.vmap(
            fun=self._recurrent_unit, in_axes=(0, None, 0), out_axes=0
        )
        time_axis = 1
        batch_size = X.shape[0]
        h_initial = self._get_initial_state(batch_size=batch_size)
        scanner = nnx.scan(
            f=batched_rnn_unit,
            in_axes=(nnx.Carry, None, time_axis),
            out_axes=time_axis,
            unroll=self.unroll,
        )
        h_N, Y = scanner(h_t_prev=h_initial, lambda_diag=lambda_diag, x_hidden=X_hidden)

        # Map Y to the output size.
        Y = jnp.real(self.C @ jnp.expand_dims(a=Y, axis=-1))
        Y = jnp.squeeze(a=Y, axis=-1)

        # Residual connection.
        X_skip = self.D @ jnp.expand_dims(a=X, axis=-1)
        X_skip = jnp.squeeze(a=X_skip, axis=-1) + self.bias_output
        Y = Y + X_skip

        return Y, h_N


class LRUOptimizedHalf(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        rngs: nnx.Rngs,
        unroll: int = 1,
    ) -> None:
        """Create a new recurrent neural network (RNN) built on the Linear Recurrent
        Unit (LRU).

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels.
            hidden_features: The recurrent hidden-state vector size.
            rngs: The random number generator.
            unroll: How many of the recurrent loop steps to unroll. This is purely a
                run-time optimization meant to speed up the calls. It has no effect on
                model behavior.
        """
        # Initialize the complex diagonal recurrent matrix (i.e., the hidden-state
        # transition matrix).
        lambda_initializer = nnx.initializers.uniform(scale=1.0)
        self.nu_log = nnx.Param(
            value=lambda_initializer(key=rngs.weights(), shape=(hidden_features,))
        )
        self.theta_log = nnx.Param(
            value=lambda_initializer(key=rngs.weights(), shape=(hidden_features,))
        )

        # Initialize the complex and real change-of-basis matrices for input -> hidden,
        # hidden -> output, etc. We follow the column vector convention, so the matrix
        # will be on the left-hand side of any change-of-basis formulas.
        weight_initializer = nnx.initializers.glorot_uniform(out_axis=0, in_axis=1)
        bias_initializer = nnx.initializers.zeros_init()
        self.B_real = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(hidden_features, in_features),
                dtype=jnp.float32,
            )
        )
        self.B_imag = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(hidden_features, in_features),
                dtype=jnp.float32,
            )
        )
        self.C_real = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, hidden_features),
                dtype=jnp.float32,
            )
        )
        self.C_imag = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, hidden_features),
                dtype=jnp.float32,
            )
        )
        self.D = nnx.Param(
            value=weight_initializer(
                key=rngs.weights(),
                shape=(out_features, in_features),
                dtype=jnp.float32,
            )
        )

        # Complex and real biases.
        self.bias_hidden_real = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(hidden_features,), dtype=jnp.float32
            )
        )
        self.bias_hidden_imag = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(hidden_features,), dtype=jnp.float32
            )
        )
        self.bias_output = nnx.Param(
            value=bias_initializer(
                key=rngs.biases(), shape=(out_features,), dtype=jnp.float32
            )
        )

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.unroll = unroll

    @jaxtyped(typechecker=typechecker)
    def _recurrent_unit(
        self,
        h_t_prev: tuple[
            Float[Array, "{self.hidden_features}"],
            Float[Array, "{self.hidden_features}"],
        ],
        lambda_diag_real: Float[Array, "{self.hidden_features}"],
        lambda_diag_imag: Float[Array, "{self.hidden_features}"],
        x_hidden_real: Float[Array, "{self.hidden_features}"],
        x_hidden_imag: Float[Array, "{self.hidden_features}"],
    ) -> tuple[
        tuple[
            Float[Array, "{self.hidden_features}"],
            Float[Array, "{self.hidden_features}"],
        ],
        tuple[
            Float[Array, "{self.hidden_features}"],
            Float[Array, "{self.hidden_features}"],
        ],
    ]:
        """Map (`h_t_prev`, `x_t`) -> (`h_t`, `y_t`).

        Args:
            h_t_prev: The complex-valued hidden-state vector at time `t-1`.
            x_t: The real-valued input vector at time `t`.

        Returns:
            Tuple of (`h_t`, `y_t`) where `h_t` is the complex-valued hidden-state
            vector at time `t` and `y_t` is the real-valued output vector at time `t`.
        """
        h_t_prev_real, h_t_prev_imag = h_t_prev

        h_t_real = (
            lambda_diag_real * h_t_prev_real
            - lambda_diag_imag * h_t_prev_imag
            + x_hidden_real
        )
        h_t_imag = (
            lambda_diag_real * h_t_prev_imag
            + lambda_diag_imag * h_t_prev_real
            + x_hidden_imag
        )

        h_t = (h_t_real, h_t_imag)

        return h_t, h_t

    @jaxtyped(typechecker=typechecker)
    def _get_initial_state(self, batch_size: int = 1) -> tuple[
        Float[Array, "{batch_size} {self.hidden_features}"],
        Float[Array, "{batch_size} {self.hidden_features}"],
    ]:
        h_init_real = jnp.zeros(
            shape=(
                batch_size,
                self.hidden_features,
            ),
            dtype=jnp.float16,
        )
        h_init_imag = jnp.zeros(
            shape=(
                batch_size,
                self.hidden_features,
            ),
            dtype=jnp.float16,
        )
        return h_init_real, h_init_imag

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, X: Float[Array, "batch_size num_timesteps {self.in_features}"]
    ) -> Float[Array, "batch_size num_timesteps {self.out_features}"]:
        """Map (`X`) -> (`Y`, `h_N`), where N is the number of timesteps.

        Each output vector at time t is only affected by the vectors in range [0, t).

        Args:
            X: The input timeseries. Follows the channels-last convention.

        Returns:
            Tuple (`Y`, `h_N`) where `h_N` is the final hidden-state vector and `Y`
            is the output timeseries of vectors.
        """
        X = X.astype(dtype=jnp.float16)

        # Compute the complex diagonal transition matrix for the hidden-state vector.
        lambda_diag = jnp.exp(
            # Real
            -jnp.exp(self.nu_log.value)
            # Imaginary
            + 1j * jnp.exp(self.theta_log.value)
        )

        # Compute the scaling factor for the X_hidden.
        lambda_diag_magnitude = jnp.abs(lambda_diag)
        gamma = jnp.sqrt(1 - lambda_diag_magnitude * lambda_diag_magnitude)

        lambda_diag_real_half = jnp.real(lambda_diag).astype(dtype=jnp.float16)
        lambda_diag_imag_half = jnp.imag(lambda_diag).astype(dtype=jnp.float16)
        gamma_half = gamma.astype(jnp.float16)

        # Pre-transform X to the complex hidden vector space.
        B_real_half = self.B_real.value.astype(dtype=jnp.float16)
        B_imag_half = self.B_imag.value.astype(dtype=jnp.float16)
        X_hidden_real = B_real_half @ jnp.expand_dims(a=X, axis=-1)
        X_hidden_imag = B_imag_half @ jnp.expand_dims(a=X, axis=-1)
        X_hidden_real = jnp.squeeze(
            X_hidden_real, axis=-1
        ) + self.bias_hidden_real.value.astype(dtype=jnp.float16)
        X_hidden_imag = jnp.squeeze(
            X_hidden_imag, axis=-1
        ) + self.bias_hidden_imag.value.astype(dtype=jnp.float16)
        X_hidden_real = gamma_half * X_hidden_real
        X_hidden_imag = gamma_half * X_hidden_imag

        # Scan across the RNN.
        batched_rnn_unit = jax.vmap(
            fun=self._recurrent_unit,
            in_axes=((0, 0), None, None, 0, 0),
            out_axes=((0, 0), (0, 0)),
        )
        time_axis = 1
        batch_size = X.shape[0]
        h_initial = self._get_initial_state(batch_size=batch_size)
        scanner = nnx.scan(
            f=batched_rnn_unit,
            in_axes=(nnx.Carry, None, None, time_axis, time_axis),
            out_axes=time_axis,
            unroll=self.unroll,
        )
        (h_N_real, h_N_imag), (Y_real, Y_imag) = scanner(
            h_t_prev=h_initial,
            lambda_diag_real=lambda_diag_real_half,
            lambda_diag_imag=lambda_diag_imag_half,
            x_hidden_real=X_hidden_real,
            x_hidden_imag=X_hidden_imag,
        )

        # Map Y to the output size.
        Y_real_half = jnp.expand_dims(a=Y_real, axis=-1)
        Y_imag_half = jnp.expand_dims(a=Y_imag, axis=-1)
        C_real_half = self.C_real.value.astype(dtype=jnp.float16)
        C_imag_half = self.C_imag.value.astype(dtype=jnp.float16)
        Y_real_half = C_real_half @ Y_real_half - C_imag_half @ Y_imag_half
        Y = jnp.squeeze(a=Y_real_half, axis=-1)

        # Residual connection.
        X_skip = self.D.value.astype(dtype=jnp.float16) @ jnp.expand_dims(a=X, axis=-1)
        X_skip = jnp.squeeze(a=X_skip, axis=-1) + self.bias_output.value.astype(
            jnp.float16
        )
        Y = Y + X_skip

        return Y
