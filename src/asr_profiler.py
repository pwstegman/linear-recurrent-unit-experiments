# %%
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import rnn

rngs = nnx.Rngs(default=0)

rnns = [
    rnn.LRUOptimized(
        in_features=256,
        out_features=256,
        hidden_size=256,
        rngs=rngs,
        unroll=32,
    )
    for _ in range(4)
]

conv = nnx.Conv(
    in_features=1, out_features=256, kernel_size=128, strides=128, rngs=rngs
)


# %%
@jax.profiler.annotate_function
@jax.named_scope(name="forward_pass")
def forward(X) -> tuple[jax.Array, jax.Array]:
    # Stack of RNNs.
    Y = X.reshape((X.shape[0], X.shape[1] // 256, 256), order="C")
    for rnn_layer in rnns:
        Y, h_N = rnn_layer(Y)
    return Y, h_N


in_shape = (2, 131_072, 1)

compiled = jax.jit(fun=forward).lower(jnp.ones(shape=in_shape)).compile()

jax.profiler.start_trace(log_dir="/tmp/jax_trace")

for step in range(10):
    X = jax.random.uniform(key=rngs.X(), shape=in_shape)
    with jax.profiler.StepTraceAnnotation("step", step_num=step):
        result = compiled(X)[1].block_until_ready()

jax.profiler.stop_trace()
