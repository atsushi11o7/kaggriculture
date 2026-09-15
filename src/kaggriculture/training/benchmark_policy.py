"""Measure steady-state fixed-slot policy throughput for selected batch sizes."""

from __future__ import annotations

import argparse
import time

import jax

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import model as M
from kaggriculture.policy.jax import policy as P
from kaggriculture.simulator.reset import reset


def _measure(function, iterations: int) -> float:
    result = function()
    jax.block_until_ready(result)
    started = time.perf_counter()
    for _ in range(iterations):
        result = function()
    jax.block_until_ready(result)
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    config = ModelConfig(128, 4, 512, 4, 3, 0.0, False, False, 2)
    model = M.PolicyValueNet(config)
    variables = P.initialize(model, jax.random.key(0))
    sample = jax.jit(P.sample_self_play_actions, static_argnums=(0,))
    for batch_size in args.batch_sizes:
        state = reset(jax.random.key(batch_size), batch_size)
        key = jax.random.key(batch_size + 1)

        def run(state=state, key=key):
            return sample(model, variables, state, key)

        seconds = _measure(run, args.iterations)
        players = batch_size * 2
        print(
            f"batch={batch_size} seconds/turn={seconds:.6f} "
            f"player-observations/s={players / seconds:.1f}"
        )


if __name__ == "__main__":
    main()
