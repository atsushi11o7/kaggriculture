"""Generate displayed-day-12 PPO handoff states with a frozen opening policy."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.jax import history as H
from kaggriculture.policy.jax import policy as P
from kaggriculture.policy.jax.model_factory import create_model
from kaggriculture.policy.opening import land_rule
from kaggriculture.rules import constants as C
from kaggriculture.simulator.action import Action
from kaggriculture.simulator.reset import reset
from kaggriculture.simulator.step import step_batch_lockstep
from kaggriculture.training.checkpoint import read_checkpoint_metadata
from kaggriculture.training.ppo.checkpointing import load_actor_checkpoint
from kaggriculture.training.ppo.state_bank import StateBank

HANDOFF_STEP = 11 * 24
_LAND_DAYS = jnp.asarray([entry[1] for entry in land_rule.LAND_PURCHASE_SCHEDULE], dtype=jnp.int32)
_LAND_PRICES = jnp.asarray(
    [entry[2] for entry in land_rule.LAND_PURCHASE_SCHEDULE], dtype=jnp.float32
)


def _force_scheduled_land(state, action, enabled):
    """Apply the submission land-purchase safety net to batched JAX actions."""
    extra = jnp.sum(state.unlocked_quadrants, axis=-1) - 1
    index = jnp.clip(extra, 0, len(land_rule.LAND_PURCHASE_SCHEDULE) - 1)
    day = state.step[:, None] // 24
    force = (
        (extra < len(land_rule.LAND_PURCHASE_SCHEDULE))
        & (day >= _LAND_DAYS[index])
        & (state.money >= _LAND_PRICES[index] + land_rule.OPERATING_RESERVE)
        & ~jnp.any(action.market_op == C.MARKET_OP_BUY_LAND, axis=-1)
        & jnp.asarray(enabled)[None, :]
    )

    def prepend(values, first):
        shifted = jnp.concatenate(
            [jnp.full_like(values[..., :1], first), values[..., :-1]], axis=-1
        )
        return jnp.where(force[..., None], shifted, values)

    return action._replace(
        market_op=prepend(action.market_op, C.MARKET_OP_BUY_LAND),
        market_arg_idx=prepend(action.market_arg_idx, -1),
        market_n=prepend(action.market_n, 0),
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--learner-seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--no-learner-land-rule", action="store_true")
    parser.add_argument("--no-opponent-land-rule", action="store_true")
    parser.add_argument("--require-learner-lands", type=int)
    parser.add_argument("--require-opponent-lands", type=int)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.games <= 0 or args.batch_size <= 0:
        raise ValueError("games and batch-size must be positive")
    metadata = read_checkpoint_metadata(args.checkpoint)
    config = metadata["config"]
    rules = config["rules"]
    model_config = ModelConfig(**metadata["model_config"])
    model = create_model(model_config)
    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    variables = P.initialize(model, init_key)
    variables = load_actor_checkpoint(args.checkpoint, variables, model_config)
    opponent_path = args.opponent_checkpoint or args.checkpoint
    opponent_metadata = read_checkpoint_metadata(opponent_path)
    if ModelConfig(**opponent_metadata["model_config"]) != model_config:
        raise ValueError("learner and opponent opening checkpoints must share model config")
    opponent_variables = load_actor_checkpoint(
        opponent_path, P.initialize(model, init_key), model_config
    )
    opponent_seat = 1 - args.learner_seat
    land_rule_enabled = [False, False]
    land_rule_enabled[args.learner_seat] = not args.no_learner_land_rule
    land_rule_enabled[opponent_seat] = not args.no_opponent_land_rule

    def combine_actions(learner_action, opponent_action):
        by_seat = [None, None]
        by_seat[args.learner_seat] = learner_action
        by_seat[opponent_seat] = opponent_action
        fields = zip(*by_seat, strict=True)
        return Action(*(jnp.stack(values, axis=1) for values in fields))

    def generate(reset_key, action_key, batch_size):
        state = reset(
            reset_key,
            batch_size,
            board_size=rules["board_size"],
            starting_money=rules["starting_money"],
        )
        counters = H.zeros(batch_size)

        def turn(carry, turn_key):
            state, counters = carry
            learner_key, opponent_key = jax.random.split(turn_key)
            learner_output = P.sample_actions(
                model,
                variables,
                state,
                jnp.full((batch_size,), args.learner_seat, jnp.int32),
                learner_key,
                counters=jax.tree.map(lambda value: value[:, args.learner_seat], counters),
                temperature=args.temperature,
                greedy=not args.sample,
                turns_per_day=rules["turns_per_day"],
                shed_capacity=rules["shed_capacity"],
                hire_mult=rules["hire_mult"],
            )
            opponent_output = P.sample_actions(
                model,
                opponent_variables,
                state,
                jnp.full((batch_size,), opponent_seat, jnp.int32),
                opponent_key,
                counters=jax.tree.map(lambda value: value[:, opponent_seat], counters),
                temperature=args.temperature,
                greedy=not args.sample,
                turns_per_day=rules["turns_per_day"],
                shed_capacity=rules["shed_capacity"],
                hire_mult=rules["hire_mult"],
            )
            action = combine_actions(learner_output.action, opponent_output.action)
            action = _force_scheduled_land(state, action, land_rule_enabled)
            stepped, _, _ = step_batch_lockstep(
                state,
                action,
                board_size=rules["board_size"],
                turns_per_day=rules["turns_per_day"],
                shed_capacity=rules["shed_capacity"],
                weed_chance=rules["weed_chance"],
                shop_unlock_interval=rules["shop_unlock_interval"],
                shop_sell_interval=rules["shop_sell_interval"],
                center_sell_interval=rules["center_sell_interval"],
                hire_mult=rules["hire_mult"],
                max_shop_instances=rules["max_shop_instances"],
                episode_steps=rules["episode_steps"],
            )
            updated = H.update_counters(
                state,
                action,
                counters,
                turns_per_day=rules["turns_per_day"],
                shed_capacity=rules["shed_capacity"],
                hire_mult=rules["hire_mult"],
            )
            return (stepped, updated), None

        keys = jax.random.split(action_key, HANDOFF_STEP)
        return jax.lax.scan(turn, (state, counters), keys)[0]

    generate = jax.jit(generate, static_argnums=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    attempted = 0
    require_lands = (args.require_learner_lands, args.require_opponent_lands)
    filtering = any(value is not None for value in require_lands)
    with args.output.open("wb") as stream:
        while written < args.games:
            size = args.batch_size if filtering else min(args.batch_size, args.games - written)
            key, reset_key, action_key = jax.random.split(key, 3)
            states, counters = jax.device_get(generate(reset_key, action_key, size))
            if set(states.step.tolist()) != {HANDOFF_STEP}:
                raise RuntimeError(f"unexpected handoff steps: {set(states.step.tolist())}")
            land_counts = states.unlocked_quadrants.sum(axis=-1)
            accepted = []
            for index in range(size):
                learner_ok = (
                    args.require_learner_lands is None
                    or land_counts[index, args.learner_seat] == args.require_learner_lands
                )
                opponent_ok = (
                    args.require_opponent_lands is None
                    or land_counts[index, opponent_seat] == args.require_opponent_lands
                )
                if learner_ok and opponent_ok:
                    accepted.append(index)
            for index in accepted[: args.games - written]:
                entry = StateBank(
                    jax.tree.map(lambda value, i=index: value[i], states),
                    jax.tree.map(lambda value, i=index: value[i], counters),
                )
                pickle.dump(entry, stream, protocol=pickle.HIGHEST_PROTOCOL)
                written += 1
            attempted += size
            if attempted > args.games * 100 and written == 0:
                raise RuntimeError("land-count requirements rejected every generated state")
            print(f"generated {written}/{args.games} (attempted {attempted})", flush=True)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "opponent_checkpoint": str(opponent_path.resolve()),
        "learner_seat": args.learner_seat,
        "handoff_step": HANDOFF_STEP,
        "games": args.games,
        "learner_land_rule": not args.no_learner_land_rule,
        "opponent_land_rule": not args.no_opponent_land_rule,
        "require_learner_lands": args.require_learner_lands,
        "require_opponent_lands": args.require_opponent_lands,
        "attempted_games": attempted,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
