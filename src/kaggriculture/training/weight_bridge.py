"""Convert parallel-policy Flax weights to the PyTorch submission model."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpy as np
import torch

from kaggriculture.policy.torch.model import PolicyValueNet


def _copy(state: dict[str, torch.Tensor], name: str, value: np.ndarray) -> None:
    tensor = torch.from_numpy(np.array(value, copy=True)).to(dtype=state[name].dtype)
    if tensor.shape != state[name].shape:
        raise ValueError(
            f"shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(state[name].shape)}"
        )
    state[name] = tensor


def _attention_to_torch(
    state: dict[str, torch.Tensor], prefix: str, attention: Mapping[str, object]
) -> None:
    weights = []
    biases = []
    for name in ("query", "key", "value"):
        projection = attention[name]
        kernel = np.asarray(projection["kernel"])
        weights.append(kernel.reshape(kernel.shape[0], -1).T)
        biases.append(np.asarray(projection["bias"]).reshape(-1))
    _copy(state, f"{prefix}.in_proj_weight", np.concatenate(weights, axis=0))
    _copy(state, f"{prefix}.in_proj_bias", np.concatenate(biases))
    out = attention["out"]
    kernel = np.asarray(out["kernel"])
    _copy(state, f"{prefix}.out_proj.weight", kernel.reshape(-1, kernel.shape[-1]).T)
    _copy(state, f"{prefix}.out_proj.bias", np.asarray(out["bias"]))


def _encoder_layer_to_torch(
    state: dict[str, torch.Tensor], prefix: str, layer: Mapping[str, object]
) -> None:
    _attention_to_torch(state, f"{prefix}.self_attn", layer["self_attn"])
    for linear in ("linear1", "linear2"):
        _copy(state, f"{prefix}.{linear}.weight", np.asarray(layer[linear]["kernel"]).T)
        _copy(state, f"{prefix}.{linear}.bias", np.asarray(layer[linear]["bias"]))
    for norm in ("norm1", "norm2"):
        _copy(state, f"{prefix}.{norm}.weight", np.asarray(layer[norm]["scale"]))
        _copy(state, f"{prefix}.{norm}.bias", np.asarray(layer[norm]["bias"]))


def jax_to_torch(
    variables: Mapping[str, object],
    net: PolicyValueNet,
    *,
    actor_only: bool = False,
) -> None:
    """Load a parallel Flax checkpoint into its PyTorch counterpart."""
    params = jax.device_get(variables["params"])
    state = net.state_dict()
    token = params["token_embedding"]
    token_prefixes = ["token_embedding", "encoder.token_embedding"]
    if net.uses_asymmetric_critic and not actor_only:
        token_prefixes.append("privileged_encoder.token_embedding")
    for prefix in token_prefixes:
        _copy(state, f"{prefix}.bag.weight", np.asarray(token["embedding"]))
        _copy(state, f"{prefix}.norm.weight", np.asarray(token["norm"]["scale"]))
        _copy(state, f"{prefix}.norm.bias", np.asarray(token["norm"]["bias"]))

    board = np.asarray(params["board_position_embedding"])
    for name in (
        "board_position_embedding.weight",
        "encoder.position_embedding.weight",
        "query_encoder.board_position_embedding.weight",
    ):
        _copy(state, name, board)

    encoder = params["encoder"]
    _copy(state, "encoder.cls_token", np.asarray(encoder["cls_token"]))
    _copy(
        state, "encoder.owner_embedding.weight", np.asarray(encoder["owner_embedding"]["embedding"])
    )
    _copy(
        state, "encoder.zone_embedding.weight", np.asarray(encoder["zone_embedding"]["embedding"])
    )
    for layer in range(net.encoder.transformer.num_layers):
        _encoder_layer_to_torch(
            state, f"encoder.transformer.layers.{layer}", encoder[f"layer_{layer}"]
        )

    query = params["query_encoder"]
    _copy(
        state,
        "query_encoder.slot_embedding.weight",
        np.asarray(query["slot_embedding"]["embedding"]),
    )
    _copy(
        state,
        "query_encoder.kind_embedding.weight",
        np.asarray(query["kind_embedding"]["embedding"]),
    )
    for index, _ in enumerate(net.query_encoder.layers):
        source = query[f"layer_{index}"]
        prefix = f"query_encoder.layers.{index}"
        _attention_to_torch(state, f"{prefix}.self_attn", source["self_attn"])
        _attention_to_torch(state, f"{prefix}.cross_attn", source["cross_attn"])
        for linear in ("linear1", "linear2"):
            _copy(
                state,
                f"{prefix}.{linear}.weight",
                np.asarray(source[linear]["kernel"]).T,
            )
            _copy(state, f"{prefix}.{linear}.bias", np.asarray(source[linear]["bias"]))
        for norm in ("norm1", "norm2", "norm3"):
            _copy(state, f"{prefix}.{norm}.weight", np.asarray(source[norm]["scale"]))
            _copy(state, f"{prefix}.{norm}.bias", np.asarray(source[norm]["bias"]))

    for name in ("policy_proj", "quantity_condition"):
        _copy(state, f"{name}.weight", np.asarray(params[name]["kernel"]).T)
        _copy(state, f"{name}.bias", np.asarray(params[name]["bias"]))

    if not actor_only:
        value = params["value_head"]
        _copy(state, "value_head.0.weight", np.asarray(value["layers_0"]["kernel"]).T)
        _copy(state, "value_head.0.bias", np.asarray(value["layers_0"]["bias"]))
        _copy(state, "value_head.2.weight", np.asarray(value["layers_2"]["kernel"]).T)
        _copy(state, "value_head.2.bias", np.asarray(value["layers_2"]["bias"]))

    if net.uses_asymmetric_critic and not actor_only:
        privileged = params["privileged_encoder"]
        _copy(state, "privileged_encoder.position_embedding.weight", board)
        _copy(state, "privileged_encoder.cls_token", np.asarray(privileged["cls_token"]))
        _copy(
            state,
            "privileged_encoder.owner_embedding.weight",
            np.asarray(privileged["owner_embedding"]["embedding"]),
        )
        _copy(
            state,
            "privileged_encoder.zone_embedding.weight",
            np.asarray(privileged["zone_embedding"]["embedding"]),
        )
        for layer in range(net.privileged_encoder.transformer.num_layers):
            _encoder_layer_to_torch(
                state,
                f"privileged_encoder.transformer.layers.{layer}",
                privileged[f"layer_{layer}"],
            )

    net.load_state_dict(state)
