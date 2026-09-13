"""PyTorch提出モデルとFlax学習モデルの重み変換。"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import freeze, unfreeze

from kaggriculture.policy.common.config import ModelConfig
from kaggriculture.policy.torch import model as TM


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _attention_to_flax(state: Mapping[str, torch.Tensor], prefix: str, heads: int) -> dict:
    weight = _numpy(state[f"{prefix}.in_proj_weight"])
    bias = _numpy(state[f"{prefix}.in_proj_bias"])
    q_weight, k_weight, v_weight = np.split(weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(bias, 3)
    d_model = q_weight.shape[0]
    head_dim = d_model // heads

    def projection(w: np.ndarray, b: np.ndarray) -> dict:
        return {
            "kernel": w.T.reshape(d_model, heads, head_dim),
            "bias": b.reshape(heads, head_dim),
        }

    out_weight = _numpy(state[f"{prefix}.out_proj.weight"])
    return {
        "query": projection(q_weight, q_bias),
        "key": projection(k_weight, k_bias),
        "value": projection(v_weight, v_bias),
        "out": {
            "kernel": out_weight.T.reshape(heads, head_dim, d_model),
            "bias": _numpy(state[f"{prefix}.out_proj.bias"]),
        },
    }


def _encoder_layer_to_flax(state: Mapping[str, torch.Tensor], prefix: str, heads: int) -> dict:
    return {
        "self_attn": _attention_to_flax(state, f"{prefix}.self_attn", heads),
        "linear1": {
            "kernel": _numpy(state[f"{prefix}.linear1.weight"]).T,
            "bias": _numpy(state[f"{prefix}.linear1.bias"]),
        },
        "linear2": {
            "kernel": _numpy(state[f"{prefix}.linear2.weight"]).T,
            "bias": _numpy(state[f"{prefix}.linear2.bias"]),
        },
        "norm1": {
            "scale": _numpy(state[f"{prefix}.norm1.weight"]),
            "bias": _numpy(state[f"{prefix}.norm1.bias"]),
        },
        "norm2": {
            "scale": _numpy(state[f"{prefix}.norm2.weight"]),
            "bias": _numpy(state[f"{prefix}.norm2.bias"]),
        },
    }


def _decoder_layer_to_flax(state: Mapping[str, torch.Tensor], prefix: str, heads: int) -> dict:
    layer = _encoder_layer_to_flax(state, prefix, heads)
    layer["multihead_attn"] = _attention_to_flax(state, f"{prefix}.multihead_attn", heads)
    layer["norm3"] = {
        "scale": _numpy(state[f"{prefix}.norm3.weight"]),
        "bias": _numpy(state[f"{prefix}.norm3.bias"]),
    }
    return layer


def torch_to_jax(net: TM.PolicyValueNet, config: ModelConfig) -> dict[str, object]:
    """PyTorchモデルをFlaxのvariablesへ変換する。

    Args:
        net: 変換元のPyTorchモデル。
        config: 対応するJAXモデル構成。

    Returns:
        `PolicyValueNet.apply`へ渡せるFlax variables。
    """
    if net.uses_asymmetric_critic != config.use_asymmetric_critic:
        raise ValueError("asymmetric critic setting does not match")
    if net.uses_episode_history != config.use_episode_history:
        raise ValueError("episode history setting does not match")
    state = net.state_dict()
    params: dict[str, object] = {
        "board_position_embedding": _numpy(state["board_position_embedding.weight"]),
        "token_embedding": {
            "embedding": _numpy(state["token_embedding.bag.weight"]),
            "norm": {
                "scale": _numpy(state["token_embedding.norm.weight"]),
                "bias": _numpy(state["token_embedding.norm.bias"]),
            },
        },
        "encoder": {
            "cls_token": _numpy(state["encoder.cls_token"]),
            "owner_embedding": {"embedding": _numpy(state["encoder.owner_embedding.weight"])},
            "zone_embedding": {"embedding": _numpy(state["encoder.zone_embedding.weight"])},
        },
        "decoder": {
            "position_embedding": {"embedding": _numpy(state["decoder.position_embedding.weight"])},
        },
        "policy_proj": {
            "kernel": _numpy(state["decoder.policy_proj.weight"]).T,
            "bias": _numpy(state["decoder.policy_proj.bias"]),
        },
        "value_hidden": {
            "kernel": _numpy(state["value_head.0.weight"]).T,
            "bias": _numpy(state["value_head.0.bias"]),
        },
        "value_out": {
            "kernel": _numpy(state["value_head.2.weight"]).T,
            "bias": _numpy(state["value_head.2.bias"]),
        },
    }
    encoder = params["encoder"]
    decoder = params["decoder"]
    for layer in range(config.num_layers_encoder):
        encoder[f"layer_{layer}"] = _encoder_layer_to_flax(
            state, f"encoder.transformer.layers.{layer}", config.num_heads
        )
    for layer in range(config.num_layers_decoder):
        decoder[f"layer_{layer}"] = _decoder_layer_to_flax(
            state, f"decoder.transformer.layers.{layer}", config.num_heads
        )
    if config.use_asymmetric_critic:
        privileged: dict[str, object] = {
            "cls_token": _numpy(state["privileged_encoder.cls_token"]),
            "owner_embedding": {
                "embedding": _numpy(state["privileged_encoder.owner_embedding.weight"])
            },
            "zone_embedding": {
                "embedding": _numpy(state["privileged_encoder.zone_embedding.weight"])
            },
        }
        for layer in range(config.num_layers_critic):
            privileged[f"layer_{layer}"] = _encoder_layer_to_flax(
                state, f"privileged_encoder.transformer.layers.{layer}", config.num_heads
            )
        params["privileged_encoder"] = privileged
    return {"params": freeze(jax.tree.map(jnp.asarray, params))}


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
    variables: Mapping[str, object], net: TM.PolicyValueNet, *, actor_only: bool = False
) -> None:
    """Flax variablesをPyTorchへ読み込む。`actor_only`ならcriticを除外する。"""
    params = jax.device_get(variables["params"])
    state = net.state_dict()
    token = params["token_embedding"]
    for prefix in ("token_embedding", "encoder.token_embedding", "decoder.token_embedding"):
        _copy(state, f"{prefix}.bag.weight", np.asarray(token["embedding"]))
        _copy(state, f"{prefix}.norm.weight", np.asarray(token["norm"]["scale"]))
        _copy(state, f"{prefix}.norm.bias", np.asarray(token["norm"]["bias"]))
    board = np.asarray(params["board_position_embedding"])
    for name in (
        "board_position_embedding.weight",
        "encoder.position_embedding.weight",
        "decoder.board_position_embedding.weight",
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

    decoder = params["decoder"]
    _copy(
        state,
        "decoder.position_embedding.weight",
        np.asarray(decoder["position_embedding"]["embedding"]),
    )
    for layer in range(net.decoder.transformer.num_layers):
        item = decoder[f"layer_{layer}"]
        prefix = f"decoder.transformer.layers.{layer}"
        _encoder_layer_to_torch(state, prefix, item)
        _attention_to_torch(state, f"{prefix}.multihead_attn", item["multihead_attn"])
        _copy(state, f"{prefix}.norm3.weight", np.asarray(item["norm3"]["scale"]))
        _copy(state, f"{prefix}.norm3.bias", np.asarray(item["norm3"]["bias"]))

    _copy(state, "decoder.policy_proj.weight", np.asarray(params["policy_proj"]["kernel"]).T)
    _copy(state, "decoder.policy_proj.bias", np.asarray(params["policy_proj"]["bias"]))
    if not actor_only:
        _copy(state, "value_head.0.weight", np.asarray(params["value_hidden"]["kernel"]).T)
        _copy(state, "value_head.0.bias", np.asarray(params["value_hidden"]["bias"]))
        _copy(state, "value_head.2.weight", np.asarray(params["value_out"]["kernel"]).T)
        _copy(state, "value_head.2.bias", np.asarray(params["value_out"]["bias"]))

    if net.uses_asymmetric_critic and not actor_only:
        privileged = params["privileged_encoder"]
        for prefix in ("privileged_encoder.token_embedding",):
            _copy(state, f"{prefix}.bag.weight", np.asarray(token["embedding"]))
            _copy(state, f"{prefix}.norm.weight", np.asarray(token["norm"]["scale"]))
            _copy(state, f"{prefix}.norm.bias", np.asarray(token["norm"]["bias"]))
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


def merge_actor_variables(source: Mapping[str, object], target: Mapping[str, object]) -> dict:
    """BC等のactor parameterだけを別構成のJAX criticへ移植する。"""
    result = unfreeze(target)
    source_params = source["params"]
    for name in (
        "board_position_embedding",
        "token_embedding",
        "encoder",
        "decoder",
        "policy_proj",
    ):
        result["params"][name] = source_params[name]
    return freeze(result)
