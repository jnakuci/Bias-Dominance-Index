# ============================================================
# Generic PyTorch utility for computing layer-resolved
# Bias Dominance Index (BDI) for a single model decision.
#
# Main definitions:
#
#   m = logit[class_a] - logit[class_b]
#
#   C_feat^(l) = < grad_{h_l} m, f_l >
#   C_bias^(l) = < grad_{h_l} m, b_l >
#
#   BDI_layer^(l) =
#       |C_bias^(l)| / (|C_feat^(l)| + |C_bias^(l)| + eps)
#
#   C_feat_all = sum_l C_feat^(l)
#   C_bias_all = sum_l C_bias^(l)
#
#   BDI_all =
#       |C_bias_all| / (|C_feat_all| + |C_bias_all| + eps)
#
# Default behavior:
#   - Includes Linear, Conv, and LayerNorm affine offsets.
#   - Excludes BatchNorm from BDI computation.
#   - Excludes Q and K projection biases by default.
#   - Excludes fused QKV projections by default.
#   - Retains separate V/value projection biases.
#
# Critical autograd rule:
#   - The hooked activation must NOT be detached.
#   - Only the numerical feature and bias components are detached.
#
# compute_bdi.py
#
# Copyright (c) 2026 Johan Nakuci
#
# Licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives
# 4.0 International License (CC BY-NC-ND 4.0).
#
# You may share this file with attribution, but you may not use it for
# commercial purposes or distribute modified versions.
#
# See LICENSE in the project root for details.
# ============================================================

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Basic utilities
# ============================================================

def disable_inplace_ops(model: nn.Module) -> nn.Module:
    """
    Disable inplace operations such as ReLU(inplace=True).

    This is recommended because inplace operations can overwrite
    activations captured by forward hooks.
    """
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    return model


def _move_to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """
    Recursively move tensors to device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)

    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]

    return obj


def _maybe_add_batch_dim_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Conservative automatic batching.

    Adds a batch dimension for:
      - 1D tensors, e.g. [features] or [tokens]
      - 3D tensors, e.g. [C, H, W] images

    Leaves 2D tensors unchanged because they may already be [B, D]
    or [B, T].
    """
    if x.ndim in (1, 3):
        return x.unsqueeze(0)
    return x


def _maybe_add_batch_dim(obj: Any) -> Any:
    """
    Recursively add a batch dimension when it is reasonably safe.
    """
    if torch.is_tensor(obj):
        return _maybe_add_batch_dim_tensor(obj)

    if isinstance(obj, dict):
        return {k: _maybe_add_batch_dim(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return tuple(_maybe_add_batch_dim(v) for v in obj)

    if isinstance(obj, list):
        return [_maybe_add_batch_dim(v) for v in obj]

    return obj


def _default_forward(model: nn.Module, x: Any) -> Any:
    """
    Default model call.

    Supports:
      - tensor input: model(x)
      - dict input: model(**x), useful for HuggingFace models
      - tuple/list input: model(*x)
    """
    if isinstance(x, dict):
        return model(**x)

    if isinstance(x, (tuple, list)):
        return model(*x)

    return model(x)


def _extract_logits_default(model_output: Any) -> torch.Tensor:
    """
    Default extraction of logits from model output.

    Supports:
      - tensor outputs
      - HuggingFace-style outputs with .logits
      - dict outputs with key "logits"
      - tuple/list outputs where the first tensor is logits
    """
    if torch.is_tensor(model_output):
        return model_output

    if hasattr(model_output, "logits"):
        return model_output.logits

    if isinstance(model_output, dict) and "logits" in model_output:
        return model_output["logits"]

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item

    raise ValueError(
        "Could not extract logits from model output. "
        "Pass output_getter=... to BDICalculator."
    )


# ============================================================
# Supported modules
# ============================================================

SUPPORTED_BDI_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.LayerNorm,
)


def has_supported_bias(module: nn.Module) -> bool:
    """
    Returns True if the module has an additive offset that can be
    separated from the input-dependent component.

    BatchNorm is intentionally excluded.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return module.bias is not None

    if isinstance(module, nn.LayerNorm):
        return module.elementwise_affine and module.bias is not None

    return False


# ============================================================
# Q/K exclusion logic
# ============================================================

DEFAULT_QK_EXCLUDE_SUBSTRINGS = [
    "q_proj",
    "k_proj",
    "to_q",
    "to_k",
    "q_linear",
    "k_linear",
    "wq",
    "wk",
    "attention.self.query",
    "attention.self.key",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "attn.q_proj",
    "attn.k_proj",
]

DEFAULT_QK_EXCLUDE_TOKENS = {
    "query",
    "key",
}

DEFAULT_FUSED_QKV_EXCLUDE_SUBSTRINGS = [
    "qkv",
    "qkv_proj",
    "in_proj",
    "c_attn",
    "query_key_value",
    "wqkv",
]


def _tokenize_module_name(name: str):
    """
    Tokenize module names on common separators.

    This prevents broad tokens such as 'key' from accidentally matching
    unrelated names such as 'monkey'.
    """
    return [t for t in re.split(r"[.\-_/]+", name.lower()) if t]


def _contains_any_substring(name: str, patterns: Iterable[str]) -> bool:
    lname = name.lower()
    return any(pattern.lower() in lname for pattern in patterns)


def should_exclude_attention_bias(
    layer_name: str,
    exclude_qk_biases: bool = True,
    exclude_fused_qkv: bool = True,
    additional_exclude_name_contains: Optional[Iterable[str]] = None,
) -> bool:
    """
    Decide whether a layer should be excluded from BDI.

    Rationale:
      - Q and K biases affect attention scores through interactions.
        They are not direct additive evidence offsets.
      - V/value biases are retained when implemented as separate modules,
        because they enter the value stream additively.
      - Fused QKV projections are excluded by default because their bias
        vector combines Q, K, and V components.
    """
    if additional_exclude_name_contains is None:
        additional_exclude_name_contains = []

    lname = layer_name.lower()
    tokens = set(_tokenize_module_name(lname))

    if _contains_any_substring(lname, additional_exclude_name_contains):
        return True

    if exclude_qk_biases:
        if _contains_any_substring(lname, DEFAULT_QK_EXCLUDE_SUBSTRINGS):
            return True

        if len(tokens.intersection(DEFAULT_QK_EXCLUDE_TOKENS)) > 0:
            return True

    if exclude_fused_qkv:
        if _contains_any_substring(lname, DEFAULT_FUSED_QKV_EXCLUDE_SUBSTRINGS):
            return True

    return False


def find_bdi_layers(
    model: nn.Module,
    exclude_qk_biases: bool = True,
    exclude_fused_qkv: bool = True,
    additional_exclude_name_contains: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> OrderedDict[str, nn.Module]:
    """
    Automatically find modules with supported additive offsets.

    Included:
      - Linear
      - Conv1d, Conv2d, Conv3d
      - LayerNorm

    Excluded:
      - BatchNorm
      - Q/K projection biases
      - fused QKV projections
    """
    layers = OrderedDict()
    excluded = OrderedDict()

    for name, module in model.named_modules():
        if name == "":
            continue

        if not isinstance(module, SUPPORTED_BDI_MODULES):
            continue

        if not has_supported_bias(module):
            continue

        if should_exclude_attention_bias(
            layer_name=name,
            exclude_qk_biases=exclude_qk_biases,
            exclude_fused_qkv=exclude_fused_qkv,
            additional_exclude_name_contains=additional_exclude_name_contains,
        ):
            excluded[name] = module
            continue

        layers[name] = module

    if verbose:
        print(f"BDI layers included: {len(layers)}")
        print(f"Q/K or fused-QKV layers excluded: {len(excluded)}")
        print("BatchNorm layers excluded by design.")

        if len(excluded) > 0:
            print("\nExcluded attention-related layers:")
            for name, module in excluded.items():
                print(f"  {name} [{module.__class__.__name__}]")

    return layers


# ============================================================
# Bias broadcasting
# ============================================================

def get_bias_contribution(module: nn.Module, output: torch.Tensor) -> torch.Tensor:
    """
    Construct the broadcasted input-independent offset contribution
    for the output of a supported module.

    The feature contribution is:
        f = output - b

    The returned tensor matches output.shape, output.device, and output.dtype.

    Notes:
      - For Linear and Conv layers, this uses the explicit bias parameter.
      - For LayerNorm, this uses the affine beta parameter.
      - BatchNorm is intentionally unsupported.
    """
    if isinstance(module, nn.Linear):
        bias = module.bias.to(device=output.device, dtype=output.dtype)
        shape = [1] * output.ndim
        shape[-1] = bias.numel()
        return bias.view(*shape).expand_as(output)

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        bias = module.bias.to(device=output.device, dtype=output.dtype)
        shape = [1, bias.numel()] + [1] * (output.ndim - 2)
        return bias.view(*shape).expand_as(output)

    if isinstance(module, nn.LayerNorm):
        bias = module.bias.to(device=output.device, dtype=output.dtype)
        normalized_shape = tuple(module.normalized_shape)
        n_norm_dims = len(normalized_shape)
        shape = [1] * (output.ndim - n_norm_dims) + list(normalized_shape)
        return bias.view(*shape).expand_as(output)

    raise TypeError(
        f"Unsupported module type for BDI bias extraction: {type(module)}. "
        "Supported modules are Linear, Conv1d/2d/3d, and LayerNorm."
    )


# ============================================================
# Layer-resolved BDI table
# ============================================================

def make_layer_resolved_bdi_table(
    rows,
    epsilon: float = 1e-12,
    sort_by: str = "depth",
):
    """
    Convert raw per-layer BDI rows into an explicit layer-resolved BDI table.

    sort_by options:
      - "depth"
      - "BDI_layer"
      - "abs_C_bias"
      - "abs_support"
    """
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df.insert(0, "depth_idx", np.arange(len(df)))

    df["C_total_local"] = df["C_feat"] + df["C_bias"]

    df["BDI_layer"] = (
        df["abs_C_bias"] /
        (df["abs_C_feat"] + df["abs_C_bias"] + epsilon)
    )

    df["feature_fraction_layer"] = (
        df["abs_C_feat"] /
        (df["abs_C_feat"] + df["abs_C_bias"] + epsilon)
    )

    df["bias_fraction_layer"] = df["BDI_layer"]
    df["abs_support_layer"] = df["abs_C_feat"] + df["abs_C_bias"]

    df["dominance"] = np.where(
        df["abs_C_bias"] > df["abs_C_feat"],
        "bias-dominant",
        "feature-dominant",
    )

    total_abs_bias = df["abs_C_bias"].sum()
    total_abs_feat = df["abs_C_feat"].sum()
    total_abs_support = df["abs_support_layer"].sum()

    df["bias_share_across_layers"] = (
        df["abs_C_bias"] / (total_abs_bias + epsilon)
    )

    df["feature_share_across_layers"] = (
        df["abs_C_feat"] / (total_abs_feat + epsilon)
    )

    df["support_share_across_layers"] = (
        df["abs_support_layer"] / (total_abs_support + epsilon)
    )

    df["rank_BDI_layer"] = df["BDI_layer"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    df["rank_abs_bias"] = df["abs_C_bias"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    df["rank_abs_support"] = df["abs_support_layer"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    if sort_by == "depth":
        df = df.sort_values("depth_idx", ascending=True)

    elif sort_by == "BDI_layer":
        df = df.sort_values("BDI_layer", ascending=False)

    elif sort_by == "abs_C_bias":
        df = df.sort_values("abs_C_bias", ascending=False)

    elif sort_by == "abs_support":
        df = df.sort_values("abs_support_layer", ascending=False)

    else:
        raise ValueError(
            "sort_by must be one of: "
            "'depth', 'BDI_layer', 'abs_C_bias', 'abs_support'"
        )

    return df.reset_index(drop=True)


def summarize_bdi_all_from_layers(
    layer_df,
    epsilon: float = 1e-12,
):
    """
    Compute aggregate BDI_all from the layer-resolved table.
    """
    if layer_df is None or len(layer_df) == 0:
        return {
            "C_feat_all": np.nan,
            "C_bias_all": np.nan,
            "BDI_all": np.nan,
            "n_layers": 0,
            "n_bias_dominant_layers": 0,
            "fraction_bias_dominant_layers": np.nan,
            "mean_BDI_layer": np.nan,
            "median_BDI_layer": np.nan,
            "max_BDI_layer": np.nan,
            "max_abs_bias_layer": None,
            "max_BDI_layer_name": None,
        }

    C_feat_all = float(layer_df["C_feat"].sum())
    C_bias_all = float(layer_df["C_bias"].sum())

    BDI_all = abs(C_bias_all) / (
        abs(C_feat_all) + abs(C_bias_all) + epsilon
    )

    n_layers = int(len(layer_df))
    n_bias_dominant = int(
        (layer_df["abs_C_bias"] > layer_df["abs_C_feat"]).sum()
    )

    return {
        "C_feat_all": C_feat_all,
        "C_bias_all": C_bias_all,
        "BDI_all": BDI_all,
        "n_layers": n_layers,
        "n_bias_dominant_layers": n_bias_dominant,
        "fraction_bias_dominant_layers": n_bias_dominant / max(n_layers, 1),
        "mean_BDI_layer": float(layer_df["BDI_layer"].mean()),
        "median_BDI_layer": float(layer_df["BDI_layer"].median()),
        "max_BDI_layer": float(layer_df["BDI_layer"].max()),
        "max_abs_bias_layer": layer_df.loc[
            layer_df["abs_C_bias"].idxmax(), "layer"
        ],
        "max_BDI_layer_name": layer_df.loc[
            layer_df["BDI_layer"].idxmax(), "layer"
        ],
    }



# ============================================================
# BDI calculator
# ============================================================

class BDICalculator:
    """
    Compute layer-resolved and aggregate BDI for a single model decision.

    Default margin:
        m = logit[class_a] - logit[class_b]

    If class_a and class_b are not provided, the model's top-1 and
    top-2 predicted classes are used.

    For language models or masked-token models, pass logits_selector.
    Example:
        logits_selector=lambda logits: logits[:, mask_idx, :]

    For non-classifier models, pass a custom margin_fn that returns
    a scalar tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[OrderedDict[str, nn.Module]] = None,
        epsilon: float = 1e-12,
        device: Optional[Union[str, torch.device]] = None,
        output_getter: Optional[Callable[[Any], torch.Tensor]] = None,
        logits_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
        exclude_qk_biases: bool = True,
        exclude_fused_qkv: bool = True,
        additional_exclude_name_contains: Optional[Iterable[str]] = None,
        disable_inplace: bool = True,
        temporarily_enable_parameter_grads: bool = True,
        verbose: bool = True,
    ):
        self.model = model
        self.model.eval()

        if disable_inplace:
            self.model = disable_inplace_ops(self.model)

        self.epsilon = epsilon
        self.output_getter = output_getter
        self.logits_selector = logits_selector
        self.forward_fn = forward_fn
        self.temporarily_enable_parameter_grads = temporarily_enable_parameter_grads

        if device is None:
            device = next(model.parameters()).device

        self.device = device

        if target_layers is None:
            target_layers = find_bdi_layers(
                model=self.model,
                exclude_qk_biases=exclude_qk_biases,
                exclude_fused_qkv=exclude_fused_qkv,
                additional_exclude_name_contains=additional_exclude_name_contains,
                verbose=verbose,
            )

        self.target_layers = target_layers
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.handles = []

        if len(self.target_layers) == 0:
            raise ValueError(
                "No BDI-compatible layers found. "
                "Check that the model has Linear, Conv, or LayerNorm layers with bias terms."
            )

    def _clear_hooks_and_cache(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cache = OrderedDict()

    def _make_hook(self, layer_name: str, module: nn.Module):
        """
        Forward hook for BDI.

        Critical:
          - Store the raw output tensor as `activation`.
          - Do NOT detach `activation`.
          - Detach only feature and bias numerical components.
        """
        def hook(mod, inputs, output):
            if not torch.is_tensor(output):
                return

            if not output.requires_grad:
                return

            # Graph-connected activation. Do not detach.
            activation = output

            # Numerical components used only for dot products with the gradient.
            # These can and should be detached.
            b_numeric = get_bias_contribution(module, activation).detach()
            f_numeric = (activation.detach() - b_numeric).detach()

            self.cache[layer_name] = {
                "module_type": module.__class__.__name__,
                "activation": activation,
                "feature": f_numeric,
                "bias": b_numeric,
                "output_shape": tuple(output.shape),
            }

        return hook

    def _register_hooks(self):
        for layer_name, module in self.target_layers.items():
            handle = module.register_forward_hook(self._make_hook(layer_name, module))
            self.handles.append(handle)

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if self.output_getter is not None:
            logits = self.output_getter(model_output)
        else:
            logits = _extract_logits_default(model_output)

        if self.logits_selector is not None:
            logits = self.logits_selector(logits)

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        return logits

    def _compute_default_margin(
        self,
        logits: torch.Tensor,
        class_a: Optional[int] = None,
        class_b: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute top-1 versus top-2 margin unless class_a/class_b are supplied.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"Expected selected logits to have shape [B, C], got {tuple(logits.shape)}. "
                "For sequence models, pass logits_selector, e.g. "
                "logits_selector=lambda logits: logits[:, mask_idx, :]."
            )

        if logits.shape[0] != 1:
            raise ValueError(
                "BDICalculator.compute is written for one trial at a time. "
                "Pass a batch with B=1 or loop over trials."
            )

        pred_top2 = torch.topk(logits[0], k=2).indices
        pred_top1 = int(pred_top2[0].item())
        pred_top2_class = int(pred_top2[1].item())

        if class_a is None:
            class_a = pred_top1

        if class_b is None:
            class_b = pred_top2_class

        margin = logits[0, class_a] - logits[0, class_b]

        with torch.no_grad():
            probs = torch.softmax(logits[0], dim=-1)
            confidence = float(probs[pred_top1].detach().cpu().item())

        meta = {
            "class_a": int(class_a),
            "class_b": int(class_b),
            "top1": pred_top1,
            "top2": pred_top2_class,
            "confidence": confidence,
        }

        return margin, meta

    def _parse_custom_margin_output(
        self,
        margin_output: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Accept either:
            margin
        or:
            margin, metadata_dict
        """
        if isinstance(margin_output, tuple):
            margin = margin_output[0]
            meta = margin_output[1] if len(margin_output) > 1 else {}
        else:
            margin = margin_output
            meta = {}

        if not torch.is_tensor(margin):
            raise ValueError(
                "Custom margin_fn must return a scalar torch.Tensor "
                "so that gradients can be computed."
            )

        margin = margin.squeeze()

        if margin.ndim != 0:
            raise ValueError(
                f"Custom margin_fn must return a scalar tensor, got shape {tuple(margin.shape)}."
            )

        if meta is None:
            meta = {}

        return margin, dict(meta)

    def _temporarily_enable_grads(self):
        """
        Temporarily set parameter requires_grad=True.

        This helps when a loaded model has been frozen. The original
        requires_grad states are restored after computation.
        """
        old_states = []

        if not self.temporarily_enable_parameter_grads:
            return old_states

        for param in self.model.parameters():
            old_states.append((param, param.requires_grad))
            param.requires_grad_(True)

        return old_states

    @staticmethod
    def _restore_grad_states(old_states):
        for param, old_state in old_states:
            param.requires_grad_(old_state)

    def compute(
        self,
        x: Any,
        class_a: Optional[int] = None,
        class_b: Optional[int] = None,
        margin_fn: Optional[
            Callable[[Any], Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]]
        ] = None,
        add_batch_dim_if_needed: bool = True,
        return_dataframe: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute BDI for one input trial.
        """
        self._clear_hooks_and_cache()
        self._register_hooks()

        old_grad_states = self._temporarily_enable_grads()

        try:
            self.model.zero_grad(set_to_none=True)

            x = _move_to_device(x, self.device)

            if add_batch_dim_if_needed:
                x = _maybe_add_batch_dim(x)

            with torch.enable_grad():
                model_output = (
                    self.forward_fn(self.model, x)
                    if self.forward_fn is not None
                    else _default_forward(self.model, x)
                )

                logits = None

                if margin_fn is None:
                    logits = self._extract_logits(model_output)
                    margin, margin_meta = self._compute_default_margin(
                        logits=logits,
                        class_a=class_a,
                        class_b=class_b,
                    )
                else:
                    margin, margin_meta = self._parse_custom_margin_output(
                        margin_fn(model_output)
                    )

                activation_names = []
                activation_tensors = []

                for layer_name, item in self.cache.items():
                    activation = item["activation"]

                    if torch.is_tensor(activation) and activation.requires_grad:
                        activation_names.append(layer_name)
                        activation_tensors.append(activation)

                if len(activation_tensors) == 0:
                    raise RuntimeError(
                        "No graph-connected hooked activations were found. "
                        "Do not run the model under torch.no_grad() or torch.inference_mode(). "
                        "Also make sure at least some model parameters require gradients."
                    )

                grads = torch.autograd.grad(
                    outputs=margin,
                    inputs=activation_tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            rows = []

            for layer_name, grad in zip(activation_names, grads):
                if grad is None:
                    continue

                item = self.cache[layer_name]

                f = item["feature"]
                b = item["bias"]

                grad_numeric = grad.detach()

                C_feat = torch.sum(grad_numeric * f).item()
                C_bias = torch.sum(grad_numeric * b).item()

                rows.append({
                    "layer": layer_name,
                    "module_type": item["module_type"],
                    "output_shape": item["output_shape"],
                    "C_feat": C_feat,
                    "C_bias": C_bias,
                    "abs_C_feat": abs(C_feat),
                    "abs_C_bias": abs(C_bias),
                    "bias_dominant": abs(C_bias) > abs(C_feat),
                })

            layer_results = make_layer_resolved_bdi_table(
                rows,
                epsilon=self.epsilon,
                sort_by="depth",
            )

            bdi_summary = summarize_bdi_all_from_layers(
                layer_results,
                epsilon=self.epsilon,
            )

            result = {
                "margin": float(margin.detach().cpu().item()),
                "C_feat_all": bdi_summary["C_feat_all"],
                "C_bias_all": bdi_summary["C_bias_all"],
                "BDI_all": bdi_summary["BDI_all"],
                "BDI_summary": bdi_summary,
                "layer_results": (
                    layer_results
                    if return_dataframe
                    else layer_results.to_dict("records")
                ),
                "included_layers": list(self.target_layers.keys()),
                "logits": logits.detach().cpu() if torch.is_tensor(logits) else None,
            }

            result.update(margin_meta)

            return result

        finally:
            self._restore_grad_states(old_grad_states)
            self._clear_hooks_and_cache()


# ============================================================
# Convenience wrapper
# ============================================================

def compute_bdi_single_trial(
    model: nn.Module,
    x: Any,
    class_a: Optional[int] = None,
    class_b: Optional[int] = None,
    target_layers: Optional[OrderedDict[str, nn.Module]] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_getter: Optional[Callable[[Any], torch.Tensor]] = None,
    logits_selector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    margin_fn: Optional[
        Callable[[Any], Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]]
    ] = None,
    exclude_qk_biases: bool = True,
    exclude_fused_qkv: bool = True,
    additional_exclude_name_contains: Optional[Iterable[str]] = None,
    add_batch_dim_if_needed: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    One-call interface for computing BDI on a single trial.
    """
    calculator = BDICalculator(
        model=model,
        target_layers=target_layers,
        device=device,
        output_getter=output_getter,
        logits_selector=logits_selector,
        forward_fn=forward_fn,
        exclude_qk_biases=exclude_qk_biases,
        exclude_fused_qkv=exclude_fused_qkv,
        additional_exclude_name_contains=additional_exclude_name_contains,
        verbose=verbose,
    )

    return calculator.compute(
        x=x,
        class_a=class_a,
        class_b=class_b,
        margin_fn=margin_fn,
        add_batch_dim_if_needed=add_batch_dim_if_needed,
    )


# ============================================================
# Minimal example
# ============================================================

if __name__ == "__main__":

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 3),
            )

        def forward(self, x):
            return self.net(x)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyNet().to(device).eval()
    x = torch.randn(5)

    result = compute_bdi_single_trial(
        model=model,
        x=x,
        device=device,
        verbose=True,
    )

    print("\nBDI_all:", result["BDI_all"])
    print("Margin:", result["margin"])
    print("Top1:", result.get("top1"))
    print("Top2:", result.get("top2"))

    layer_df = result["layer_results"]

    print("\nLayer-resolved BDI:")
    print(layer_df[[
        "depth_idx",
        "layer",
        "module_type",
        "C_feat",
        "C_bias",
        "BDI_layer",
        "dominance",
        "bias_share_across_layers",
        "support_share_across_layers",
    ]])
