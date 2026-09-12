"""Validate optional AdamW moments before installing them in a fresh optimizer."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def require_finite_model(state) -> None:
    """Check parameters AND buffers before a checkpoint can become visible."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Model state must be a nonempty tensor mapping")
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Model state {name} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite model state: {name}")


def _scalar(value, label):
    if torch.is_tensor(value):
        if value.numel() != 1 or value.is_complex():
            raise ValueError(f"{label} must be a finite scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite scalar")
    return value


def checked_adamw_state(optimizer, saved):
    """Bind saved IDs to current parameters only after checking every moment.

    PyTorch zips parameter lists without checking tensor shapes. Preserve the
    NEW optimizer's execution flags (CPU/CUDA, fused, capturable, foreach), not
    flags from another machine. The caller's learning rate also takes priority.
    """
    if not isinstance(saved, Mapping) or not isinstance(saved.get("state"), Mapping):
        raise ValueError("Optimizer has no state mapping")
    groups = saved.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count mismatch")
    fresh = optimizer.state_dict()
    restored, seen = {}, set()
    for old, live, target in zip(groups, optimizer.param_groups, fresh["param_groups"]):
        if not isinstance(old, Mapping) or not isinstance(old.get("params"), list):
            raise ValueError("Invalid optimizer parameter group")
        if len(old["params"]) != len(live["params"]):
            raise ValueError("Optimizer parameter count mismatch")
        for name in ("lr", "eps", "weight_decay"):
            value = _scalar(old.get(name), name)
            if value < 0 or (name == "eps" and value == 0):
                raise ValueError(f"Invalid optimizer {name}")
        betas = old.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("Invalid optimizer betas")
        if any(not 0 <= _scalar(value, "beta") < 1 for value in betas):
            raise ValueError("Invalid optimizer beta")
        # These settings define the meaning of the saved moments. Changing
        # them requires a reset rather than silently reinterpreting history.
        for name in ("betas", "eps", "weight_decay", "amsgrad", "maximize"):
            if old.get(name) != live[name]:
                raise ValueError(f"Incompatible optimizer {name}")
        for identifier, parameter, target_id in zip(old["params"], live["params"], target["params"]):
            if type(identifier) is not int or identifier in seen:
                raise ValueError("Invalid or duplicate optimizer parameter ID")
            seen.add(identifier)
            values = saved["state"].get(identifier)
            if values is None:
                continue  # AdamW initializes untouched parameters lazily.
            if not isinstance(values, Mapping):
                raise ValueError("Invalid optimizer parameter state")
            required = {"step", "exp_avg", "exp_avg_sq"}
            if live["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(values) != required:
                raise ValueError("Incomplete or unsupported AdamW moment fields")
            counter = _scalar(values["step"], "step")
            if counter < 0 or int(counter) != counter:
                raise ValueError("Optimizer step must be a nonnegative integer")
            for name in required - {"step"}:
                moment = values[name]
                if (not torch.is_tensor(moment) or moment.layout != torch.strided
                        or moment.shape != parameter.shape or moment.dtype != parameter.dtype):
                    raise ValueError(f"Optimizer {name} shape/dtype mismatch")
                if not bool(torch.isfinite(moment).all()):
                    raise ValueError(f"Non-finite optimizer {name}")
                if name != "exp_avg" and bool((moment < 0).any()):
                    raise ValueError(f"Negative optimizer {name}")
            restored[target_id] = dict(values)
    if set(saved["state"]) - seen:
        raise ValueError("Optimizer contains unknown parameter IDs")
    return {"state": restored, "param_groups": fresh["param_groups"]}


def restore_optimizer(factory, path, *, device, log=print):
    """A failed restore cannot leak partially installed state into training."""
    candidate = factory()
    if not path:
        return candidate
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("format", 1) != 1:
            raise ValueError("Unsupported optimizer sidecar format")
        state = checked_adamw_state(candidate, payload.get("optimizer"))
        candidate.load_state_dict(state)
        # Verify again after PyTorch has moved state to the current device.
        checked_adamw_state(candidate, candidate.state_dict())
    except Exception as exc:
        log(f"optimizer sidecar ignored: {type(exc).__name__}: {exc}", flush=True)
        del candidate
        return factory()
    log(f"optimizer moments restored from {path}", flush=True)
    return candidate
