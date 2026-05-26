"""PhaseMemory Checkpoint Utilities for ACE-Step Training.

Provides save/load functions for PhaseMemory weights, which are embedded
directly in the DiT model layers (no PEFT/LyCORIS wrappers).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PHASEMEMORY_CONFIG_NAME = "phase_memory_config.json"
_PHASEMEMORY_WEIGHTS_NAME = "phase_memory_weights.safetensors"


def save_phase_memory_weights(
    model: nn.Module,
    output_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Save only PhaseMemory weights and buffers from the model.

    Extracts parameters and buffers whose name contains "phase_memory"
    and writes them as a safetensors file alongside a config JSON.

    Args:
        model: The model containing PhaseMemory sub-modules.
        output_dir: Directory to write weights to.
        metadata: Optional dict to merge into the config file.

    Returns:
        Path to the saved weights directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    state_dict: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "phase_memory" in name:
            state_dict[name] = param.data.clone().cpu()
    for name, buf in model.named_buffers():
        if "phase_memory" in name:
            state_dict[name] = buf.data.clone().cpu()

    if not state_dict:
        logger.warning("[WARN] No PhaseMemory parameters or buffers found to save!")
        return output_dir

    # Save weights as safetensors
    weights_path = os.path.join(output_dir, _PHASEMEMORY_WEIGHTS_NAME)
    try:
        from safetensors.torch import save_file as safe_save

        safe_save(state_dict, weights_path)
        logger.info("[OK] PhaseMemory weights saved to %s (%d tensors)", weights_path, len(state_dict))
    except ImportError:
        # Fallback to PyTorch .pt format
        pt_path = os.path.join(output_dir, "phase_memory_weights.pt")
        torch.save(state_dict, pt_path)
        logger.info("[OK] PhaseMemory weights saved to %s (safetensors not available)", pt_path)

    # Save config
    config = {
        "num_tensors": len(state_dict),
        "tensor_keys": sorted(state_dict.keys()),
    }
    if metadata:
        config.update(metadata)

    config_path = os.path.join(output_dir, _PHASEMEMORY_CONFIG_NAME)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("[OK] PhaseMemory config saved to %s", config_path)
    return output_dir


def load_phase_memory_weights(
    model: nn.Module,
    weights_dir: str,
) -> nn.Module:
    """Load PhaseMemory weights into the model.

    Reads the safetensors (or .pt fallback) file from *weights_dir*
    and loads matching keys into the model's ``phase_memory`` sub-modules.

    Args:
        model: The model to load weights into.
        weights_dir: Directory containing phase_memory_weights.safetensors.

    Returns:
        The model with PhaseMemory weights loaded (same instance).

    Raises:
        FileNotFoundError: If no PhaseMemory weights file is found.
    """
    weights_dir = str(weights_dir)

    # Try safetensors first, then .pt fallback
    sf_path = os.path.join(weights_dir, _PHASEMEMORY_WEIGHTS_NAME)
    pt_path = os.path.join(weights_dir, "phase_memory_weights.pt")

    if os.path.isfile(sf_path):
        from safetensors.torch import load_file as safe_load

        state_dict = safe_load(sf_path)
        logger.info("[OK] Loaded PhaseMemory weights from %s", sf_path)
    elif os.path.isfile(pt_path):
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        logger.info("[OK] Loaded PhaseMemory weights from %s (fallback)", pt_path)
    else:
        raise FileNotFoundError(
            f"No PhaseMemory weights found in {weights_dir}. "
            f"Expected {_PHASEMEMORY_WEIGHTS_NAME} or phase_memory_weights.pt."
        )

    # Selectively load into model
    model_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = 0

    for key, tensor in state_dict.items():
        if key in model_state:
            if model_state[key].shape == tensor.shape:
                model_state[key].copy_(tensor)
                loaded_keys += 1
            else:
                logger.warning(
                    "[WARN] Shape mismatch for %s: model=%s, weights=%s -- skipping",
                    key, model_state[key].shape, tensor.shape,
                )
                skipped_keys += 1
        else:
            logger.debug("[DEBUG] Key %s not found in model -- skipping", key)
            skipped_keys += 1

    logger.info(
        "[OK] PhaseMemory weights loaded: %d keys applied, %d skipped",
        loaded_keys, skipped_keys,
    )
    return model


def verify_phase_memory_weights(output_dir: str) -> bool:
    """Check that saved PhaseMemory weights exist and are non-trivial.

    Args:
        output_dir: Directory to check for PhaseMemory weights.

    Returns:
        True if weights exist and contain non-zero tensors.
    """
    sf_path = os.path.join(output_dir, _PHASEMEMORY_WEIGHTS_NAME)
    pt_path = os.path.join(output_dir, "phase_memory_weights.pt")

    weight_path = None
    if os.path.isfile(sf_path):
        weight_path = sf_path
    elif os.path.isfile(pt_path):
        weight_path = pt_path
    else:
        logger.warning("[WARN] No PhaseMemory weights file found in %s", output_dir)
        return False

    # Quick sanity: load and count non-zero elements
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        state_dict = safe_load(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    total_nonzero = sum((t != 0).sum().item() for t in state_dict.values())
    total_elements = sum(t.numel() for t in state_dict.values())

    if total_nonzero == 0 and total_elements > 0:
        logger.warning(
            "[WARN] PhaseMemory weights are all zeros! Training may have failed."
        )
        return False

    logger.info("[OK] PhaseMemory weights verified: %d/%d non-zero elements", total_nonzero, total_elements)
    return True
