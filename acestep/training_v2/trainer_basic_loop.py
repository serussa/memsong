"""
Basic (non-Fabric) training loop for FixedLoRATrainer.

Extracted from ``FixedLoRATrainer._train_basic`` to keep
``trainer_fixed.py`` under the LOC limit.  This module provides a single
generator function that yields ``TrainingUpdate`` objects exactly like
the Fabric loop, but uses manual ``loss.backward()`` and
``torch.nn.utils.clip_grad_norm_`` instead of Fabric wrappers.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from acestep.phase_memory import PhaseMemory
from acestep.training_v2.optim import build_optimizer, build_scheduler
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.trainer_helpers import configure_memory_features, save_checkpoint, save_final
from acestep.training_v2.ui import TrainingUpdate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extracted helpers
# ---------------------------------------------------------------------------

def _flush_accumulated(
    trainable_params: list,
    optimizer: Any,
    scheduler: Any,
    accumulated_loss: float,
    accumulation_step: int,
    cfg: Any,
    tb: TrainingLogger,
    module: Any,
    epoch: int,
    global_step: int,
    steps_per_epoch: int,
) -> Tuple[int, float, List[TrainingUpdate]]:
    """Clip gradients, step optimizer/scheduler, zero grads, and log.

    Consolidates the duplicated optimizer-step sequence used both inside
    the accumulation check and the end-of-epoch flush.

    Returns:
        ``(global_step, avg_loss, updates)`` where *updates* is a list
        of ``TrainingUpdate`` objects the caller should yield.
    """
    # ---- DEBUG: detect NaN source BEFORE clip (which throws on NaN) ----
    grad_nan_params = []
    for n, p in module.model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            grad_nan_params.append(n)
    if grad_nan_params:
        print("\n[DEBUG] NaN/Inf grad in params:", grad_nan_params[:10])
    # ---- END DEBUG ----

    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    global_step += 1

    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
    _lr = scheduler.get_last_lr()[0]
    updates: List[TrainingUpdate] = []

    if global_step % cfg.log_every == 0:
        tb.log_loss(avg_loss, global_step)
        tb.log_lr(_lr, global_step)
        updates.append(TrainingUpdate(
            step=global_step, loss=avg_loss,
            msg=f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
            steps_per_epoch=steps_per_epoch,
        ))

    if global_step % cfg.log_heavy_every == 0:
        tb.log_per_layer_grad_norms(module.model, global_step)

    # ---- PhaseMemory gate monitoring (no impact on training) ----
    _log_phase_memory_gate(module, tb, global_step)

    return global_step, avg_loss, updates


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_basic_training_loop(
    trainer: Any,
    data_module: Any,
    training_state: Optional[Dict[str, Any]],
) -> Generator[TrainingUpdate, None, None]:
    """Execute the basic (non-Fabric) training loop.

    Args:
        trainer: The ``FixedLoRATrainer`` instance.
        data_module: ``PreprocessedDataModule`` with training data.
        training_state: Optional dict with ``should_stop`` flag.

    Yields:
        ``TrainingUpdate`` tuples for each step/epoch/event.
    """
    cfg = trainer.training_config
    module = trainer.module
    assert module is not None

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yield TrainingUpdate(0, 0.0, "[INFO] Starting basic training loop (no Fabric)", kind="info")

    tb = TrainingLogger(cfg.effective_log_dir)
    train_loader = data_module.train_dataloader()

    trainable_params = [p for p in module.model.parameters() if p.requires_grad]
    if not trainable_params:
        yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
        tb.close()
        return

    device_type = module.device_type if hasattr(module, "device_type") else str(module.device).split(":")[0]
    optimizer_type = getattr(cfg, "optimizer_type", "adamw")
    optimizer = build_optimizer(
        trainable_params,
        optimizer_type=optimizer_type,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        device_type=device_type,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
    total_steps = steps_per_epoch * cfg.max_epochs

    scheduler = build_scheduler(
        optimizer,
        scheduler_type=getattr(cfg, "scheduler_type", "cosine"),
        total_steps=total_steps,
        warmup_steps=cfg.warmup_steps,
        lr=cfg.learning_rate,
        optimizer_type=optimizer_type,
    )

    # -- Training memory features (same as Fabric path) ----------------
    if getattr(cfg, "gradient_checkpointing", True):
        ckpt_ok, cache_off, grads_ok = configure_memory_features(module.model.decoder)
        module.force_input_grads_for_checkpointing = ckpt_ok
        if ckpt_ok:
            yield TrainingUpdate(
                0, 0.0,
                f"[INFO] Gradient checkpointing enabled "
                f"(use_cache={not cache_off}, input_grads={grads_ok})",
                kind="info",
            )

    # -- Resume ---------------------------------------------------------
    start_epoch = 0
    global_step = 0

    if cfg.resume_from and Path(cfg.resume_from).exists():
        try:
            yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
            from acestep.training_v2.trainer_helpers import resume_checkpoint
            resumed = yield from resume_checkpoint(trainer, cfg.resume_from, optimizer, scheduler)
            if resumed is not None:
                start_epoch, global_step = resumed
        except Exception as exc:
            logger.exception("Failed to load checkpoint")
            yield TrainingUpdate(0, 0.0, f"[WARN] Checkpoint load failed: {exc} -- starting fresh", kind="warn")

    accumulation_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    module.model.decoder.train()

    for epoch in range(start_epoch, cfg.max_epochs):
        epoch_loss = 0.0
        num_updates = 0
        epoch_start = time.time()

        for batch in train_loader:
            if training_state and training_state.get("should_stop", False):
                _stop_loss = accumulated_loss * cfg.gradient_accumulation_steps / max(accumulation_step, 1)
                yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped", kind="complete")
                tb.close()
                return

            loss = module.training_step(batch)
            loss = loss / cfg.gradient_accumulation_steps

            # ---- DEBUG ----
            if not torch.isfinite(loss):
                print("\n[DEBUG] loss is NaN/Inf BEFORE backward!!")
                print("[DEBUG] loss value:", loss.item())
            # ---- END DEBUG ----

            loss.backward()
            accumulated_loss += loss.item()
            del loss
            accumulation_step += 1

            if accumulation_step >= cfg.gradient_accumulation_steps:
                global_step, avg_loss, updates = _flush_accumulated(
                    trainable_params, optimizer, scheduler,
                    accumulated_loss, accumulation_step, cfg, tb, module,
                    epoch, global_step, steps_per_epoch,
                )
                yield from updates
                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

                if torch.cuda.is_available() and global_step % cfg.log_every == 0:
                    torch.cuda.empty_cache()

        # Flush remainder
        if accumulation_step > 0:
            global_step, avg_loss, updates = _flush_accumulated(
                trainable_params, optimizer, scheduler,
                accumulated_loss, accumulation_step, cfg, tb, module,
                epoch, global_step, steps_per_epoch,
            )
            yield from updates
            epoch_loss += avg_loss
            num_updates += 1
            accumulated_loss = 0.0
            accumulation_step = 0

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_updates, 1)
        tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
        yield TrainingUpdate(
            step=global_step, loss=avg_epoch_loss,
            msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s",
            kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
        )

        if (epoch + 1) % cfg.save_every_n_epochs == 0:
            ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}")
            save_checkpoint(trainer, optimizer, scheduler, epoch + 1, global_step, ckpt_dir)
            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Checkpoint saved at epoch {epoch + 1}",
                kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                checkpoint_path=ckpt_dir,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Sanity check: did we actually train? ----------------------------
    if global_step == 0:
        tb.close()
        yield TrainingUpdate(
            step=0, loss=0.0,
            msg=(
                "[FAIL] Training completed 0 steps -- no batches were processed.\n"
                "       Possible causes:\n"
                "         - Dataset directory is empty or contains no valid .pt files\n"
                "         - DataLoader failed to yield batches (device/platform issue)\n"
                "       Check the dataset path and try again."
            ),
            kind="fail",
        )
        return

    final_path = str(output_dir / "final")
    save_final(trainer, final_path)
    final_loss = module.training_losses[-1] if module.training_losses else 0.0

    adapter_label = "LoKR" if trainer.adapter_type == "lokr" else "LoRA"
    if trainer.adapter_type == "phase_memory":
        adapter_label = "PhaseMemory"
    tb.flush()
    tb.close()
    yield TrainingUpdate(
        step=global_step, loss=final_loss,
        msg=(
            f"[OK] Training complete! {adapter_label} saved to {final_path}\n"
            f"     For inference, set your LoRA path to: {final_path}"
        ),
        kind="complete",
    )


# ---------------------------------------------------------------------------
# PhaseMemory gate monitoring (no impact on training / no graph mutations)
# ---------------------------------------------------------------------------

def _log_phase_memory_gate(module: Any, tb: Any, global_step: int) -> None:
    """Log gate statistics from any PhaseMemory sub-module found in the model.

    Scans the model for PhaseMemory instances and logs:
    - phase_memory/g_mean  (scalar write gate average)
    - phase_memory/g_std   (scalar write gate std)
    - phase_memory/r_mean  (scalar read gate average)
    - phase_memory/r_std   (scalar read gate std)

    If no PhaseMemory module is found, silently returns.
    """
    log_every = 50
    if global_step % log_every != 0:
        return

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    stats: Dict[str, List[float]] = {
        "g_mean": [],
        "g_std": [],
        "r_mean": [],
        "r_std": [],
        "omega_mean": [],
        "omega_std": [],
        "zmag_mean": [],
        "zmag_max": [],
    }

    for child in module.model.modules():
        if not isinstance(child, PhaseMemory):
            continue
        g_mean = _as_float(getattr(child, "last_g_mean", None))
        g_std = _as_float(getattr(child, "last_g_std", None))
        r_mean = _as_float(getattr(child, "last_r_mean", None))
        r_std = _as_float(getattr(child, "last_r_std", None))
        omega_mean = _as_float(getattr(child, "last_omega_mean", None))
        omega_std = _as_float(getattr(child, "last_omega_std", None))
        zmag_mean = _as_float(getattr(child, "last_zmag_mean", None))
        zmag_max = _as_float(getattr(child, "last_zmag_max", None))

        if g_mean is None:
            continue

        stats["g_mean"].append(g_mean)
        stats["g_std"].append(g_std or 0.0)
        stats["r_mean"].append(r_mean or 0.0)
        stats["r_std"].append(r_std or 0.0)
        stats["omega_mean"].append(omega_mean or 0.0)
        stats["omega_std"].append(omega_std or 0.0)
        stats["zmag_mean"].append(zmag_mean or 0.0)
        stats["zmag_max"].append(zmag_max or 0.0)

    if not stats["g_mean"]:
        return

    def _mean(values: List[float]) -> float:
        return sum(values) / max(len(values), 1)

    g_mean = _mean(stats["g_mean"])
    g_std = _mean(stats["g_std"])
    r_mean = _mean(stats["r_mean"])
    r_std = _mean(stats["r_std"])
    omega_mean = _mean(stats["omega_mean"])
    omega_std = _mean(stats["omega_std"])
    zmag_mean = _mean(stats["zmag_mean"])
    zmag_max = _mean(stats["zmag_max"])

    if any(math.isnan(v) for v in (g_mean, g_std, r_mean, r_std, omega_mean, omega_std, zmag_mean, zmag_max)):
        logger.warning("[PM] NaN detected in PhaseMemory diagnostics")
        return

    logger.info(
        "[PM] "
        f"g={g_mean:.4f}±{g_std:.4f} "
        f"r={r_mean:.4f}±{r_std:.4f} "
        f"omega={omega_mean:.4f}±{omega_std:.4f} "
        f"zmag={zmag_mean:.4f} "
        f"zmax={zmag_max:.4f}"
    )

    if g_mean < 0.01:
        logger.warning("[PM] Gate collapse detected (g_mean < 0.01)")
    if g_mean > 0.99:
        logger.warning("[PM] Gate saturation detected (g_mean > 0.99)")
    if omega_mean < 1e-3:
        logger.warning("[PM] Omega collapse detected (omega_mean < 1e-3)")
    if zmag_max > 5.0:
        logger.warning("[PM] zmag explosion detected (zmag_max > 5.0)")

    if tb is not None:
        tb.log_scalar("phase_memory/g_mean", g_mean, global_step)
        tb.log_scalar("phase_memory/g_std", g_std, global_step)
        tb.log_scalar("phase_memory/r_mean", r_mean, global_step)
        tb.log_scalar("phase_memory/r_std", r_std, global_step)
