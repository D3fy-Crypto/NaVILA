"""Gradient/parameter monitoring callback for motion-related modules."""

from __future__ import annotations

import math
from typing import Dict, List

import transformers
from transformers.modeling_utils import unwrap_model
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger("transformers")


class GRUTrainingMonitorCallback(transformers.TrainerCallback):
    """Log parameter stats and gradient norms for motion-related modules."""

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._logged_status = False
        self._grad_sumsq: Dict[str, float] = {}
        self._hook_handles: List = []

    def _accumulate_grad(self, key: str, grad):
        if grad is None:
            return
        g = grad.detach()
        self._grad_sumsq[key] = self._grad_sumsq.get(key, 0.0) + float(g.float().pow(2).sum().item())

    @staticmethod
    def _param_stats(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _unwrap(self, model):
        return unwrap_model(model)

    def _log_model_status(self, model):
        if self._logged_status:
            return
        base = self._unwrap(model)

        logger.info("=" * 80)
        logger.info("[Monitor] Model Parameter Status")
        logger.info("=" * 80)

        total_params = 0
        trainable_params = 0

        llm = base.get_llm() if hasattr(base, "get_llm") else getattr(base, "llm", None)
        if llm is not None:
            llm_params, llm_trainable = self._param_stats(llm)
            logger.info(f"LLM:            {llm_params:>12,} params ({llm_trainable:>12,} trainable)")
            total_params += llm_params
            trainable_params += llm_trainable

        vision_tower = base.get_vision_tower() if hasattr(base, "get_vision_tower") else getattr(base, "vision_tower", None)
        if vision_tower is not None:
            vt_params, vt_trainable = self._param_stats(vision_tower)
            logger.info(f"Vision Tower:   {vt_params:>12,} params ({vt_trainable:>12,} trainable)")
            total_params += vt_params
            trainable_params += vt_trainable

        mm_projector = base.get_mm_projector() if hasattr(base, "get_mm_projector") else getattr(base, "mm_projector", None)
        if mm_projector is not None:
            proj_params, proj_trainable = self._param_stats(mm_projector)
            logger.info(f"MM Projector:   {proj_params:>12,} params ({proj_trainable:>12,} trainable)")
            total_params += proj_params
            trainable_params += proj_trainable

        motion_encoder = base.get_motion_encoder() if hasattr(base, "get_motion_encoder") else getattr(base, "motion_encoder", None)
        if motion_encoder is not None:
            me_params, me_trainable = self._param_stats(motion_encoder)
            logger.info(f"Motion Encoder: {me_params:>12,} params ({me_trainable:>12,} trainable)")
            total_params += me_params
            trainable_params += me_trainable

            if hasattr(motion_encoder, "gru"):
                gru_params, gru_trainable = self._param_stats(motion_encoder.gru)
                logger.info(f"  - GRU:        {gru_params:>12,} params ({gru_trainable:>12,} trainable)")
            if hasattr(motion_encoder, "input_proj"):
                ip_params, ip_trainable = self._param_stats(motion_encoder.input_proj)
                logger.info(f"  - InputProj:  {ip_params:>12,} params ({ip_trainable:>12,} trainable)")
            if hasattr(motion_encoder, "embed_proj"):
                ep_params, ep_trainable = self._param_stats(motion_encoder.embed_proj)
                logger.info(f"  - EmbedProj:  {ep_params:>12,} params ({ep_trainable:>12,} trainable)")

        motion_projector = (
            base.get_motion_projector() if hasattr(base, "get_motion_projector") else getattr(base, "motion_projector", None)
        )
        if motion_projector is not None:
            mp_params, mp_trainable = self._param_stats(motion_projector)
            logger.info(f"Motion Projector:{mp_params:>11,} params ({mp_trainable:>12,} trainable)")
            total_params += mp_params
            trainable_params += mp_trainable

        logger.info("-" * 80)
        if total_params > 0:
            logger.info(f"TOTAL:          {total_params:>12,} params ({trainable_params:>12,} trainable)")
            logger.info(f"Trainable:      {100 * trainable_params / total_params:>11.3f}%")
        logger.info("=" * 80)
        self._logged_status = True

    def _register_module_hooks(self, module, key: str):
        if module is None:
            return
        for p in module.parameters():
            if not p.requires_grad:
                continue
            handle = p.register_hook(lambda grad, k=key: self._accumulate_grad(k, grad))
            self._hook_handles.append(handle)
        if key not in self._grad_sumsq:
            self._grad_sumsq[key] = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is not None:
            self._log_model_status(model)
            base = self._unwrap(model)
            # Register hooks for gradient capture.
            mm_projector = (
                base.get_mm_projector() if hasattr(base, "get_mm_projector") else getattr(base, "mm_projector", None)
            )
            self._register_module_hooks(mm_projector, "grad_norm/mm_projector")

            motion_encoder = (
                base.get_motion_encoder() if hasattr(base, "get_motion_encoder") else getattr(base, "motion_encoder", None)
            )
            if motion_encoder is not None:
                if hasattr(motion_encoder, "gru"):
                    self._register_module_hooks(motion_encoder.gru, "grad_norm/motion_gru")
                if hasattr(motion_encoder, "input_proj"):
                    self._register_module_hooks(motion_encoder.input_proj, "grad_norm/motion_input_proj")
                if hasattr(motion_encoder, "embed_proj"):
                    self._register_module_hooks(motion_encoder.embed_proj, "grad_norm/motion_embed_proj")
                if not hasattr(motion_encoder, "gru") and not hasattr(motion_encoder, "input_proj") and not hasattr(
                    motion_encoder, "embed_proj"
                ):
                    self._register_module_hooks(motion_encoder, "grad_norm/motion_encoder")

            motion_projector = (
                base.get_motion_projector()
                if hasattr(base, "get_motion_projector")
                else getattr(base, "motion_projector", None)
            )
            self._register_module_hooks(motion_projector, "grad_norm/motion_projector")

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return
        metrics = {}
        for key, sumsq in self._grad_sumsq.items():
            if sumsq <= 0.0:
                continue
            metrics[key] = math.sqrt(sumsq)
        if metrics:
            print(f"[Step {state.global_step}] Gradient norms: {metrics}")
        # Reset accumulators after logging.
        for key in list(self._grad_sumsq.keys()):
            self._grad_sumsq[key] = 0.0
