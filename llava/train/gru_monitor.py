#!/usr/bin/env python3
"""
Training monitoring callback to log gradient flow and parameter statistics.
Add this to the training script to monitor GRU integration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class GRUTrainingMonitor:
    """Monitor gradient flow and parameter updates during training."""
    
    def __init__(self, model, log_every_n_steps=10):
        self.model = model
        self.log_every_n_steps = log_every_n_steps
        self.step = 0
        self.first_log = True
    
    def log_model_status(self):
        """Log model parameter statistics at start of training."""
        if not self.first_log:
            return
        
        logger.info("="*80)
        logger.info("[Monitor] Model Parameter Status")
        logger.info("="*80)
        
        # Count parameters for each module
        total_params = 0
        trainable_params = 0
        
        if hasattr(self.model, 'llm'):
            llm_params = sum(p.numel() for p in self.model.llm.parameters())
            llm_trainable = sum(p.numel() for p in self.model.llm.parameters() if p.requires_grad)
            logger.info(f"LLM:            {llm_params:>12,} params ({llm_trainable:>12,} trainable)")
            total_params += llm_params
            trainable_params += llm_trainable
        
        if hasattr(self.model, 'vision_tower') and self.model.vision_tower is not None:
            vt_params = sum(p.numel() for p in self.model.vision_tower.parameters())
            vt_trainable = sum(p.numel() for p in self.model.vision_tower.parameters() if p.requires_grad)
            logger.info(f"Vision Tower:   {vt_params:>12,} params ({vt_trainable:>12,} trainable)")
            total_params += vt_params
            trainable_params += vt_trainable
        
        if hasattr(self.model, 'mm_projector') and self.model.mm_projector is not None:
            proj_params = sum(p.numel() for p in self.model.mm_projector.parameters())
            proj_trainable = sum(p.numel() for p in self.model.mm_projector.parameters() if p.requires_grad)
            logger.info(f"MM Projector:   {proj_params:>12,} params ({proj_trainable:>12,} trainable)")
            total_params += proj_params
            trainable_params += proj_trainable
        
        if hasattr(self.model, 'motion_encoder') and self.model.motion_encoder is not None:
            me_params = sum(p.numel() for p in self.model.motion_encoder.parameters())
            me_trainable = sum(p.numel() for p in self.model.motion_encoder.parameters() if p.requires_grad)
            logger.info(f"Motion Encoder: {me_params:>12,} params ({me_trainable:>12,} trainable)")
            
            # Breakdown
            gru_params = sum(p.numel() for p in self.model.motion_encoder.gru.parameters())
            gru_trainable = sum(p.numel() for p in self.model.motion_encoder.gru.parameters() if p.requires_grad)
            logger.info(f"  - GRU:        {gru_params:>12,} params ({gru_trainable:>12,} trainable)")
            
            g2v_params = sum(p.numel() for p in self.model.motion_encoder.projector.parameters())
            g2v_trainable = sum(p.numel() for p in self.model.motion_encoder.projector.parameters() if p.requires_grad)
            logger.info(f"  - Grid2Vision:{g2v_params:>12,} params ({g2v_trainable:>12,} trainable)")
            
            total_params += me_params
            trainable_params += me_trainable
        
        logger.info("-"*80)
        logger.info(f"TOTAL:          {total_params:>12,} params ({trainable_params:>12,} trainable)")
        logger.info(f"Trainable:      {100*trainable_params/total_params:>11.3f}%")
        logger.info("="*80)
        
        self.first_log = False
    
    def log_gradient_norms(self):
        """Log gradient norms for trainable modules."""
        self.step += 1
        
        if self.step % self.log_every_n_steps != 0:
            return {}
        
        metrics = {}
        
        # Compute gradient norms with detailed logging
        if hasattr(self.model, 'mm_projector') and self.model.mm_projector is not None:
            grad_norm = 0.0
            param_count = 0
            grad_count = 0
            for name, p in self.model.mm_projector.named_parameters():
                param_count += 1
                if p.grad is not None:
                    grad_count += 1
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            metrics['grad_norm/mm_projector'] = grad_norm
            
            # CRITICAL: Log detailed gradient status
            if self.step % (self.log_every_n_steps * 10) == 0:
                logger.info(f"[Step {self.step}] mm_projector: {param_count} params, {grad_count}/{param_count} have gradients, norm={grad_norm:.6f}")
        
        if hasattr(self.model, 'motion_encoder') and self.model.motion_encoder is not None:
            # Check GRU gradients (should be None/0)
            gru_grad_norm = 0.0
            gru_grad_count = 0
            for p in self.model.motion_encoder.gru.parameters():
                if p.grad is not None:
                    gru_grad_count += 1
                    gru_grad_norm += p.grad.norm().item() ** 2
            gru_grad_norm = gru_grad_norm ** 0.5
            metrics['grad_norm/motion_gru'] = gru_grad_norm
            
            # Check projector gradients (should be non-zero)
            g2v_grad_norm = 0.0
            g2v_grad_count = 0
            for p in self.model.motion_encoder.projector.parameters():
                if p.grad is not None:
                    g2v_grad_count += 1
                    g2v_grad_norm += p.grad.norm().item() ** 2
            g2v_grad_norm = g2v_grad_norm ** 0.5
            metrics['grad_norm/grid_to_vision'] = g2v_grad_norm
            
            if self.step % (self.log_every_n_steps * 10) == 0:
                logger.info(f"[Step {self.step}] motion_gru: {gru_grad_count} grads, norm={gru_grad_norm:.6f}")
                logger.info(f"[Step {self.step}] grid_to_vision: {g2v_grad_count} grads, norm={g2v_grad_norm:.6f}")
        
        return metrics


def add_training_hooks(model, log_every_n_steps=10):
    """Add monitoring hooks to the model."""
    monitor = GRUTrainingMonitor(model, log_every_n_steps)
    
    # Log model status once
    monitor.log_model_status()
    
    return monitor
