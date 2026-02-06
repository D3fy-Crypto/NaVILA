#!/usr/bin/env python3
"""
Add backward hooks to check gradient flow in real-time.
This should be called during model initialization.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def attach_gradient_debug_hooks(model):
    """
    Attach hooks to mm_projector to print gradients during backward.
    This runs DURING backprop so we see real values.
    """
    if not hasattr(model, 'mm_projector'):
        return
    
    projector = model.mm_projector
    
    # Store hook handles
    handles = []
    
    for name, module in projector.named_modules():
        if isinstance(module, torch.nn.Linear):
            def make_hook(module_name):
                def hook(grad_input, grad_output):
                    grad_norm = 0.0
                    if grad_output[0] is not None:
                        grad_norm = grad_output[0].norm().item()
                    logger.info(f"[BACKWARD] {module_name} grad output norm: {grad_norm:.6f}")
                    return grad_input
                return hook
            
            h = module.register_full_backward_hook(make_hook(f"projector.{name}"))
            handles.append(h)
    
    logger.info(f"[DEBUG] Attached {len(handles)} backward hooks to mm_projector")
    return handles


def check_gradient_requires_grad(model):
    """Check which parameters require gradients."""
    logger.info("="*80)
    logger.info("[DEBUG] Parameter gradient status:")
    logger.info("="*80)
    
    for name, p in model.named_parameters():
        if 'mm_projector' in name or 'motion_encoder' in name:
            logger.info(f"{name:60s} requires_grad={p.requires_grad} shape={tuple(p.shape)}")
    
    logger.info("="*80)
