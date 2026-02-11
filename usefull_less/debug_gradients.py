#!/usr/bin/env python3
"""
Real-time gradient flow verification.
Run this in parallel to training to check if mm_projector is actually getting gradients.
"""

import sys
import json
import time
import re
from pathlib import Path
from collections import deque

def parse_loss_from_log(logfile, last_n_lines=500):
    """Parse loss values and check patterns from log file."""
    try:
        with open(logfile, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None
    
    # Get last N lines
    recent = lines[-last_n_lines:] if len(lines) > last_n_lines else lines
    
    # Find loss patterns
    losses = []
    grad_norms = {}
    
    for line in recent:
        # Match {'loss': X.XXX, ...}
        if "'loss':" in line or '"loss":' in line:
            try:
                # Extract dict-like string
                match = re.search(r'\{.*?\}', line)
                if match:
                    dict_str = match.group(0)
                    # Try to parse as dict
                    try:
                        parsed = eval(dict_str)
                        if isinstance(parsed, dict) and 'loss' in parsed:
                            losses.append({
                                'loss': parsed['loss'],
                                'step': parsed.get('step', '?'),
                                'epoch': parsed.get('epoch', '?'),
                                'grad_norm/mm_projector': parsed.get('grad_norm/mm_projector', '?'),
                                'grad_norm/motion_gru': parsed.get('grad_norm/motion_gru', '?'),
                                'grad_norm/grid_to_vision': parsed.get('grad_norm/grid_to_vision', '?'),
                            })
                    except:
                        pass
            except:
                pass
    
    return losses

def main():
    logfile = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/sanity_check_run.log"
    
    print("="*80)
    print("GRADIENT FLOW VERIFICATION")
    print("="*80)
    print(f"Monitoring: {logfile}\n")
    
    last_checked = 0
    while True:
        losses = parse_loss_from_log(logfile)
        
        if losses and len(losses) > last_checked:
            new_losses = losses[last_checked:]
            
            for entry in new_losses:
                print(f"Step {entry['step']}: loss={entry['loss']:.4f} | "
                      f"mm_proj_grad={entry['grad_norm/mm_projector']} | "
                      f"motion_gru_grad={entry['grad_norm/motion_gru']} | "
                      f"grid2vis_grad={entry['grad_norm/grid_to_vision']}")
            
            last_checked = len(losses)
            
            # Analysis
            if len(losses) >= 2:
                recent = losses[-5:]
                
                # Check loss trend
                loss_trend = [e['loss'] for e in recent]
                print(f"\n  Loss trend (last 5): {[f'{l:.4f}' for l in loss_trend]}")
                
                # Check if mm_projector grad is 0
                mm_grad_values = [e['grad_norm/mm_projector'] for e in recent if e['grad_norm/mm_projector'] != '?']
                if mm_grad_values:
                    all_zero = all(g == 0.0 for g in mm_grad_values)
                    if all_zero:
                        print(f"  ⚠️  WARNING: mm_projector gradients are ALL 0.0!")
                        print(f"     This suggests gradients are NOT flowing to mm_projector.")
                        print(f"     Likely causes:")
                        print(f"       1. mm_projector is not in gradient path")
                        print(f"       2. Output is detached somewhere")
                        print(f"       3. c10d::broadcast_ breaking autograd")
                    else:
                        print(f"  ✓ mm_projector gradients are flowing: {mm_grad_values[-1]}")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
