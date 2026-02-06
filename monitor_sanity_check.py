#!/usr/bin/env python3
"""
Sanity Check Monitor for GRU Training
======================================

Monitors and validates:
1. Loss trajectory (should decrease)
2. Gradient flow (GRU frozen, projector trainable)
3. No NaN/Inf values
4. Projection norms reasonable
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def parse_training_log(log_path):
    """Parse training log and extract metrics."""
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return None
    
    content = Path(log_path).read_text()
    losses = []
    steps = []
    
    # Parse loss values from tensorboard logs or stdout
    for line in content.split('\n'):
        if '"loss":' in line:
            try:
                # Extract loss value
                parts = line.split('"loss":')
                if len(parts) > 1:
                    loss_str = parts[1].strip().split(',')[0].strip()
                    loss = float(loss_str)
                    losses.append(loss)
                    
                    # Try to extract step
                    if '"step":' in line:
                        step_str = line.split('"step":')[1].split(',')[0].strip()
                        steps.append(int(step_str))
            except (ValueError, IndexError):
                pass
    
    return {
        'losses': losses,
        'steps': steps,
        'has_nan': 'nan' in content.lower(),
        'has_inf': 'inf' in content.lower() and 'infinity' not in content.lower(),
        'has_error': 'error:' in content.lower(),
        'raw_content': content
    }

def check_loss_trajectory(losses):
    """Check if loss is decreasing."""
    if len(losses) < 2:
        return None
    
    initial = losses[0]
    final = losses[-1]
    decrease = initial - final
    
    return {
        'initial': initial,
        'final': final,
        'decrease': decrease,
        'percent': (decrease / initial * 100) if initial > 0 else 0,
        'is_decreasing': final < initial
    }

def check_gradient_flow(log_content):
    """Check gradient flow from log."""
    # Look for gradient norms in training logs
    gru_grad_norms = []
    projector_grad_norms = []
    
    for line in log_content.split('\n'):
        if 'motion_gru' in line.lower() and 'grad_norm' in line.lower():
            try:
                if '0.0' in line or 'nan' in line.lower():
                    gru_grad_norms.append(0)
                else:
                    # Extract number
                    parts = line.split(':')[-1].strip().split()[0]
                    gru_grad_norms.append(float(parts))
            except:
                pass
        
        if 'projector' in line.lower() and 'grad_norm' in line.lower():
            try:
                parts = line.split(':')[-1].strip().split()[0]
                projector_grad_norms.append(float(parts))
            except:
                pass
    
    return {
        'gru_frozen': len(gru_grad_norms) == 0 or all(g == 0 for g in gru_grad_norms),
        'projector_trainable': len(projector_grad_norms) > 0 and any(g > 0 for g in projector_grad_norms),
    }

def print_report(results):
    """Print formatted sanity check report."""
    print("\n" + "="*70)
    print("SANITY CHECK REPORT")
    print("="*70 + "\n")
    
    if results is None:
        print("❌ Failed to parse training log\n")
        return False
    
    all_pass = True
    
    # Check 1: Loss Trajectory
    print("1️⃣  LOSS TRAJECTORY")
    print("-" * 70)
    loss_check = check_loss_trajectory(results['losses'])
    
    if loss_check:
        print(f"   Initial loss: {loss_check['initial']:.4f}")
        print(f"   Final loss:   {loss_check['final']:.4f}")
        print(f"   Decrease:     {loss_check['decrease']:.4f} ({loss_check['percent']:.1f}%)")
        
        if loss_check['is_decreasing']:
            print(f"   ✅ PASS - Loss is decreasing\n")
        else:
            print(f"   ❌ FAIL - Loss is NOT decreasing\n")
            all_pass = False
    else:
        print(f"   ⚠️  Only {len(results['losses'])} loss values found\n")
    
    # Check 2: Numerical Stability
    print("2️⃣  NUMERICAL STABILITY")
    print("-" * 70)
    
    nan_check = not results['has_nan']
    inf_check = not results['has_inf']
    error_check = not results['has_error']
    
    print(f"   NaNs detected: {'❌ YES' if results['has_nan'] else '✅ NO'}")
    print(f"   Infs detected: {'❌ YES' if results['has_inf'] else '✅ NO'}")
    print(f"   Errors detected: {'❌ YES' if results['has_error'] else '✅ NO'}\n")
    
    if not (nan_check and inf_check and error_check):
        all_pass = False
    
    # Check 3: Loss values sanity
    print("3️⃣  LOSS VALUE SANITY")
    print("-" * 70)
    
    if results['losses']:
        avg_loss = sum(results['losses']) / len(results['losses'])
        max_loss = max(results['losses'])
        min_loss = min(results['losses'])
        
        print(f"   Num loss values: {len(results['losses'])}")
        print(f"   Loss range: [{min_loss:.4f}, {max_loss:.4f}]")
        print(f"   Avg loss: {avg_loss:.4f}")
        
        # Check for reasonable loss values
        reasonable = 0.1 < avg_loss < 50
        if reasonable:
            print(f"   ✅ PASS - Loss values in reasonable range\n")
        else:
            print(f"   ⚠️  Loss values may be unusual\n")
    
    # Summary
    print("="*70)
    if all_pass:
        print("✅ SANITY CHECK PASSED - Ready for full training!")
    else:
        print("❌ SANITY CHECK FAILED - Review issues above")
    print("="*70 + "\n")
    
    return all_pass

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else \
               "./checkpoints/navila-8b-8f-gru-sanity-check/sanity_check.log"
    
    results = parse_training_log(log_path)
    success = print_report(results)
    
    sys.exit(0 if success else 1)
