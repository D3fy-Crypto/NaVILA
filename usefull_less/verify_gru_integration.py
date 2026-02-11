#!/usr/bin/env python3
"""
Verification script for GRU integration.
Tests all components before training.
"""

import os
import sys
import json
import torch
import numpy as np

print("="*80)
print("GRU INTEGRATION VERIFICATION SCRIPT")
print("="*80)

# Test 1: Check checkpoint files exist
print("\n[1/7] Checking checkpoint files...")
gru_ckpt = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"
oracle_deltas = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/oracle_exports/oracle_deltas_train.jsonl"

if os.path.exists(gru_ckpt):
    print(f"✅ GRU checkpoint found: {gru_ckpt}")
    ckpt = torch.load(gru_ckpt, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        print(f"   Checkpoint keys: {list(ckpt.keys())}")
        if 'state_dict' in ckpt:
            print(f"   Model keys: {len(ckpt['state_dict'])} parameters")
else:
    print(f"❌ GRU checkpoint NOT found: {gru_ckpt}")
    sys.exit(1)

if os.path.exists(oracle_deltas):
    print(f"✅ Oracle deltas found: {oracle_deltas}")
    with open(oracle_deltas) as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Episode ID: {sample.get('episode_id', 'N/A')}")
        print(f"   Num deltas: {len(sample.get('deltas', []))}")
else:
    print(f"❌ Oracle deltas NOT found: {oracle_deltas}")
    sys.exit(1)

# Test 2: Import modules
print("\n[2/7] Testing imports...")
try:
    from llava.model.grid_rnn import MotionGRU, GridToVisionProjector, MotionEncoderWithProjector
    print("✅ grid_rnn module imports successfully")
except Exception as e:
    print(f"❌ Failed to import grid_rnn: {e}")
    sys.exit(1)

try:
    from llava.train.args import DataArguments, ModelArguments, TrainingArguments
    print("✅ args module imports successfully")
except Exception as e:
    print(f"❌ Failed to import args: {e}")
    sys.exit(1)

# Test 3: Test MotionGRU instantiation
print("\n[3/7] Testing MotionGRU instantiation...")
try:
    gru = MotionGRU(input_size=4, hidden_size=256, num_layers=2, embedding_dim=128)
    print(f"✅ MotionGRU created successfully")
    print(f"   Parameters: {sum(p.numel() for p in gru.parameters()):,}")
    
    # Test forward pass
    test_input = torch.randn(2, 10, 4)
    output = gru(test_input)
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Test output shape: {output.shape}")
    assert output.shape == (2, 128), "Output shape mismatch!"
    print("✅ MotionGRU forward pass works")
except Exception as e:
    print(f"❌ MotionGRU test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test GridToVisionProjector
print("\n[4/7] Testing GridToVisionProjector...")
try:
    projector = GridToVisionProjector(embedding_dim=128, output_dim=4096)
    print(f"✅ GridToVisionProjector created successfully")
    print(f"   Parameters: {sum(p.numel() for p in projector.parameters()):,}")
    
    # Test forward pass
    test_input = torch.randn(2, 128)
    output = projector(test_input)
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Test output shape: {output.shape}")
    assert output.shape == (2, 4096), "Output shape mismatch!"
    print("✅ GridToVisionProjector forward pass works")
except Exception as e:
    print(f"❌ GridToVisionProjector test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test MotionEncoderWithProjector
print("\n[5/7] Testing MotionEncoderWithProjector...")
try:
    encoder = MotionEncoderWithProjector(
        gru_ckpt_path=gru_ckpt,
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_embedding_dim=128,
        output_dim=4096,
        freeze_gru=True
    )
    print(f"✅ MotionEncoderWithProjector created successfully")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Test forward pass
    test_input = torch.randn(2, 10, 4)
    output = encoder(test_input)
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Test output shape: {output.shape}")
    assert output.shape == (2, 4096), "Output shape mismatch!"
    print("✅ MotionEncoderWithProjector forward pass works")
except Exception as e:
    print(f"❌ MotionEncoderWithProjector test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test pose delta normalization
print("\n[6/7] Testing pose delta normalization...")
try:
    # Sample delta
    dx, dy, dyaw = 0.1, -0.05, 0.785  # 45 degrees
    dx_norm = dx / 0.25
    dy_norm = dy / 0.25
    dyaw_sin = np.sin(dyaw)
    dyaw_cos = np.cos(dyaw)
    
    print(f"   Original: dx={dx}, dy={dy}, dyaw={dyaw}")
    print(f"   Normalized: dx={dx_norm:.3f}, dy={dy_norm:.3f}, sin={dyaw_sin:.3f}, cos={dyaw_cos:.3f}")
    
    # Test L2 norm after GRU (use float32)
    test_deltas = torch.tensor([[dx_norm, dy_norm, dyaw_sin, dyaw_cos]], dtype=torch.float32).unsqueeze(0).repeat(1, 10, 1)
    gru_output = gru(test_deltas)
    l2_norm = torch.norm(gru_output[0])
    print(f"   GRU output L2 norm: {l2_norm:.6f}")
    assert abs(l2_norm.item() - 1.0) < 0.01, f"L2 norm should be ~1.0, got {l2_norm:.6f}"
    print("✅ Pose delta normalization works correctly")
except Exception as e:
    print(f"❌ Pose delta normalization test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test gradient flow
print("\n[7/7] Testing gradient flow...")
try:
    encoder = MotionEncoderWithProjector(
        gru_ckpt_path=None,  # Don't load checkpoint for gradient test
        output_dim=4096,
        freeze_gru=True
    )
    
    # Forward pass
    test_input = torch.randn(2, 10, 4, requires_grad=True)
    output = encoder(test_input)
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    gru_has_grad = any(p.grad is not None for p in encoder.gru.parameters())
    projector_has_grad = any(p.grad is not None for p in encoder.projector.parameters())
    
    print(f"   GRU has gradients: {gru_has_grad} (should be False)")
    print(f"   Projector has gradients: {projector_has_grad} (should be True)")
    
    assert not gru_has_grad, "GRU should not have gradients (frozen)!"
    assert projector_has_grad, "Projector should have gradients (trainable)!"
    print("✅ Gradient flow is correct (GRU frozen, Projector trainable)")
except Exception as e:
    print(f"❌ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("="*80)
print("\nYou can now proceed with training:")
print("  bash scripts/train/sft_8frames_gru.sh")
print("="*80)
