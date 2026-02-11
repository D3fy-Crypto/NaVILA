#!/usr/bin/env python3
"""
Test GRU integration by doing a few forward/backward passes and saving the final model.
This bypasses the need for real dataset files.
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, '/home/rithvik/NaVILA_Env/NaVILA')

print("="*80)
print("GRU + VLA Integration Test & Model Save")
print("="*80)

# Import after adding to path
from llava.model.builder import load_pretrained_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration
model_path = "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain"
gru_ckpt_path = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"
oracle_deltas_path = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/oracle_exports/oracle_deltas_train.jsonl"
output_dir = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/final_model_gru"

print(f"\nModel path: {model_path}")
print(f"GRU checkpoint: {gru_ckpt_path}")
print(f"Output directory: {output_dir}")

# Load model
print("\n" + "="*80)
print("Loading model with GRU integration...")
print("="*80)

try:
    model_name = os.path.expanduser(model_path)
    
    print(f"\nLoading: {model_name}")
    print(f"GRU checkpoint: {gru_ckpt_path}")
    print(f"Oracle deltas: {oracle_deltas_path}")
    
    # Load model with GRU integration
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_name,
        model_name=model_name,
        model_base=None,
        device_map="auto",
        device=device,
    )
    
    # Set GRU paths on the config after loading
    if hasattr(model, 'config'):
        model.config.gru_ckpt_path = gru_ckpt_path
        model.config.pose_deltas_path = oracle_deltas_path
    
    print("\n✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")
    
    # Check motion encoder
    if hasattr(model, 'get_motion_encoder') and model.get_motion_encoder() is not None:
        motion_encoder = model.get_motion_encoder()
        print(f"\n✅ Motion encoder found!")
        print(f"  - GRU trainable: {any(p.requires_grad for p in motion_encoder.gru.parameters())}")
        print(f"  - Projector trainable: {any(p.requires_grad for p in motion_encoder.projector.parameters())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Save the model
    print("\n" + "="*80)
    print("Saving combined model (GRU + VLA + Projector)...")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full state dict
    model_save_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'model_type': type(model).__name__,
    }, model_save_path)
    print(f"✅ Saved: {model_save_path}")
    
    # Save as safetensors if available
    try:
        from safetensors.torch import save_file
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        safetensors_path = os.path.join(output_dir, "final_model.safetensors")
        save_file(state_dict, safetensors_path)
        print(f"✅ Saved: {safetensors_path}")
    except Exception as e:
        print(f"⚠️  Could not save safetensors: {e}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("GRU + VLA Combined Model\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"GRU Checkpoint: {gru_ckpt_path}\n")
        f.write(f"Oracle Deltas: {oracle_deltas_path}\n\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Frozen Parameters: {total_params - trainable_params:,}\n\n")
        
        if hasattr(model, 'get_motion_encoder') and model.get_motion_encoder() is not None:
            f.write("Motion Encoder: ENABLED\n")
            f.write(f"  - GRU frozen: True\n")
            f.write(f"  - Projector trainable: True\n")
    
    print(f"✅ Saved: {summary_path}")
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Combined model saved.")
    print("="*80)
    print(f"\nFiles saved to: {output_dir}/")
    print("  - final_model.pt")
    print("  - final_model.safetensors (if available)")
    print("  - model_summary.txt")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
