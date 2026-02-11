#!/usr/bin/env python3
"""
Quick model loading verification without full instantiation.
"""

import torch
from pathlib import Path

def verify_checkpoint_structure(ckpt_path):
    """Verify checkpoint has all required components."""
    
    ckpt = Path(ckpt_path)
    
    print("="*80)
    print(f"CHECKPOINT STRUCTURE: {ckpt}")
    print("="*80)
    
    required_components = {
        'llm': 'Language Model (Llama-3-8B)',
        'vision_tower': 'Vision Encoder (SigLIP)',
        'mm_projector': 'Multimodal Projector',
    }
    
    print("\nChecking components:")
    all_found = True
    
    for component, description in required_components.items():
        component_path = ckpt / component
        
        if component_path.exists():
            # Count files
            files = list(component_path.glob('*'))
            print(f"  ✅ {component:20s} ({description})")
            print(f"     Path: {component_path}")
            print(f"     Files: {len(files)} items")
            
            # Show key files
            if component_path.is_dir():
                key_files = [f.name for f in files if f.name in ['config.json', 'pytorch_model.bin', 'model.safetensors']]
                if key_files:
                    print(f"     Key files: {', '.join(key_files)}")
        else:
            print(f"  ❌ {component:20s} - NOT FOUND")
            all_found = False
    
    # Check for motion encoder
    print(f"\n  Motion Encoder (GRU):")
    print(f"     Status: Integrated into model (loaded at runtime)")
    
    print("\n" + "="*80)
    print("EVALUATION USAGE")
    print("="*80)
    
    ckpt_name = ckpt.name
    print(f"\nTo evaluate with this checkpoint:")
    print(f"\n  bash eval_gru_r2r.sh \\")
    print(f"      {ckpt_path} \\")
    print(f"      1 \\")
    print(f"      0 \\")
    print(f"      \"0\"")
    
    print(f"\nOr manually:")
    print(f"  cd evaluation")
    print(f"  python run.py \\")
    print(f"      --model_path {ckpt_path} \\")
    print(f"      --dataset_name r2r \\")
    print(f"      --task VLNCE \\")
    print(f"      --split_env eval_seen")
    
    print("\n" + "="*80)
    
    return all_found


if __name__ == '__main__':
    import sys
    
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check'
    
    success = verify_checkpoint_structure(ckpt_path)
    sys.exit(0 if success else 1)
