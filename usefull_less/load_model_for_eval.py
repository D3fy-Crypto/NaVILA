#!/usr/bin/env python3
"""
Load the trained GRU+NaVILA model for evaluation.
This shows how to load:
  - LlamaForCausalLM (LLM)
  - SigLIP Vision Tower
  - Multimodal Projector
  - Motion GRU Encoder
"""

import torch
import sys
from pathlib import Path

# Add NaVILA to path
sys.path.insert(0, '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA')

from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaModel


def load_gru_navila_model(checkpoint_path=None, device='cuda'):
    """
    Load the trained GRU+NaVILA model for evaluation.
    
    Args:
        checkpoint_path: Path to checkpoint directory. If None, uses final_model_gru
        device: Device to load on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model with GRU, LLM, and vision tower
        processor: Image processor (if available)
    """
    
    if checkpoint_path is None:
        # Default to the trained sanity check checkpoint
        checkpoint_path = '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check'
    
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Device: {device}")
    print("="*80)
    
    # Method 1: Load from saved checkpoint (recommended for evaluation)
    if (checkpoint_path / 'llm').exists():
        print("\n[Method 1] Loading from checkpoint directory...")
        print("  - Loading LLM...")
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(
            checkpoint_path / 'llm',
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print("  - Loading Vision Tower...")
        from transformers import AutoModel
        vision_tower = AutoModel.from_pretrained(
            checkpoint_path / 'vision_tower',
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print("  - Loading MM Projector...")
        mm_projector = torch.load(
            checkpoint_path / 'mm_projector' / 'pytorch_model.bin',
            map_location=device
        )
        
        print("  ✅ Components loaded")
        
    else:
        # Method 2: Load from model hub (for base model)
        print("\n[Method 2] Loading from model hub...")
        model_path = 'a8cheng/navila-siglip-llama3-8b-v1.5-pretrain'
        
        try:
            model = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name='navila',
                device=device
            )
            print(f"  ✅ Model loaded from {model_path}")
            return model, None
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return None, None
    
    return None, None


def load_with_gru_integration(checkpoint_path=None, device='cuda'):
    """
    Load the COMPLETE trained model (LLM + Vision + GRU).
    
    This is the full model as saved after training.
    """
    
    if checkpoint_path is None:
        checkpoint_path = '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check'
    
    checkpoint_path = Path(checkpoint_path)
    
    # Method 1: Load from checkpoint directory (has config.json)
    if (checkpoint_path / 'config.json').exists():
        print(f"\n[CHECKPOINT] Loading from: {checkpoint_path}")
        
        from llava.model.builder import load_pretrained_model
        
        print("  - Loading model from checkpoint...")
        try:
            result = load_pretrained_model(
                model_path=str(checkpoint_path),
                model_base=None,
                model_name='navila',
                device=device
            )
            
            # Handle tuple return (tokenizer, model, image_processor, context_len)
            if isinstance(result, tuple):
                model = result[1] if len(result) > 1 else result[0]
            else:
                model = result
            
            model.eval()
            print(f"  ✅ Model loaded from checkpoint")
            return model
        except Exception as e:
            print(f"  ⚠️  load_pretrained_model failed: {e}")
            print(f"     Attempting fallback method...")
    
    # Method 2: Load safetensors directly
    if (checkpoint_path / 'final_model.safetensors').exists():
        print(f"\n[SAFETENSORS] Loading from: {checkpoint_path / 'final_model.safetensors'}")
        
        from safetensors.torch import load_file
        print("  - Loading weights...")
        state_dict = load_file(checkpoint_path / 'final_model.safetensors')
        
        # Initialize base model
        print("  - Initializing model architecture...")
        from llava.model.builder import load_pretrained_model
        
        # Load base model first
        model = load_pretrained_model(
            model_path='a8cheng/navila-siglip-llama3-8b-v1.5-pretrain',
            model_base=None,
            model_name='navila',
            device=device
        )
        
        # Load trained weights
        print("  - Loading trained weights...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  - Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        model.eval()
        
        print(f"  ✅ Model loaded with trained weights")
        return model
    
    # Method 3: Load torch checkpoint
    elif (checkpoint_path / 'final_model.pt').exists():
        print(f"\n[TORCH] Loading from: {checkpoint_path / 'final_model.pt'}")
        model = torch.load(checkpoint_path / 'final_model.pt', map_location=device)
        model.to(device)
        model.eval()
        print(f"  ✅ Model loaded")
        return model
    
    else:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print(f"   Expected one of:")
        print(f"     - config.json (for HF checkpoint)")
        print(f"     - final_model.safetensors")
        print(f"     - final_model.pt")
        return None


def load_for_evaluation(use_full_model=True):
    """
    Simple wrapper to load model for evaluation.
    """
    print("\n" + "="*80)
    print("LOADING TRAINED GRU+NAVILA MODEL")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if use_full_model:
        # Load the complete trained model
        model = load_with_gru_integration(device=device)
        
        if model:
            print("\n" + "="*80)
            print("MODEL ARCHITECTURE")
            print("="*80)
            print(model)
            
            # Check what's in the model
            print("\n" + "="*80)
            print("MODEL COMPONENTS")
            print("="*80)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            
            # Check for GRU
            if hasattr(model, 'motion_encoder'):
                print(f"\n✅ Motion Encoder (GRU) found!")
                if model.motion_encoder is not None:
                    print(f"   GRU parameters: {sum(p.numel() for p in model.motion_encoder.gru.parameters()):,}")
                    print(f"   Projector parameters: {sum(p.numel() for p in model.motion_encoder.projector.parameters()):,}")
                else:
                    print(f"   ⚠️  Motion encoder attribute exists but is None")
                    print(f"   Note: GRU weights need to be loaded separately for inference")
            
            if hasattr(model, 'llm'):
                print(f"\n✅ LLM found!")
                llm_params = sum(p.numel() for p in model.llm.parameters())
                print(f"   LLM parameters: {llm_params:,}")
            
            if hasattr(model, 'vision_tower'):
                print(f"\n✅ Vision Tower (SigLIP) found!")
                vt_params = sum(p.numel() for p in model.vision_tower.parameters()) if model.vision_tower else 0
                print(f"   Vision Tower parameters: {vt_params:,}")
            
            if hasattr(model, 'mm_projector'):
                print(f"\n✅ MM Projector found!")
                mm_params = sum(p.numel() for p in model.mm_projector.parameters())
                print(f"   MM Projector parameters: {mm_params:,}")
            
            return model
    else:
        # Load base model without GRU
        model, _ = load_gru_navila_model(device=device)
        return model


# Usage Example
if __name__ == '__main__':
    model = load_for_evaluation(use_full_model=True)
    
    if model:
        print("\n" + "="*80)
        print("✅ MODEL READY FOR EVALUATION")
        print("="*80)
        print("\nYou can now use this model for:")
        print("  1. R2R evaluation (navigation)")
        print("  2. VLNCE benchmark")
        print("  3. Custom inference")
        print("\nModel is on device:", next(model.parameters()).device)
