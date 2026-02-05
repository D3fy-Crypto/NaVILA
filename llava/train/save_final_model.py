"""
Save final combined GRU + VLA model after training completes.
"""

import os
import torch
from transformers import TrainerCallback


class SaveFinalModelCallback(TrainerCallback):
    """
    Callback to save the complete model (GRU + VLA with projector) after training.
    """
    
    def __init__(self, save_path):
        self.save_path = save_path
        
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """
        Called at the end of training.
        """
        if model is None:
            return
            
        print("\n" + "="*80)
        print("Training completed! Saving final combined model...")
        print("="*80)
        
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        
        # Save the complete model state
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': model.config,
        }
        
        # Save motion encoder specifically if it exists
        if hasattr(model, 'get_motion_encoder') and model.get_motion_encoder() is not None:
            motion_encoder = model.get_motion_encoder()
            save_dict['motion_encoder'] = {
                'gru': motion_encoder.gru.state_dict(),
                'projector': motion_encoder.grid_to_vision.state_dict(),
                'config': {
                    'gru_hidden_size': motion_encoder.gru.hidden_size,
                    'gru_num_layers': motion_encoder.gru.num_layers,
                    'gru_embedding_dim': motion_encoder.gru.embedding_dim,
                    'output_dim': motion_encoder.output_dim,
                }
            }
            print("✅ Motion encoder (GRU + Projector) included in checkpoint")
        
        # Save model
        model_path = os.path.join(self.save_path, "final_model.pt")
        torch.save(save_dict, model_path)
        print(f"✅ Saved complete model to: {model_path}")
        
        # Also save as safetensors if possible
        try:
            from safetensors.torch import save_file
            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            safetensors_path = os.path.join(self.save_path, "final_model.safetensors")
            save_file(state_dict, safetensors_path)
            print(f"✅ Saved safetensors to: {safetensors_path}")
        except Exception as e:
            print(f"⚠️  Could not save safetensors: {e}")
        
        # Save a summary
        summary_path = os.path.join(self.save_path, "model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Final Combined Model Summary\n")
            f.write("="*80 + "\n\n")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Frozen parameters: {total_params - trainable_params:,}\n\n")
            
            # Motion encoder info
            if hasattr(model, 'get_motion_encoder') and model.get_motion_encoder() is not None:
                motion_encoder = model.get_motion_encoder()
                gru_params = sum(p.numel() for p in motion_encoder.gru.parameters())
                proj_params = sum(p.numel() for p in motion_encoder.grid_to_vision.parameters())
                
                f.write("Motion Encoder Components:\n")
                f.write(f"  GRU: {gru_params:,} parameters (frozen)\n")
                f.write(f"  Grid-to-Vision Projector: {proj_params:,} parameters (trained)\n")
                f.write(f"  Total Motion Encoder: {gru_params + proj_params:,} parameters\n\n")
            
            f.write(f"\nModel saved to: {model_path}\n")
            
        print(f"✅ Saved model summary to: {summary_path}")
        print("="*80 + "\n")
