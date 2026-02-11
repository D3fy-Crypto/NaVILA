# How to Load & Evaluate the Trained GRU+NaVILA Model

## Quick Summary

Your trained model contains:
- **LLM**: Llama3-8B (language model) 
- **Vision Tower**: SigLIP (image encoder)
- **MM Projector**: Multimodal connector (trained)
- **Motion GRU**: Grid motion encoder (frozen)

## Model Checkpoints

### Location 1: Sanity Check Checkpoint (Recommended for quick testing)
```
/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check/
├── llm/                    # Llama3-8B weights
├── vision_tower/           # SigLIP weights  
├── mm_projector/           # Trained multimodal projector
└── trainer_state.json
```

### Location 2: Final Combined Model (For deployment)
```
/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/final_model_gru/
├── final_model.pt          # Complete model state dict
├── final_model.safetensors # SafeTensors format
└── model_summary.txt
```

---

## Loading the Model in Python

### Method 1: Load with Script (Easiest)
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA
python load_model_for_eval.py
```

This will:
- Load the full trained model with GRU
- Display model architecture
- Show parameter counts
- Verify all components (LLM, Vision, GRU)

### Method 2: Load in Your Code

```python
import torch
import sys
sys.path.insert(0, '/home/rithvik/NaVILA/llava')

from llava.model import LlavaLlamaForCausalLM
from safetensors.torch import load_file

# Option A: Load from safetensors (recommended)
checkpoint_path = '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check'

model = LlavaLlamaForCausalLM.from_pretrained(
    'a8cheng/navila-siglip-llama3-8b-v1.5-pretrain',
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)
model.eval()

# Option B: Load from checkpoint directory
if (Path(checkpoint_path) / 'llm').exists():
    # Load individual components
    from transformers import AutoModelForCausalLM
    
    llm = AutoModelForCausalLM.from_pretrained(
        f'{checkpoint_path}/llm',
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    )
    # ... load vision_tower, mm_projector similarly
```

### Method 3: Load with torch.load
```python
import torch

# Load the complete saved model
model = torch.load(
    '/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/final_model_gru/final_model.pt',
    map_location='cuda'
)
model.eval()
```

---

## Using with R2R Evaluation

### Run Evaluation Script (Recommended)
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA

# Run with default checkpoint
bash eval_gru_r2r.sh

# Or specify checkpoint and GPU
bash eval_gru_r2r.sh \
    /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check \
    1 \
    0 \
    "0,1"
```

### Manual R2R Evaluation
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation

CKPT_PATH="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check"
NUM_CHUNKS=1
CHUNK_START_IDX=0
GPU_IDS="0"

CUDA_VISIBLE_DEVICES=$GPU_IDS python run.py \
    --model_path "$CKPT_PATH" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_start_idx "$CHUNK_START_IDX" \
    --dataset_name r2r \
    --task VLNCE \
    --split_env eval_seen
```

---

## Model Architecture Verification

After loading, verify the model has all components:

```python
import torch

model = load_your_model()  # Load as shown above

# Check for GRU
assert hasattr(model, 'motion_encoder'), "Missing motion GRU!"
print("✅ Motion Encoder (GRU):", model.motion_encoder)

# Check for LLM  
assert hasattr(model, 'llm'), "Missing LLM!"
print("✅ LLM:", type(model.llm).__name__)

# Check for Vision Tower
assert hasattr(model, 'vision_tower'), "Missing Vision Tower!"
print("✅ Vision Tower:", type(model.vision_tower).__name__)

# Check for MM Projector
assert hasattr(model, 'mm_projector'), "Missing MM Projector!"
print("✅ MM Projector:", type(model.mm_projector).__name__)

# Print parameter counts
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal Parameters: {total:,}")
print(f"Trainable Parameters: {trainable:,}")
```

---

## Key Configuration for R2R Evaluation

The model was trained with:
- **Vision Tower**: `google/siglip-so400m-patch14-384`
- **Language Model**: `Llama-3-8B`
- **MM Projector**: `mlp_downsample`
- **Motion GRU**: Motion encoder (frozen, not updated)
- **Batch Size**: 1 (for inference)
- **Precision**: bfloat16

Make sure your evaluation script uses compatible settings:
```python
# In your eval config:
vision_tower = "google/siglip-so400m-patch14-384"
mm_projector = "mlp_downsample"
torch_dtype = torch.bfloat16
```

---

## Troubleshooting

### "Missing motion_encoder"
- The model needs GRU integrated. Load from checkpoint, not base model.
- Use: `checkpoints/navila-8b-8f-gru-sanity-check/` (has trained weights)

### "Shape mismatch" errors
- Verify checkpoint path is correct
- Check that you're loading from a saved checkpoint, not the model hub

### CUDA OOM
- Use `device_map='cpu'` for smaller GPUs
- Load with `torch.bfloat16` (half precision)

### Model not in eval mode
- Always call `model.eval()` before inference
- Disables dropout, batch norm, etc.

---

## Files Created

- `load_model_for_eval.py` - Model loading script
- `eval_gru_r2r.sh` - R2R evaluation script
- This guide

---

## Next Steps

1. **Test Model Loading**:
   ```bash
   python load_model_for_eval.py
   ```

2. **Run R2R Evaluation**:
   ```bash
   cd evaluation && bash ../eval_gru_r2r.sh
   ```

3. **Check Results**:
   ```bash
   ls /path/to/checkpoint/r2r_eval_results/
   ```

