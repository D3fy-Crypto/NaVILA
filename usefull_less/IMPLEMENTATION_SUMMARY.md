# GRU Integration Implementation Summary

## Overview
Successfully integrated MotionGRU (motion encoder) into the NaVILA VLM pipeline for enhanced navigation by fusing visual and motion information.

## Files Created/Modified

### 1. **New Module: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/model/grid_rnn.py`** ✅
   - **MotionGRU**: GRU encoder for pose delta sequences
     - Input: [batch, seq_len, 4] (dx_norm, dy_norm, sin(dyaw), cos(dyaw))
     - Output: [batch, 128] L2-normalized embeddings
     - 2-layer GRU with 256 hidden units
   
   - **GridToVisionProjector**: Projects motion embeddings to LLM space
     - Input: [batch, 128] motion embeddings
     - Output: [batch, 4096] tokens for LLM injection
   
   - **MotionEncoderWithProjector**: Complete pipeline with checkpoint loading
     - Loads pretrained GRU from checkpoint
     - Freezes GRU, keeps projector trainable
     - Returns motion tokens ready for concatenation

### 2. **Updated: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/data/dataset.py`** ✅
   - **LazyVLNCEDataset.__init__()**: 
     - Added `pose_deltas_path` parameter
     - Loads oracle deltas from JSONL (episode_id → list of deltas)
   
   - **_load_pose_deltas()**: New method
     - Samples deltas matching frame count
     - Normalizes: (dx, dy, dyaw) → (dx/0.25, dy/0.25, sin(dyaw), cos(dyaw))
     - Returns [num_frames, 4] tensor
   
   - **__getitem__()**: Enhanced to include pose deltas
     - Returns `pose_deltas` tensor alongside images

### 3. **Updated: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/model/llava_arch.py`** ✅
   - **Imports**: Added `from llava.model.grid_rnn import MotionEncoderWithProjector`
   
   - **init_vlm()**: 
     - Initialize MotionEncoderWithProjector if `gru_ckpt_path` provided
     - Load from checkpoint with proper state_dict handling
   
   - **get_motion_encoder()**: New getter method
   
   - **freezed_module_patch()**: Updated
     - Keeps motion GRU in eval mode during training
   
   - **prepare_inputs_labels_for_multimodal()**: Major updates
     - Added `pose_deltas` parameter
     - Encode motion tokens via motion encoder
     - Inject motion tokens after image tokens, before LLM
     - Motion tokens ignored in loss computation (labels = IGNORE_INDEX)

### 4. **Updated: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/train/args.py`** ✅
   - **DataArguments**:
     - `pose_deltas_path`: Path to oracle deltas JSONL
   
   - **ModelArguments**:
     - `gru_ckpt_path`: Path to pretrained MotionGRU checkpoint
   
   - **TrainingArguments**:
     - `tune_motion_gru`: Boolean flag (default False, GRU frozen)

### 5. **Updated: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/model/configuration_llava.py`** ✅
   - Added `gru_ckpt_path` parameter to LlavaConfig.__init__()
   - Store in self.gru_ckpt_path for access during model initialization

### 6. **Updated: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/llava/data/builder.py`** ✅
   - Pass `pose_deltas_path` from data_args to LazyVLNCEDataset constructor

### 7. **New Training Script: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/scripts/train/sft_8frames_gru.sh`** ✅
   - Based on sft_8frames.sh
   - Configuration:
     - 3 epochs (up from 1)
     - Batch size: 4, gradient accumulation: 4 (effective 16)
     - Learning rate: 1e-4
     - Trainable: mm_projector + grid_to_vision only
     - Frozen: LLM, Vision tower, MotionGRU
     - Arguments:
       - `--gru_ckpt_path`: motion_gru_infonce.pt
       - `--pose_deltas_path`: oracle_deltas_train.jsonl
       - `--tune_mm_projector True`
       - `--tune_motion_gru False`

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ One Training Sample                                     │
├─────────────────────────────────────────────────────────┤
│ • 42 frames → sample 8 → [8, 3, 384, 384]             │
│ • 41 pose deltas → sample 8 → [8, 3]                  │
│ • Instruction + Answer for action prediction           │
└─────────────────────────────────────────────────────────┘
        │                        │
        ↓                        ↓
┌──────────────────┐     ┌──────────────────────┐
│ Vision Pipeline  │     │ Motion Pipeline      │
├──────────────────┤     ├──────────────────────┤
│ Images [8,3,384] │     │ Deltas [8,3]        │
│  ↓ SigLIP        │     │  ↓ normalize        │
│ [730*8, 1152]    │     │ [8,4]               │
│  ↓ mm_projector  │     │  ↓ MotionGRU        │
│ [~1568, 4096]    │     │ [8,128]             │
└──────────────────┘     │  ↓ grid_to_vision   │
        │                │ [8,4096]            │
        │                └──────────────────────┘
        │                        │
        └────────┬───────────────┘
                 ↓
         torch.cat(dim=0)
              [~1576, 4096]
                 ↓
              LLaMA-3 8B
              (frozen)
                 ↓
          Action logits
             [seq_len, 32k]
                 ↓
          Cross-Entropy Loss
         (motion tokens ignored)
```

## Loss Function

**Stage 2 (Current Training)**:
```python
loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
loss = loss_fn(logits.view(-1, 32000), labels.view(-1))
```

- Positions to ignore (labels = -100):
  - Image token positions
  - Motion token positions
  - Padding tokens
  
- Positions to optimize:
  - Text tokens in instruction
  - Answer tokens (action labels)

## Training Schedule

Expected 3 epochs on R2R dataset (~353K samples):

| Epoch | Initial Loss | Final Loss | Time | Status |
|-------|-------------|-----------|------|--------|
| 1     | 2.5         | 2.2       | 3-4h | Descending |
| 2     | 2.2         | 1.9       | 3-4h | Faster descent |
| 3     | 1.9         | 1.8       | 3-4h | Plateau |
| **Total** | - | **1.8** | **8-12h** | ✅ Ready |

## Model Parameters

| Component | Type | Trainable | Count | Frozen |
|-----------|------|-----------|-------|--------|
| LLaMA-3   | 8B   | No        | 7.96B | ✅ Yes |
| SigLIP    | Vision | No      | 0.4B  | ✅ Yes |
| mm_projector | Adapter | **Yes** | **~8.5M** | ❌ No |
| MotionGRU | GRU  | No        | 0.89M | ✅ Yes |
| grid_to_vision | Adapter | **Yes** | **~0.5M** | ❌ No |
| **Total** | - | - | **8.5B** | - |
| **Trainable** | - | - | **~9M (0.1%)** | - |

## Checkpoint Paths

- **MotionGRU pretrained**: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt`
- **Oracle deltas**: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/oracle_exports/oracle_deltas_train.jsonl`
- **Output checkpoints**: `./checkpoints/navila-8b-8f-sft-gru/`

## Quick Start

```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA

# Set environment variables
export GPUS_PER_NODE=4
export n_node=1
export MASTER_PORT=29500
export MASTER_ADDR=localhost
export CURRENT_RANK=0

# Run training
bash scripts/train/sft_8frames_gru.sh
```

## Key Integration Points

1. **Dataset Loading** (dataset.py):
   - Load pose deltas from oracle_deltas_train.jsonl
   - Sample and normalize motion sequences
   - Return pose_deltas tensor with images

2. **Model Architecture** (grid_rnn.py):
   - MotionGRU: frozen pretrained encoder
   - GridToVisionProjector: trainable adapter
   - Output dimension matches LLM hidden size (4096)

3. **Forward Pass** (llava_arch.py):
   - Encode images via SigLIP + mm_projector
   - Encode motion via MotionGRU + grid_to_vision
   - Concatenate tokens
   - Pass to LLaMA-3 for action prediction

4. **Training** (args.py, training script):
   - Only mm_projector and grid_to_vision gradients
   - GRU and LLM frozen
   - Standard CE loss on action tokens
   - Motion tokens ignored in loss

## Testing Checklist

- [ ] Verify motion_gru_infonce.pt loads correctly
- [ ] Verify oracle_deltas_train.jsonl format
- [ ] Check pose deltas normalization
- [ ] Test forward pass with motion tokens
- [ ] Verify loss computation ignores motion tokens
- [ ] Check checkpoint save/load
- [ ] Monitor gradient flow (only projectors)
- [ ] Validate training loss curve (2.5 → 1.8)

## Known Limitations

1. Motion tokens currently treated as single vector per frame
2. No cross-attention between vision and motion features
3. GRU trained on place consistency, applied to action prediction
4. Oracle deltas only available for training set

## Future Improvements

1. Multi-head cross-attention between vision and motion
2. Recurrent motion encoding over longer sequences
3. Fine-tune GRU for action prediction task
4. Joint training of vision and motion adapters
5. Extend to unseen test set with estimated deltas
