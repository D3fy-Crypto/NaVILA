# Summary: GRU Integration into NaVILA VLM Pipeline

## 1. What Is the GRU Input?

```python
# GRU Input: Motion sequence (10 steps)
input: [batch, window_size=10, 3]
       where 3 = (dx, dy, dyaw)

# Example from one trajectory:
[
  (0.022, 0.036, 0.056),    # motion from step 0→1
  (0.012, -0.015, 0.047),   # motion from step 1→2
  ...
  (0.018, 0.008, -0.023),   # motion from step 9→10
]

# Normalized format for network:
[batch, 10, 4] where 4 = (dx/0.25, dy/0.25, sin(dyaw), cos(dyaw))
```

## 2. What Is GRU Training?

**Stage 1: InfoNCE Pretraining (Done in Notebook)**

```python
# Objective: Learn place-consistent embeddings
loss = InfoNCELoss(anchor_embed, positive_embed, negative_embeds)

# Where:
# • anchor_embed:     motion sequence A encoded
# • positive_embed:   motion sequence B (same place, different path)
# • negative_embeds:  motion sequences (different places)

# Result:
# - Same places → embeddings are similar (cos_sim ≈ 0.8)
# - Different places → embeddings are orthogonal (cos_sim ≈ 0)
# - GRU learns to encode "place identity" from egomotion patterns
```

## 3. How to Project GRU into the Pipeline

**The integration happens in 3 steps:**

```
Step 1: Load pretrained GRU
  └─ checkpoint: motion_gru_infonce.pt
  └─ status: FROZEN (no gradients)

Step 2: Add Grid-to-Vision projector (NEW)
  └─ Input:  [batch, 128] place embeddings
  └─ Layers: Linear(128→256) + GELU + Linear(256→4096)
  └─ Output: [batch, 4096] motion tokens
  └─ status: TRAINABLE

Step 3: Inject into VLM forward pass
  └─ location: after vision tokens, before LLM
  └─ operation: concatenate vision_tokens + motion_tokens
```

## 4. Where to Inject in the Pipeline

```
                    8 RGB Frames
                   [8, 3, 384, 384]
                          │
             ┌────────────┼────────────┐
             │            │            │
        FROZEN         FROZEN       (NEW) Motion Path
        Vision          GRU              │
             │            │             ↓
             ├─ SigLIP ────┤─────┐  motion[8,4]
             │ [~1568,4096] │    │      │
             │            │    │   MotionGRU
             │            │    │   (frozen)
             │            │    │      │
             │            │    │   embed[8,128]
             │            │    │      │
             │            │    │  Grid2Vision
             │            │    │  (trainable)
             │            │    │      │
             └────────┬────┴────┴─→ motion[8,4096]
                      │      │
                  vision+motion
                   fused tokens
                   [~1576, 4096]
                      │
                      ↓
                  LLaMA-3 8B
                  (frozen)
                      │
                      ↓
                   Logits →
              "move forward 25cm"
```

## 5. Complete Input for One Prediction

```python
# Dataset returns ONE sample:
sample = {
    'image': [8, 3, 384, 384],        # 8 sampled frames
    'pose_deltas': [8, 3],             # 8 corresponding deltas
    'input_ids': [seq_len],            # tokenized instruction
    'labels': [seq_len],               # gold action tokens
}

# Batch of 4 samples:
batch = {
    'image': [4, 8, 3, 384, 384],
    'pose_deltas': [4, 8, 3],
    'input_ids': [4, seq_len],
    'labels': [4, seq_len],
}

# Data flow during forward pass:
image → SigLIP → mm_projector → [~1568, 4096] vision_tokens
pose_deltas → MotionGRU → grid_to_vision → [8, 4096] motion_tokens
                                                 │
                                                 ↓
                                        concatenate + LLM
                                                 │
                                                 ↓
                                        logits[seq_len, 32k]
                                                 │
                                                 ↓
                                             CE Loss
```

## 6. Losses for Training

### Stage 1: InfoNCE (Already Done)

```python
# Place revisitation loss
# Purpose: Train GRU to recognize revisited locations

loss = InfoNCELoss(temperature=0.07)
     = CrossEntropyLoss(logits, target=0)

where:
  logits[0] = cos_sim(anchor, positive)  ← want high (≈ 0.8)
  logits[1:9] = cos_sim(anchor, negatives)  ← want low (≈ 0)

Result: Positive similarity ≈ 0.7-0.8, Negative ≈ 0.1-0.3
```

### Stage 2: Navigation CE Loss (Current)

```python
# Action prediction loss
# Purpose: Train projectors to fuse vision + motion

loss = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
     = loss_fn(logits.view(-1, 32000), labels.view(-1))

where:
  logits: [seq_len, vocab_size=32000]
  labels: [seq_len] gold action tokens
  
  ignored positions: image tokens, padding

Result: Loss should decrease smoothly (2.5 → 1.8 over 3 epochs)
```

## 7. Key Design Choices

| Choice | Reason |
|--------|--------|
| **10-step motion window** | Captures ~1-2 seconds of navigation, enough for place context |
| **L2 normalization** | Maps embeddings to unit sphere, proper for contrastive learning |
| **Same-scene negatives** | Prevents GRU from using scene ID as shortcut |
| **Freeze GRU in Stage 2** | GRU already learned place patterns, focus on projector alignment |
| **Separate projectors** | Vision and motion have different input dims, need separate adapters |
| **Concat before LLM** | Simple fusion, allows LLM to learn cross-modal attention |

## 8. Expected Results

```
After 3 epochs on RTX A6000:

Training Loss:  2.5 → 1.8
Grad Norms:     0.05-0.15 (healthy range)
Time:           8-12 hours
Memory:         ~42GB (ZeRO-2)

Qualitative:
✓ Better action prediction on ambiguous frames
✓ Reduced hallucination on repetitive environments
✓ Improved place recognition via motion context
✓ More physically consistent action sequences
```

## 9. Files to Check

```
Dataset code:     llava/data/dataset.py (LazyVLNCEDataset)
GRU loading:      llava/model/grid_rnn.py (grid_rnn_ckpt_path)
Training loop:    llava/train/train.py (prepare_inputs_labels_for_multimodal)
GRU model:        extract_and_train_gru_info_nce.ipynb (MotionGRU class)
Training config:  scripts/finetune_projector_with_gru_v2.sh
```

## 10. Quick Reference: Data Dimensions

```
One sample through pipeline:

                Input          Processing        Output
                ─────          ──────────        ──────
Frames:         [42, H, W]  →  sample 8       → [8, 3, 384, 384]
Deltas:         [41, 3]     →  sample 8       → [8, 3]
                                normalize      → [8, 4]

Vision:         [8, 3, 384]  →  SigLIP        → [8, 730, 1152]
                             →  mm_projector  → [8×196, 4096]
                                              = [1568, 4096]

Motion:         [8, 4]       →  GRU           → [8, 128]
                             →  grid_to_vision→ [8, 4096]

Fused:          Both         →  concat        → [1576, 4096]

LLM:            Text + tokens→  LLaMA-3       → logits[seq_len, 32k]
```

---

**TL;DR:**
- **GRU input:** 10-step motion sequences (dx, dy, dyaw)
- **GRU training:** InfoNCE with place revisitation (already done)
- **Integration:** Grid2Vision projector projects [8, 128] → [8, 4096]
- **Injection point:** Concatenate with vision tokens before LLM
- **Loss to train:** Standard cross-entropy on action tokens
- **Trainable modules:** mm_projector + grid_to_vision only (~9M params)
