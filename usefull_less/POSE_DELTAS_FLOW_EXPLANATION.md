# Pose Deltas Flow Explanation

## Overview
This document explains how pose deltas (motion data) flow from oracle files to the GRU and into the VLM for evaluation.

## Complete Data Flow

### 1. **Oracle Deltas Loading** (navila_trainer.py:67-99)
```
Location: evaluation/vlnce_baselines/navila_trainer.py:_load_oracle_deltas()
```

**Input File**: `evaluation/oracle_exports/oracle_deltas_val_unseen.jsonl`

**Format** (per line):
```json
{
  "episode_id": 1,
  "deltas": [[dx, dy, dyaw], [dx, dy, dyaw], ...],
  "poses": [...],
  "actions": [2, 2, 1, ...]
}
```

**Process**:
- Reads JSONL file line by line
- Extracts `episode_id` and `deltas` array
- Stores in dict: `{episode_id: deltas_list}`
- Loaded ONCE at eval start for split (val_unseen)

**Result**: Dictionary with 1839 episodes loaded

---

### 2. **Per-Episode Delta Tensor Building** (navila_trainer.py:101-127, 290-300)

**When**: Every evaluation step (inside eval loop)

**Process**:
1. Get current episode_id: `episode_id = current_episodes[0].episode_id`
2. Lookup deltas: `deltas_list = oracle_deltas.get(episode_id)`
3. If found, call `_build_pose_deltas_tensor(deltas_list, num_frames=8, device)`

**Transformation** (in `_build_pose_deltas_tensor`):
```python
# Input: Raw deltas [[dx, dy, dyaw], ...]
# Output: Tensor [1, num_frames, 4]

# Step 1: Sample to num_frames-1 (7 deltas for 8 frames)
if len(deltas_list) >= num_frames:
    # Linear sampling across trajectory
    indices = np.linspace(0, len(deltas_list)-1, num_frames-1)
    sampled = [deltas_list[idx] for idx in indices]

# Step 2: Normalize each delta
for [dx, dy, dyaw] in sampled:
    dx_norm = dx / 0.25          # Normalize by step size
    dy_norm = dy / 0.25
    processed.append([
        dx_norm, 
        dy_norm, 
        np.sin(dyaw),            # Convert angle to sin/cos
        np.cos(dyaw)
    ])

# Step 3: Pad to num_frames (add zero delta at start)
if len(processed) < num_frames:
    processed.extend([[0, 0, 0, 1]] * padding_needed)

# Step 4: Convert to tensor
tensor = torch.tensor(processed, dtype=float32, device=cuda)
return tensor.unsqueeze(0)  # Shape: [1, 8, 4]
```

**Why normalize?**
- dx/0.25, dy/0.25: Normalize by Habitat's step size (0.25m)
- sin(dyaw), cos(dyaw): Continuous representation of rotation (avoids wrap-around)

---

### 3. **Passing to Model.generate()** (navila_trainer.py:342-351)

**Code**:
```python
output_ids = model.generate(
    input_ids,                              # Text tokens
    images=images_tensor.half().cuda(),     # [1, 8, 3, H, W] video frames
    pose_deltas=pose_deltas_tensor,         # [1, 8, 4] motion deltas <<< HERE
    do_sample=False,
    temperature=0.0,
    max_new_tokens=32,
    use_cache=True,
    stopping_criteria=[stopping_criteria],
    pad_token_id=tokenizer.eos_token_id,
)
```

**What's passed**:
- `images`: 8 RGB frames (224x224) from history + current
- `pose_deltas`: 8 motion deltas (4-dim vectors)

---

### 4. **Model.generate() Processing** (llava_arch.py:868-889)

**Entry Point**: `LlavaMetaForCausalLM.generate()`

**Step 1: Extract pose_deltas**:
```python
def generate(self, input_ids, images, **generation_kwargs):
    # Extract pose_deltas (don't pass to LLM.generate)
    pose_deltas = generation_kwargs.pop('pose_deltas', None)
    
    # Prepare multimodal inputs
    (_, _, attention_mask, _, inputs_embeds, _) = 
        self.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None, 
            images, 
            pose_deltas=pose_deltas  # <<< Pass here
        )
```

**Why pop()?**: LLM's generate() doesn't accept pose_deltas, only our wrapper does

---

### 5. **Motion Encoding in prepare_inputs_labels_for_multimodal()** (llava_arch.py:342-350)

**GRU Processing**:
```python
# Encode images (Vision Tower → MM Projector)
image_features = self.encode_images(images)  # [1, 8, 576, 4096]

# Encode motion tokens if available
motion_features = None
if pose_deltas is not None and self.get_motion_encoder() is not None:
    # pose_deltas: [1, 8, 4]
    motion_features = self.get_motion_encoder()(pose_deltas)  
    # motion_features: [1, 128] (GRU final hidden state)
```

**GRU Architecture** (MotionEncoderWithProjector):
```python
# 1. GRU processes sequence
gru_input: [1, 8, 4]  # (batch, seq_len, input_size)
gru_output, hidden = self.gru(gru_input)  # hidden: [2 layers, 1, 256]

# 2. Take final hidden state from last layer
embedding = hidden[-1]  # [1, 256]

# 3. Project to 128-dim
embedding = self.embedding_layer(embedding)  # [1, 128]

# 4. Grid-to-Vision Projector
motion_features = self.projector(embedding)  # [1, 4096]
#   Projector: Linear(128 → 256 → 4096) to match LLM hidden size
```

**Result**: Single 4096-dim motion token representing entire trajectory

---

### 6. **Token Injection** (llava_arch.py:441-453)

**Injection Point**: After ALL image tokens, before instruction text

```python
# For each sample in batch:
cur_new_input_embeds = []

# Add text tokens before first <image>
cur_new_input_embeds.append(text_before_images)

# Add all 8 image features (576 tokens each = 4608 tokens total)
for image_idx in range(8):
    cur_new_input_embeds.append(image_features[image_idx])  # [576, 4096]
    cur_new_input_embeds.append(text_between_images)

# INJECT MOTION TOKEN HERE (1 token, 4096-dim)
if motion_features is not None:
    cur_motion_features = motion_features[batch_idx:batch_idx+1]  # [1, 4096]
    cur_new_input_embeds.append(cur_motion_features)

# Add remaining text tokens
cur_new_input_embeds.append(text_after_images)

# Concatenate all embeddings
final_embeds = torch.cat(cur_new_input_embeds)  # [total_tokens, 4096]
```

**Final Token Sequence**:
```
[BOS] <system_prompt> <image_tokens_576> ... <image_tokens_576> [MOTION_TOKEN] <instruction> <answer_start>
  └── text ──┘  └───────── 8 images (4608 tokens) ──────────┘  └─ 1 token ─┘  └── text ──┘
```

---

### 7. **LLM Generation** (llava_arch.py:887)

```python
outputs = self.llm.generate(
    inputs_embeds=final_embeds,  # Includes motion token!
    attention_mask=attention_mask,
    **generation_kwargs
)
```

The LLM sees motion information as an additional context token.

---

## Why Only One Output Per Episode?

**Current Behavior**: Model outputs "stop" immediately on most episodes

**Reasons**:

1. **Model is only 0.1 epoch trained** (sanity check):
   - Not enough training to learn navigation
   - Loss went from 4.21 → 0.16 but that's just 2212 steps
   - Needs 3 full epochs to learn proper navigation

2. **All Past Frames Are Sent Every Step**:
   ```python
   past_and_current_rgbs = past_rgbs[0] + [curr_rgb]  # Growing history
   ```
   The model sees the ENTIRE trajectory so far, which may be confusing for an undertrained model

3. **No Action History Context**:
   - Model doesn't know what actions it took before
   - Only sees visual + motion features

---

## Expected Flow for Proper Navigation

**Ideal Sequence** (after full training):
1. **Step 1**: See instruction + 1 frame → "turn left 45°"
2. **Step 2**: See instruction + 2 frames + past motion → "move forward 50cm"
3. **Step 3**: See instruction + 3 frames + past motion → "turn right 30°"
4. ... continue ...
5. **Step N**: Recognize goal reached → "stop"

---

## Current Issues Summary

### ✅ Working:
1. Oracle deltas loading: **1839 episodes loaded**
2. Tensor building: **[1, 8, 4] tensors created**
3. GRU loading: **Checkpoint loaded successfully**
4. Motion encoding: **GRU → Projector → 4096-dim token**
5. Token injection: **Motion token appended after images**

### ❓ Unclear (Need Logging):
1. Are pose_deltas **actually found** for each episode_id?
2. Is motion_features **None or valid** during forward pass?
3. Does the log "[Forward] Motion encoding" ever appear?

### 🔴 Known Issues:
1. **Model immediately outputs "stop"** → Undertrained (0.1 epoch)
2. **Only one action per episode** → Model not confident in navigation

---

## How to Verify GRU Integration

### Add These Logs:

1. **In navila_trainer.py** (already added):
   ```python
   logger.info(f"[Eval] Episode {episode_id}: deltas_list={'found' if deltas_list else 'NOT FOUND'}")
   logger.info(f"[Eval] Built pose_deltas_tensor: {pose_deltas_tensor.shape}")
   ```

2. **In llava_arch.py:348** (add this):
   ```python
   if pose_deltas is not None:
       logger.info(f"[Forward] Motion encoding: {pose_deltas.shape} → {motion_features.shape}")
   else:
       logger.warning("[Forward] No pose_deltas provided!")
   ```

3. **Check logs for**:
   ```bash
   grep "Motion encoding" eval_100_run.log
   grep "deltas_list=" eval_100_run.log | head -10
   ```

---

## Next Steps

1. **Verify GRU is receiving inputs**: Check for "Motion encoding" logs
2. **Continue full training**: Run 3 epochs to properly train navigation
3. **Compare with/without GRU**: Run eval with GRU_CKPT_PATH="" to see baseline

---

## File Summary

| File | Purpose |
|------|---------|
| `oracle_deltas_val_unseen.jsonl` | Raw trajectory deltas per episode |
| `navila_trainer.py:_load_oracle_deltas()` | Load deltas into memory |
| `navila_trainer.py:_build_pose_deltas_tensor()` | Convert to [1,8,4] tensor |
| `llava_arch.py:generate()` | Extract pose_deltas from kwargs |
| `llava_arch.py:prepare_inputs_labels_for_multimodal()` | Encode via GRU |
| `grid_rnn.py:MotionEncoderWithProjector` | GRU + Projector architecture |
| `llava_arch.py` (token injection) | Append motion token after images |

