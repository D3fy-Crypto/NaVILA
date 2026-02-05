# GRU Integration: Verification & Monitoring Guide

## Implementation Status: ✅ COMPLETE

All components have been implemented and instrumented with comprehensive logging and monitoring.

## Quick Start

```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA
bash launch_gru_training.sh
```

This will:
1. Run verification tests (7 comprehensive checks)
2. Set environment variables
3. Start training with full monitoring enabled

## Verification Tests

The `verify_gru_integration.py` script runs 7 critical tests:

### Test 1: Checkpoint Files
- ✅ Checks GRU checkpoint exists at expected path
- ✅ Verifies oracle deltas file is present

### Test 2: Module Imports
- ✅ Tests grid_rnn module imports
- ✅ Tests args module with gru_ckpt_path

### Test 3: MotionGRU Forward Pass
- ✅ Instantiates MotionGRU (256→256)
- ✅ Tests forward pass with batch of 8 pose deltas
- ✅ Verifies output shape: (batch=4, seq=8, hidden=256)

### Test 4: GridToVisionProjector
- ✅ Tests projector (256→4096)
- ✅ Verifies motion features are upsampled correctly

### Test 5: MotionEncoderWithProjector
- ✅ Loads pretrained GRU checkpoint
- ✅ Tests end-to-end: deltas → motion features → vision tokens
- ✅ Verifies shapes: (4,8,4) → (4,8,256) → (4,8,4096)

### Test 6: Pose Delta Normalization
- ✅ Tests dx normalization (÷0.25)
- ✅ Tests yaw trigonometry (sin/cos)
- ✅ Verifies 4D input: [dx, dy, dz, yaw]

### Test 7: Gradient Flow
- ✅ Tests frozen GRU (gradients = 0)
- ✅ Tests trainable projector (gradients > 0)
- ✅ Verifies proper parameter freezing

## Logging Infrastructure

### 1. Checkpoint Loading (grid_rnn.py)
```
================================================
Loading pretrained GRU checkpoint from:
/path/to/motion_gru_infonce.pt
Checkpoint keys: ['model_state_dict', 'metrics']
✅ Loaded 169,472 parameters into MotionGRU
================================================
```

### 2. Pose Deltas Loading (dataset.py)
```
================================================
✅ Loading pose deltas from oracle_deltas_train.jsonl
✅ Loaded 1,234 episodes with pose deltas
================================================
```

### 3. Motion Encoder Initialization (llava_arch.py)
```
================================================
🚀 Initializing MotionEncoder with GRU
  GRU checkpoint: /path/to/motion_gru_infonce.pt
  Output dimension: 4096
================================================
```

### 4. Forward Pass Logging (llava_arch.py)
```
[MOTION] pose_deltas shape: torch.Size([4, 8, 4])
[MOTION] motion_features shape: torch.Size([4, 8, 4096])
```

### 5. Gradient Monitoring (gru_monitor.py)
Every 10 training steps:
```
========== GRU Training Monitor ==========
[Frozen] motion_gru: 169,472 params
[Trainable] grid_to_vision: 1,052,672 params
==========================================

Step 100 Gradient Norms:
  grad_norm/motion_gru: 0.0000 (frozen ✅)
  grad_norm/grid_to_vision: 2.3456 (training ✅)
```

## WANDB Integration

Gradient metrics are automatically logged to Weights & Biases:

### Logged Metrics
- `grad_norm/motion_gru` - Should always be 0 (frozen)
- `grad_norm/grid_to_vision` - Should be > 0 (trainable)
- `loss` - Training loss (expect 2.5 → 1.8)
- `learning_rate` - LR schedule
- `epoch` - Training progress

### Expected Patterns
1. **Frozen GRU**: `grad_norm/motion_gru` = 0.0 throughout training
2. **Training Projector**: `grad_norm/grid_to_vision` varies but typically 1.0-5.0
3. **Loss Convergence**: Decreasing from ~2.5 to ~1.8 over 8-12 hours

## Modified Files Summary

### Core Implementation (7 files)
1. **grid_rnn.py** (NEW) - 280 lines
   - MotionGRU class
   - GridToVisionProjector class
   - MotionEncoderWithProjector class
   - Checkpoint loading with logging

2. **dataset.py** - Modified
   - Added pose_deltas_path loading
   - Enhanced logging for oracle deltas
   - Normalization: dx/0.25, sin/cos(yaw)

3. **llava_arch.py** - Modified
   - Added motion_encoder initialization
   - Forward pass motion token injection
   - Logging for motion encoding

4. **args.py** - Modified
   - Added pose_deltas_path argument
   - Added gru_ckpt_path argument
   - Added tune_motion_gru flag

5. **configuration_llava.py** - Modified
   - Added gru_ckpt_path to LlavaConfig

6. **builder.py** - Modified
   - Pass pose_deltas_path to dataset

7. **sft_8frames_gru.sh** (NEW)
   - Training script with GRU args

### Monitoring & Verification (3 files)
8. **verify_gru_integration.py** (NEW) - 7 tests
9. **gru_monitor.py** (NEW) - Gradient monitoring
10. **launch_gru_training.sh** (NEW) - Training launcher

### Training Integration (2 files)
11. **llava/train/train.py** - Modified
    - Import GRUTrainingMonitor
    - Instantiate and attach to trainer
    - Set motion encoder trainability

12. **llava/train/llava_trainer.py** - Modified
    - Inject gradient metrics into logs
    - Log to WANDB automatically

13. **llava/train/utils.py** - Modified
    - Pass gru_ckpt_path to config
    - Pass tune_motion_gru to config

## Configuration Flow

```
Training Script (sft_8frames_gru.sh)
  ↓ --gru_ckpt_path, --tune_motion_gru
ModelArguments / TrainingArguments (args.py)
  ↓ model_args.gru_ckpt_path, training_args.tune_motion_gru
prepare_config_for_training() (utils.py)
  ↓ config.gru_ckpt_path, config.tune_motion_gru
LlavaConfig (configuration_llava.py)
  ↓ stored in config object
init_vlm() (llava_arch.py)
  ↓ MotionEncoderWithProjector(gru_ckpt_path)
Checkpoint Loaded & Motion Encoder Active
```

## Gradient Flow

```
Input: pose_deltas (batch, 8, 4)
  ↓
MotionGRU [FROZEN ❄️]
  - 169,472 parameters
  - gradients = 0
  ↓ hidden states (batch, 8, 256)
GridToVisionProjector [TRAINABLE 🔥]
  - 1,052,672 parameters
  - gradients > 0
  ↓
motion_features (batch, 8, 4096)
  ↓
Injected into LLM input_embeds
```

## Troubleshooting

### Issue: "No module named 'transformers'"
**Solution**: Activate correct conda environment before running
```bash
conda activate navila  # or your environment name
```

### Issue: Checkpoint loading fails
**Check**: Path is correct in sft_8frames_gru.sh
```bash
--gru_ckpt_path /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt
```

### Issue: Oracle deltas not found
**Check**: Path exists
```bash
ls /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/oracle_exports/oracle_deltas_train.jsonl
```

### Issue: WANDB not logging
**Solution**: Set WANDB environment variables
```bash
export WANDB_PROJECT="navila-gru"
export WANDB_RUN_NAME="gru_8frames_sft"
```

### Issue: Gradients are NaN
**Check**: 
1. GRU gradients should be 0 (frozen)
2. Projector gradients should be 1-5 range
3. If loss diverges, reduce learning rate in training script

## Expected Training Output

```
====================================
GRU Integration Training Launcher
====================================

Step 1: Running implementation verification...
--------------------------------------
[Test 1/7] Checking checkpoint files... ✅ PASS
[Test 2/7] Testing imports... ✅ PASS
[Test 3/7] Testing MotionGRU... ✅ PASS
[Test 4/7] Testing GridToVisionProjector... ✅ PASS
[Test 5/7] Testing MotionEncoderWithProjector... ✅ PASS
[Test 6/7] Testing pose delta normalization... ✅ PASS
[Test 7/7] Testing gradient flow... ✅ PASS

All tests passed! ✅

✅ Verification passed!

Step 2: Setting environment variables...
--------------------------------------
GPUS_PER_NODE=4
n_node=1
MASTER_PORT=29500
MASTER_ADDR=localhost
CURRENT_RANK=0

Step 3: Starting training with GRU integration...
--------------------------------------
Training script: scripts/train/sft_8frames_gru.sh
Expected training time: 8-12 hours
Loss trajectory: 2.5 → 1.8

Monitoring features enabled:
  ✅ Checkpoint loading logs
  ✅ Pose deltas loading logs
  ✅ Forward pass motion encoding logs
  ✅ Gradient flow monitoring (every 10 steps)
  ✅ WANDB integration (if configured)

[Model initialization logs...]
================================================
Loading pretrained GRU checkpoint from:
/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt
✅ Loaded 169,472 parameters into MotionGRU
================================================

[Dataset loading logs...]
================================================
✅ Loading pose deltas from oracle_deltas_train.jsonl
✅ Loaded 1,234 episodes with pose deltas
================================================

[Training begins...]
Step 10: loss=2.456, grad_norm/motion_gru=0.000, grad_norm/grid_to_vision=2.134
Step 20: loss=2.398, grad_norm/motion_gru=0.000, grad_norm/grid_to_vision=2.087
...
```

## Post-Training Validation

After training completes, verify:

1. **Check loss convergence**:
   ```bash
   tail -n 100 checkpoints/gru_8frames_sft/log_history.json
   ```

2. **Verify gradient logs**:
   - motion_gru gradients should always be 0
   - grid_to_vision gradients should be non-zero

3. **Test inference**:
   ```bash
   python test_gru_inference.py --checkpoint checkpoints/gru_8frames_sft
   ```

4. **Evaluate on validation set**:
   ```bash
   cd evaluation
   bash scripts/evaluate_gru.sh
   ```

## Files Locations

- Implementation: `/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/`
- GRU checkpoint: `evaluation/checkpoints/motion_gru_infonce.pt`
- Oracle deltas: `evaluation/oracle_exports/oracle_deltas_train.jsonl`
- Training output: `checkpoints/gru_8frames_sft/`
- Logs: `checkpoints/gru_8frames_sft/log_history.json`

## Contact & Issues

If you encounter issues:
1. Check logs in training output directory
2. Review verification script output
3. Ensure all paths are correct in training script
4. Verify conda environment is activated
5. Check GPU availability: `nvidia-smi`

---

**Status**: ✅ Ready for training
**Last Updated**: 2024
**Implementation**: Complete with full monitoring
