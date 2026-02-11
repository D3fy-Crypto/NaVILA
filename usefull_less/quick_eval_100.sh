#!/bin/bash

# Quick evaluation for 100 episodes with GRU integration

export PYTHONPATH="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA:$PYTHONPATH"
export GRU_CKPT_PATH="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"
export CUDA_VISIBLE_DEVICES=0

cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation

python run.py \
    --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
    --run-type eval \
    --num-chunks 1 \
    --chunk-idx 0 \
    EVAL_CKPT_PATH_DIR /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check \
    EVAL.EPISODE_COUNT 100
