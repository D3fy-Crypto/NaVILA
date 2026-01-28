#!/bin/bash
# Export all splits for R2R and RxR datasets
# Run this from evaluation/scripts/

set -e  # Exit on error

echo "========================================"
echo "NaVILA Pose Export - Full Pipeline"
echo "========================================"
echo

# Configuration
NAVILA_ROOT="/home/rithvik/NaVILA-Dataset"
DATA_ROOT="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets"
FRAMES=8

# Function to run export with error handling
run_export() {
    local dataset=$1
    local split=$2
    local output=$3
    
    echo "----------------------------------------"
    echo "Exporting: $dataset $split"
    echo "Output: $output"
    echo "----------------------------------------"
    
    bash run_pose_export.sh \
        --dataset "$dataset" \
        --split "$split" \
        --frames $FRAMES \
        --navila-root "$NAVILA_ROOT" \
        --data-root "$DATA_ROOT" \
        --output "$output"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully exported $dataset $split"
        # Show file info
        if [ -f "$output" ]; then
            lines=$(wc -l < "$output")
            size=$(du -h "$output" | cut -f1)
            echo "  Records: $lines"
            echo "  Size: $size"
        fi
    else
        echo "✗ Failed to export $dataset $split"
        return 1
    fi
    echo
}

# Export R2R
echo "Starting R2R exports..."
echo

run_export "r2r" "train" "$NAVILA_ROOT/R2R/gru_pose_train.jsonl"
run_export "r2r" "val_seen" "$NAVILA_ROOT/R2R/gru_pose_val_seen.jsonl"
run_export "r2r" "val_unseen" "$NAVILA_ROOT/R2R/gru_pose_val_unseen.jsonl"

# Export RxR
echo "Starting RxR exports..."
echo

run_export "rxr" "train" "$NAVILA_ROOT/RxR/gru_pose_train.jsonl"
run_export "rxr" "val_unseen" "$NAVILA_ROOT/RxR/gru_pose_val_unseen.jsonl"

echo "========================================"
echo "All exports complete!"
echo "========================================"
echo
echo "Files created:"
ls -lh "$NAVILA_ROOT"/R2R/gru_pose_*.jsonl 2>/dev/null || echo "  (No R2R files)"
ls -lh "$NAVILA_ROOT"/RxR/gru_pose_*.jsonl 2>/dev/null || echo "  (No RxR files)"
echo
echo "Next steps:"
echo "1. Verify exports with sanity checks (see POSE_EXPORT_README.md)"
echo "2. Update llava/data/datasets_mixture.py with pose paths"
echo "3. Integrate poses into dataset loader (see INTEGRATION_GUIDE.md)"
