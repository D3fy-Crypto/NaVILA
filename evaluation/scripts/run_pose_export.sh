#!/bin/bash
# Wrapper script to run pose export in the correct environment

# Activate navila-eval environment
eval "$(conda shell.bash hook)"
conda activate navila-eval

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "navila-eval" ]]; then
    echo "ERROR: Failed to activate navila-eval environment"
    echo "Please create it first using the README instructions"
    exit 1
fi

# Run export script
cd "$(dirname "$0")"
python export_gru_poses.py "$@"
