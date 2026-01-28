# Oracle Path Visualization - Quick Start Guide

## Script Location
`evaluation/scripts/visualize_oracle_path.py`

## What It Does
- Loads a single environment from the R2R dataset
- Agent follows the oracle (shortest path) to the goal
- Visualizes each step with:
  - RGB observation
  - Current instruction
  - Action taken
  - Distance to goal
- Saves output as video and images

## Basic Usage

### 1. Run with defaults (single episode from train split):
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation

conda activate navila-eval  # or your env name

python scripts/visualize_oracle_path.py
```

### 2. Run on validation split:
```bash
python scripts/visualize_oracle_path.py --split val_unseen
```

### 3. Run multiple episodes:
```bash
python scripts/visualize_oracle_path.py --split train --num-episodes 5
```

### 4. Custom output directory:
```bash
python scripts/visualize_oracle_path.py --output-dir my_visualizations
```

### 5. Skip video (only save images):
```bash
python scripts/visualize_oracle_path.py --no-video
```

## Command-Line Arguments

```
--config PATH          Config file path (default: vlnce_baselines/config/r2r_baselines/navila.yaml)
--split SPLIT          Dataset split: train, val_seen, val_unseen, test (default: train)
--episode-id ID        Specific episode ID to visualize (optional)
--max-steps N          Maximum steps per episode (default: 500)
--output-dir DIR       Output directory (default: oracle_visualization)
--no-video             Skip saving video, only save images
--no-map               Disable top-down map
--num-episodes N       Number of episodes to run (default: 1)
```

## Output Files

After running, you'll find in `oracle_visualization/`:

1. **Video**: `oracle_episode_<ID>.mp4`
   - Shows agent navigation step-by-step
   - 10 FPS for easy viewing

2. **Images**:
   - `episode_<ID>_start.png` - First frame
   - `episode_<ID>_end.png` - Last frame

3. **Metrics**: `episode_<ID>_metrics.txt`
   - Episode information
   - Final navigation metrics (Success, SPL, etc.)

## Example Output

```
================================================================================
Oracle Path Visualization
================================================================================

[1] Creating environment...
[2] Resetting environment...

[Episode Info]
  Episode ID: 1234
  Scene: 17DRP5sb8fy
  Start Position: [1.23, 0.10, -4.56]
  Goal Position: [7.89, 0.10, 2.34]
  Instruction: Walk forward through the doorway and turn left...

[3] Running oracle navigation...
--------------------------------------------------------------------------------
Step   0: Action=MOVE_FORWARD  | Distance to Goal=12.45m
Step   1: Action=MOVE_FORWARD  | Distance to Goal=12.20m
Step   2: Action=TURN_LEFT     | Distance to Goal=12.20m
...
Step  42: Action=STOP          | Distance to Goal=0.32m
--------------------------------------------------------------------------------

[4] Episode Complete!

[Final Metrics]
  Total Steps: 43
  Success: True
  SPL: 0.847
  Distance to Goal: 0.32m
  Path Length: 10.75m
  Oracle Success: 1.000
  NDTW: 0.923

[5] Saving video to: oracle_visualization/oracle_episode_1234.mp4
  Saved 43 frames at 10 FPS
  Saved start frame: oracle_visualization/episode_1234_start.png
  Saved end frame: oracle_visualization/episode_1234_end.png
  Saved metrics: oracle_visualization/episode_1234_metrics.txt

================================================================================
Visualization Complete!
================================================================================
```

## Watch the Video

After running, open the video:

```bash
# Linux
xdg-open oracle_visualization/oracle_episode_*.mp4

# Or copy to your local machine and open
scp user@server:/path/to/oracle_visualization/*.mp4 ./
```

## Troubleshooting

### Issue: Missing imageio or PIL
```bash
pip install imageio imageio-ffmpeg pillow
```

### Issue: Can't find config file
Make sure you're in the `evaluation/` directory when running the script.

### Issue: No episodes found
Check that your dataset is properly set up:
```bash
ls data/datasets/R2R_VLNCE_v1-3_preprocessed/train/
# Should see: train.json.gz, train_gt.json.gz
```

### Issue: Video doesn't play
Install ffmpeg:
```bash
conda install ffmpeg  # or: sudo apt install ffmpeg
```

## Integration with Your Workflow

This script demonstrates:
1. How to load the environment with oracle sensor
2. How to extract oracle actions
3. How to step through episodes
4. How to visualize observations

You can adapt this code for:
- Testing your own agent
- Comparing model vs oracle
- Debugging navigation issues
- Creating training demonstrations

## See Also

- Main guide: `evaluation/TRAINING_AND_ORACLE_GUIDE.md`
- Environment utils: `evaluation/vlnce_baselines/common/env_utils.py`
- Shortest path follower: `evaluation/habitat_extensions/shortest_path_follower.py`
