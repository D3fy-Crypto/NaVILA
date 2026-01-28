# Integration Guide: Adding GRU Pose Data to NaVILA Training

This guide shows how to integrate the exported pose JSONL files into NaVILA's training pipeline.

## Step 1: Update `datasets_mixture.py`

Add `gru_pose_path` field to the R2R and RxR dataset registrations.

**File**: `llava/data/datasets_mixture.py`

```python
# Add this field to the Dataset dataclass (around line 20)
@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    gru_pose_path: str = field(default=None, metadata={"help": "Path to GRU pose sidecar JSONL."})  # NEW
    # ... rest of fields
```

Then update the dataset registrations:

```python
def register_datasets_mixtures():
    # ... other datasets ...
    
    r2r = Dataset(
        dataset_name="r2r",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/R2R/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/R2R/train",
        gru_pose_path="/PATH_TO_DATA/NaVILA-Dataset/R2R/gru_pose_train.jsonl",  # NEW
        description="350K VLN-CE R2R data. (augmented aith duplicate samples)",
    )
    add_dataset(r2r)

    rxr = Dataset(
        dataset_name="rxr",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/train",
        gru_pose_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/gru_pose_train.jsonl",  # NEW
        description="400K RxR data. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(rxr)
```

**Important**: Update paths for each split (train/val_seen/val_unseen) when needed.

## Step 2: Update `LazyVLNCEDataset`

Modify the dataset class to load and attach pose data.

**File**: `llava/data/dataset.py` (around line 2156)

### 2.1: Add pose loading in `__init__`

```python
class LazyVLNCEDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        gru_pose_path: Optional[str] = None,  # NEW parameter
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
                list_data_dict = [json.loads(q) for q in fp]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        
        # NEW: Load pose sidecar
        self.pose_data = {}
        if gru_pose_path is not None and os.path.exists(gru_pose_path):
            print(f"Loading GRU pose data from {gru_pose_path}")
            with open(gru_pose_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    self.pose_data[record['video_id']] = record
            print(f"Loaded {len(self.pose_data)} pose records")
        else:
            print("No GRU pose data provided or file not found")
```

### 2.2: Attach pose data in `__getitem__`

```python
def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    sources = self.list_data_dict[i]
    if isinstance(i, int):
        sources = [sources]
    assert len(sources) == 1, "Don't know why it is wrapped to a list"
    
    if ("frames" in sources[0]) and ("video_id" in sources[0]):
        video_id = sources[0]["video_id"]  # NEW: Extract video_id
        num_video_frames = self.data_args.num_video_frames
        frames = sources[0]["frames"]
        video_folder = self.image_folder
        video_paths = [os.path.join(video_folder, frame) for frame in frames]

        images, video_loading_succeed = self._load_video(video_paths, num_video_frames, self.data_args)
        num_frames_loaded_successfully = len(images)
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        # ... existing prompt construction code ...

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            data_dict["image"] = None
        
        # NEW: Attach pose data if available
        if video_id in self.pose_data:
            pose_record = self.pose_data[video_id]
            # Add poses as numpy arrays (convert to tensors in model if needed)
            data_dict["poses"] = np.array(pose_record["poses"], dtype=np.float32)
            data_dict["deltas"] = np.array(pose_record["deltas"], dtype=np.float32)
            data_dict["dist_bins"] = np.array(pose_record["dist_bins"], dtype=np.int64)
            data_dict["yaw_bins"] = np.array(pose_record["yaw_bins"], dtype=np.int64)
        else:
            # No pose data - fill with zeros or None
            data_dict["poses"] = None
            data_dict["deltas"] = None
            data_dict["dist_bins"] = None
            data_dict["yaw_bins"] = None
        
        return data_dict
    else:
        raise ValueError(f"Unknown data type: {sources[0]}")
```

## Step 3: Update Data Collator (Optional)

If you want to batch pose data, update the collator:

**File**: `llava/data/dataset.py` (around line 2305)

```python
@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # ... existing code for input_ids, labels, images ...
        
        # NEW: Collect pose data
        poses_list = []
        deltas_list = []
        dist_bins_list = []
        yaw_bins_list = []
        
        for instance in instances:
            if instance.get("poses") is not None:
                poses_list.append(torch.from_numpy(instance["poses"]))
                deltas_list.append(torch.from_numpy(instance["deltas"]))
                dist_bins_list.append(torch.from_numpy(instance["dist_bins"]))
                yaw_bins_list.append(torch.from_numpy(instance["yaw_bins"]))
        
        # Stack if available
        if len(poses_list) > 0:
            batch["poses"] = torch.stack(poses_list, dim=0)  # [B, num_frames, 3]
            batch["deltas"] = torch.stack(deltas_list, dim=0)  # [B, num_frames-1, 3]
            batch["dist_bins"] = torch.stack(dist_bins_list, dim=0)  # [B, num_frames-1]
            batch["yaw_bins"] = torch.stack(yaw_bins_list, dim=0)  # [B, num_frames-1]
        else:
            batch["poses"] = None
            batch["deltas"] = None
            batch["dist_bins"] = None
            batch["yaw_bins"] = None
        
        return batch
```

## Step 4: Update Dataset Builder

Make sure the `gru_pose_path` is passed to the dataset constructor.

**File**: `llava/data/builder.py` (or wherever datasets are instantiated)

Find where `LazyVLNCEDataset` is created and add the parameter:

```python
# Example (exact location depends on your builder code)
if dataset_cfg.dataset_type == "vlnce":
    dataset = LazyVLNCEDataset(
        data_path=dataset_cfg.data_path,
        image_folder=dataset_cfg.image_path,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        gru_pose_path=dataset_cfg.gru_pose_path,  # NEW
    )
```

## Step 5: Use Pose Data in Model

In your model forward pass (e.g., in `llava/model/llava_arch.py` or your GRU integration):

```python
def forward(self, input_ids, images, labels, **kwargs):
    # Extract pose data from kwargs
    poses = kwargs.get("poses", None)
    deltas = kwargs.get("deltas", None)
    dist_bins = kwargs.get("dist_bins", None)
    yaw_bins = kwargs.get("yaw_bins", None)
    
    if deltas is not None:
        # Feed deltas to GRU
        # deltas: [B, T-1, 3] where T=8 (num frames)
        motion_embeddings = self.gru_encoder(deltas)  # Your GRU module
        
        # Option A: Fuse motion embeddings with vision tokens
        # combined_features = self.fusion_layer(vision_features, motion_embeddings)
        
        # Option B: Add as separate tokens
        # input_embeds = torch.cat([text_embeds, motion_tokens, vision_tokens], dim=1)
        
        # Option C: Use for auxiliary loss
        # dist_logits, yaw_logits = self.classification_heads(motion_embeddings)
        # motion_loss = ce_loss(dist_logits, dist_bins) + ce_loss(yaw_logits, yaw_bins)
        # total_loss = lm_loss + motion_loss
    
    # ... rest of forward pass ...
```

## Testing the Integration

### Test 1: Verify pose data is loaded

```python
from llava.data.dataset import LazyVLNCEDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

dataset = LazyVLNCEDataset(
    data_path="/home/rithvik/NaVILA-Dataset/R2R/annotations.json",
    image_folder="/home/rithvik/NaVILA-Dataset/R2R/train",
    tokenizer=tokenizer,
    data_args=your_data_args,
    training_args=your_training_args,
    gru_pose_path="/home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl",
)

sample = dataset[0]
print(f"Poses shape: {sample['poses'].shape if sample['poses'] is not None else 'None'}")
print(f"Deltas shape: {sample['deltas'].shape if sample['deltas'] is not None else 'None'}")
```

Expected output:
```
Loading GRU pose data from /home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl
Loaded 353894 pose records
Poses shape: (8, 3)
Deltas shape: (7, 3)
```

### Test 2: Verify batching works

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=your_collator,
)

batch = next(iter(dataloader))
print(f"Batch poses: {batch['poses'].shape}")  # Should be [4, 8, 3]
print(f"Batch deltas: {batch['deltas'].shape}")  # Should be [4, 7, 3]
```

## Common Issues

### Issue 1: "KeyError: video_id"
- Make sure your annotations.json contains `video_id` field
- Check that video_id format matches between annotations and pose JSONL

### Issue 2: "File not found" for pose JSONL
- Verify the path in `datasets_mixture.py` is absolute and correct
- Check that you've run the pose export for the correct split

### Issue 3: Shape mismatch
- Verify num_frames in training matches export (default: 8)
- Check that poses are [T, 3] and deltas are [T-1, 3]

### Issue 4: Out of memory
- Pose data is small (~24 bytes per sample) but accumulated in memory
- For very large datasets, consider lazy loading from JSONL per-sample

## Next Steps

After integration:

1. **Train GRU separately** (optional but recommended):
   - Extract deltas and bins from all samples
   - Train a simple GRU on next-step + k-step prediction
   - Freeze GRU weights for NaVILA training

2. **Train NaVILA with GRU**:
   - Use existing training script: `scripts/train/sft_8frames.sh`
   - Add GRU module to model architecture
   - Choose fusion strategy (Option A/B/C above)

3. **Evaluate**:
   - Run evaluation as usual: `evaluation/scripts/eval/r2r.sh`
   - Check if motion encoding improves navigation metrics

## Questions?

If you need help with:
- Specific model architecture changes for GRU integration
- Loss function design for multi-task learning
- Hyperparameter tuning

Please refer to the main NaVILA documentation or consult with the team.
