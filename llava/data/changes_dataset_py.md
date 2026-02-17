# README — Multimodal Preprocessing Update (C2: Separate Motion + Image Tokens)
# Changes def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
## 🧠 Goal

We added support for a new modality:

```
<motion>
```

and we use the **C2 design**:

```
<motion>
<image>
```

for every timestep.

This means:

* motion and vision are **separate tokens**
* but they are **placed next to each other**
* so they represent the **same time step**

This matches how modern multimodal LLMs (LLaVA / VILA / video-LLMs) interleave tokens.

---

# 🧩 What this function does (simple)

`preprocess_multimodal()` **does NOT create tensors**.

It only:

🧹 cleans the prompt text
📏 makes special tokens consistent

so the tokenizer can later convert:

```
<motion> → MOTION_TOKEN_INDEX
<image>  → IMAGE_TOKEN_INDEX
```

and the model can inject embeddings at the correct positions.

---

# 🆕 What changed

### Before

The function only knew how to format:

```
<image>
```

### Now

It formats **both**:

```
<image>
<motion>
```

in the **same canonical way**.

🚨 Image behavior is unchanged.

---

# ✨ Why formatting matters

Bad formatting:

```
turn left<motion>go forward
```

Tokenizer sees:

```
["turn", "left<motion>go", "forward"]
```

❌ motion token is LOST inside a word.

Good formatting:

```
turn left <motion>
go forward
```

Tokenizer sees:

```
["turn", "left", MOTION_TOKEN]
```

✅ model can inject motion embedding.

---

# 🔍 What the code does step-by-step

## 1️⃣ Auto-insert `<image>` at the beginning (existing behavior)

If the conversation has no image at all:

### Before

```
"Go to the chair"
```

### After

```
<image>
Go to the chair
```

This is required by LLaVA-style training.

---

## 2️⃣ Canonical cleanup for `<image>` (unchanged)

It fixes spacing and ensures:

```
<image>\n
```

### Example

#### Input

```
"Go to the chair<image>turn left"
```

#### Output

```
Go to the chair <image>
turn left
```

---

## 3️⃣ Optional training wrappers for image (unchanged)

If enabled:

```
<image>
```

becomes:

```
<im_start><image><im_end>
```

or

```
<Image><image></Image>
```

This depends on your conversation template.

Motion does **not** use these.

---

## 4️⃣ New: Canonical cleanup for `<motion>`

We added the **same logic as image**, but without wrappers.

### Example

#### Input

```
turn left<motion>go forward
```

#### Output

```
turn left <motion>
go forward
```

---

## 5️⃣ Newline enforcement

We guarantee:

```
<motion>
<image>
```

never:

```
<motion><image>
```

or

```
<motion>

```

Always exactly one newline.

---

# 🎬 Full Example — Before vs After

## Input prompt (from dataset)

```
"History: <motion><image><motion><image> Go to the chair"
```

## Output after preprocessing

```
History:
<motion>
<image>
<motion>
<image>
Go to the chair
```

Now the tokenizer produces:

```
[MOTION_TOKEN_INDEX,
 IMAGE_TOKEN_INDEX,
 MOTION_TOKEN_INDEX,
 IMAGE_TOKEN_INDEX,
 text tokens...]
```

Perfect alignment with:

```
motion_tensor.shape[0]
image_tensor.shape[0]
```

---

# 🧠 Why we use C2 (two tokens) instead of fusing them

We want the transformer to learn:

```
state_t = motion_t + vision_t
```

Self-attention will automatically fuse them.

This gives:

✅ temporal alignment
✅ modality-specific reasoning
✅ compatibility with LLaVA / VILA training

---

# 📦 What this function does NOT do

It does **not**:

* create motion tensors
* run GRU
* insert embeddings

It only prepares the text so later stages can do that.

---

# 🔗 Where this connects in the pipeline

```
Dataset builds prompt
        ↓
preprocess_multimodal()  ← (THIS CHANGE)
        ↓
tokenizer → special token indices
        ↓
collator → batch tensors
        ↓
model → inject motion & vision embeddings
```

---

# 🧪 Rule you must follow in the dataset

For C2 to work:

```
#number of <motion> tokens
        ==
motion_tensor.shape[0]
```

and

```
#number of <image> tokens
        ==
image_tensor.shape[0]
```

---

# 🏁 Final mental model

We turned messy text like:

```
<motion><image>go left
```

into a clean, structured multimodal timeline:

```
<motion>
<image>
go left
```

so the transformer can understand:

> “This motion and this image happened at the same time.”

---

# ✅ That’s the entire change

Image logic: unchanged
Motion logic: added in the same style
Architecture: C2 interleaved tokens

---
