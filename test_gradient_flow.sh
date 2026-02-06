#!/bin/bash
# Test gradient flow WITHOUT deepspeed (single GPU)
# If this shows nonzero gradients, the issue is deepspeed-specific

source ~/miniconda3/etc/profile.d/conda.sh
conda activate navila

cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA

echo "=========================================="
echo "Testing gradient flow WITHOUT DeepSpeed"
echo "=========================================="
echo ""
echo "This will run 5 training steps on 1 GPU without DeepSpeed ZeRO"
echo "to check if mm_projector gradients are nonzero."
echo ""

python3 << 'EOF'
import torch
import sys
sys.path.insert(0, '/home/rithvik/NaVILA/llava')

print('[1] Loading model (this takes a minute)...')
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN

model_path = 'a8cheng/navila-siglip-llama3-8b-v1.5-pretrain'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

from llava.model.builder import load_pretrained_model
model_name = 'navila'
device = 'cuda'
model = load_pretrained_model(
    model_path, 
    model_base=None, 
    model_name=model_name,
    device=device
)
print('[1] ✓ Model loaded')

# Freeze everything except mm_projector
print('[2] Freezing non-projector parameters...')
for name, param in model.named_parameters():
    if 'mm_projector' not in name:
        param.requires_grad = False

print('[3] Checking mm_projector requires_grad...')
for name, param in model.named_parameters():
    if 'mm_projector' in name and param.requires_grad:
        print(f'   {name:50s} requires_grad=True')

print('[4] Creating dummy forward pass...')
batch_size = 1
seq_len = 16
vocab_size = 128259

input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
pixel_values = torch.randn(batch_size, 3, 384, 384).cuda()
attention_mask = torch.ones_like(input_ids).cuda()
labels = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

try:
    print('[5] Forward pass...')
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
    )
    
    loss = outputs.loss
    print(f'   Loss: {loss.item():.6f}')
    
    print('[6] Backward pass...')
    loss.backward()
    
    print('[7] Checking gradients after backward:')
    mm_proj_grad_sum = 0.0
    mm_proj_grad_count = 0
    for name, param in model.named_parameters():
        if 'mm_projector' in name and param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                mm_proj_grad_sum += grad_norm
                mm_proj_grad_count += 1
                if mm_proj_grad_count <= 3:
                    print(f'   {name:50s} grad_norm={grad_norm:.6f}')
    
    print('')
    if mm_proj_grad_sum > 0:
        print(f'✓✓✓ SUCCESS: mm_projector HAS gradients!')
        print(f'    Total gradient norm across {mm_proj_grad_count} param groups: {mm_proj_grad_sum:.6f}')
        print('')
        print('    This means:')
        print('    - Gradients ARE flowing through the model')
        print('    - The c10d::broadcast_ warning might be from DeepSpeed ZeRO overhead')
        print('    - Training should work, but verify in full training mode')
    else:
        print(f'✗✗✗ PROBLEM: mm_projector has NO gradients')
        print(f'    This means the loss does not depend on mm_projector.')
        print(f'    Check: is pixel_values being used in forward pass?')
        
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
EOF
