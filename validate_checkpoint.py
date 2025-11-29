"""Quick checkpoint validator"""
import torch

checkpoint = torch.load('data/training_results/model_20251117_230212.pth', weights_only=False)

print("="*80)
print("🔍 CHECKPOINT VALIDATION")
print("="*80)

# Structure
print(f"✅ Checkpoint loadable")
print(f"Keys: {list(checkpoint.keys())}")

# Model state
state = checkpoint['model_state_dict']
print(f"\n📊 Model State:")
print(f"  Layers: {len(state)} tensors")

# Check each tensor for NaN/Inf
all_clean = True
for name, tensor in state.items():
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
        all_clean = False

if all_clean:
    print("  ✅ NO NaN, NO Inf in any layer")

# Sample weights
fc1_weight = state['fc1.weight']
print(f"\n🎯 Sample Layer (fc1.weight):")
print(f"  Shape: {fc1_weight.shape}")
print(f"  Min: {fc1_weight.min().item():.6f}")
print(f"  Max: {fc1_weight.max().item():.6f}")
print(f"  Mean: {fc1_weight.mean().item():.6f}")
print(f"  Std: {fc1_weight.std().item():.6f}")

# Optimizer state
opt_state = checkpoint['optimizer_state_dict']
print(f"\n⚙️  Optimizer State:")
print(f"  Param groups: {len(opt_state['param_groups'])}")
print(f"  State entries: {len(opt_state['state'])}")

# Training history
history = checkpoint['history']
print(f"\n📈 Training History:")
print(f"  Epochs: {len(history['epochs'])}")
print(f"  Final loss: {history['losses'][-1]:.6f}")
print(f"  Final accuracy: {history['accuracies'][-1]*100:.2f}%")

print("\n" + "="*80)
if all_clean:
    print("✅ CHECKPOINT IS WORK-PROOFED - CLEAN AND LOADABLE")
else:
    print("❌ CHECKPOINT HAS CORRUPTION")
print("="*80)
