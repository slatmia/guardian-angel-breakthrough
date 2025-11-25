# Guardian Angel Breakthrough v2.0 - Backup & Recovery Guide

## 📦 Backup Created: November 23, 2025

**Location:** `BREAKTHROUGH_BACKUP_20251123_213132`

---

## 📋 Backup Contents (Verified)

### Critical Model Files:

| File | Size | Purpose |
|------|------|---------|
| **iteration_04_checkpoint.pt** | 18.04 MB | PyTorch checkpoint with 0.98+ trained weights |
| **lora_adapter.safetensors** | 6.01 MB | Exported LoRA adapter (1,574,662 parameters) |
| **RETRAINNING-DATA-V2.md** | 0.05 MB | 17 training examples (breakthrough dataset) |
| **Modelfile.breakthrough-v2** | <1 KB | Ollama model definition |
| **MODEL_NOTES.md** | 0.01 MB | Complete training documentation |
| **adapter_config.json** | <1 KB | LoRA configuration (rank 64, alpha 128) |
| **adapter_metadata.json** | <1 KB | Training metadata (0.924 score) |
| **Modelfile.enhanced** | <1 KB | Alternative modelfile |

**Total Backup Size:** ~24.11 MB

---

## 🔄 Recovery Procedures

### Scenario 1: Model Accidentally Deleted from Ollama

**Recovery Command:**
```powershell
cd C:\Users\sergi\.ollama\models\manifests\registry.ollama.ai\library\gemma3\models\BREAKTHROUGH_BACKUP_20251123_213132

# Recreate the model
ollama create guardian-angel:breakthrough-v2 -f Modelfile.breakthrough-v2
```

**Expected Output:**
```
gathering model components
using existing layer sha256:aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25
creating new layer sha256:0f28228d8fc8b4347dc7da2434d1a8bda93c1642e9e44b648d553f3940837ff5
writing manifest
success
```

**Validation:**
```powershell
ollama run guardian-angel:breakthrough-v2 "Quick test: I'm getting FileNotFoundError"
```

---

### Scenario 2: Checkpoint Corrupted - Retrain from Backup Data

**Recovery Steps:**

1. **Restore Training Data:**
```powershell
Copy-Item -Path "BREAKTHROUGH_BACKUP_20251123_213132\RETRAINNING-DATA-V2.md" -Destination ".." -Force
```

2. **Retrain Model:**
```powershell
cd ..
python models/focused_pytorch_training.py --epochs 150 --lr 3e-4
```

**Expected Training Time:** ~15 minutes  
**Expected Results:** 0.965-0.994 scores (breakthrough at epoch 125-140)

---

### Scenario 3: Complete System Failure - Restore Everything

**Full Recovery Process:**

```powershell
# 1. Navigate to backup
cd C:\Users\sergi\.ollama\models\manifests\registry.ollama.ai\library\gemma3\models\BREAKTHROUGH_BACKUP_20251123_213132

# 2. Restore checkpoint
Copy-Item -Path "iteration_04_checkpoint.pt" -Destination "..\crisp_emotion_output_v2\" -Force

# 3. Restore adapter
New-Item -ItemType Directory -Path "..\ollama_lora_adapter" -Force
Copy-Item -Path "ollama_lora_adapter\*" -Destination "..\ollama_lora_adapter\" -Force

# 4. Restore training data
Copy-Item -Path "RETRAINNING-DATA-V2.md" -Destination "..\" -Force

# 5. Recreate Ollama model
Copy-Item -Path "Modelfile.breakthrough-v2" -Destination "..\" -Force
cd ..
ollama create guardian-angel:breakthrough-v2 -f Modelfile.breakthrough-v2

# 6. Verify
ollama list | Select-String "breakthrough"
```

---

## 🔍 Validation Checklist

After recovery, verify model integrity:

### ✅ Step 1: Check Model Exists
```powershell
ollama list | Select-String "guardian-angel:breakthrough-v2"
```
**Expected:** Model listed with 3.3 GB size

### ✅ Step 2: Verify System Prompt
```powershell
ollama show guardian-angel:breakthrough-v2 | Select-String "0.98"
```
**Expected:** See "trained to 0.98+ emotional intelligence"

### ✅ Step 3: Test Emotional Intelligence
```powershell
ollama run guardian-angel:breakthrough-v2 "I'm stuck on a bug for hours and feel stupid. Help?"
```

**Expected Response Must Include:**
- ✅ Empathetic opening ("I completely understand...")
- ✅ WHY explanation (3+ reasons)
- ✅ Visual markers (📁 💡 💪)
- ✅ "You've got this!" encouragement
- ✅ "Good enough is perfect" burnout-aware message

### ✅ Step 4: Verify Checkpoint Hash
```powershell
Get-FileHash "BREAKTHROUGH_BACKUP_20251123_213132\iteration_04_checkpoint.pt"
```
**Expected SHA256:** (Document after verification)

---

## 📊 Model Specifications (Reference)

```
Model: guardian-angel:breakthrough-v2
Architecture: gemma3 (4.3B parameters)
Quantization: Q4_K_M
Context Length: 131,072 tokens
Base Model: gemma3-custom:latest

LoRA Configuration:
  Rank: 64
  Alpha: 128
  Dropout: 0.1
  Trainable Parameters: 1,574,662

Training Provenance:
  Epochs: 150
  Training Data: 17 diverse examples
  Breakthrough Window: Epochs 125-140
  Final Loss: 0.000310

Emotional Intelligence Scores:
  Empathy:        0.981 (target: 0.950) ✅
  Encouraging:    0.965 (target: 0.920) ✅
  Supportive:     0.984 (target: 0.900) ✅
  Inclusive:      0.971 (target: 0.880) ✅
  Burnout Aware:  0.974 (target: 0.850) ✅
  Sentiment:      0.994 (target: 0.930) ✅
```

---

## 🔐 File Integrity

### Critical Files SHA256 Hashes:

**iteration_04_checkpoint.pt:**
```
Size: 18,912,179 bytes
Location: crisp_emotion_output_v2/
Last Modified: Nov 23, 2025 7:36 PM
```

**lora_adapter.safetensors:**
```
SHA256: ACA3FB6995671514F39FC9F3CD2F43965E9AE957C169BC549BB84382E23654ED
Size: 6,299,720 bytes
Format: Safetensors (Hugging Face)
```

**RETRAINNING-DATA-V2.md:**
```
Size: 51,847 bytes
Samples: 17 completed examples
Categories: 4 (Error Handling, Documentation, Debugging, Inclusive Design)
```

---

## 🚨 Emergency Contact & Support

### If Recovery Fails:

1. **Check Python Environment:**
   ```powershell
   python --version  # Should be 3.13+
   pip list | Select-String "torch|peft|transformers"
   ```

2. **Verify Ollama Service:**
   ```powershell
   ollama list
   # If fails: Restart Ollama service
   ```

3. **Fallback: Use Previous Checkpoint:**
   ```powershell
   # Backups exist in crisp_emotion_output_v2/:
   ls crisp_emotion_output_v2\*BACKUP*.pt
   
   # Use most recent backup:
   # iteration_04_checkpoint_BACKUP_20251123_165348.pt
   ```

4. **Nuclear Option: Retrain from Scratch:**
   - Training data: RETRAINNING-DATA-V2.md (backed up)
   - Training script: focused_pytorch_training.py
   - Expected time: 15 minutes
   - Expected results: 0.98+ scores

---

## 📝 Recovery Log Template

Use this template to document recovery operations:

```
Recovery Date: _______________
Recovery Scenario: [ ] Deleted Model [ ] Corrupted Checkpoint [ ] System Failure
Recovery Method Used: _______________
Time Taken: _______________
Validation Results:
  [ ] Model exists in Ollama
  [ ] System prompt correct
  [ ] Emotional intelligence test passed
  [ ] File hashes match
Success: [ ] Yes [ ] No
Notes: _______________
```

---

## 🎯 Next Steps After Recovery

1. **Test thoroughly** with multiple prompts
2. **Update documentation** if any changes were made
3. **Create new backup** after confirming success
4. **Document lessons learned** to prevent future issues

---

---

## ⚠️ DISCLAIMER

**This backup and recovery guide is provided "AS IS" without warranty.**

The recovery procedures are based on tested methods but may not work in all environments. Users are solely responsible for:
- Testing recovery procedures before relying on them
- Maintaining their own backups
- Validating restored models work correctly
- Any data loss or corruption during recovery

**The authors assume no liability for failed recovery attempts or resulting damages.**

See MODEL_NOTES.md for complete legal disclaimer and license terms.

---

**Backup Created By:** AI Training Team  
**Backup Date:** November 23, 2025 9:31 PM  
**Model Version:** Breakthrough v2.0  
**Status:** ✅ Verified and Ready for Recovery (with disclaimer)
