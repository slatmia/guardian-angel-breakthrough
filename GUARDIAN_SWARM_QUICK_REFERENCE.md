# 🛡️ GUARDIAN SWARM TRIFECTA - QUICK REFERENCE

**Deployment Date**: November 24, 2025  
**Status**: ✅ **PRODUCTION READY - ALL MODELS VALIDATED**

---

## 🐝 DEPLOYED MODELS

### **1. Emotion Agent: gemma3-emotion-enhanced:latest**
```powershell
ollama run gemma3-emotion-enhanced:latest "Your empathy prompt here"
```

**Specifications**:
- **Model ID**: `4e518fef31dc`
- **Size**: 3.3 GB
- **Acceptance Rate**: 86.1% (31/36 examples)
- **Best Score**: 0.924
- **Specialty**: Empathy, burnout awareness, user experience
- **Use Cases**: 
  - Developer encouragement
  - Burnout support
  - Emotional intelligence in code review
  - User-friendly explanations

**Example Prompts**:
- "I'm feeling overwhelmed with this refactoring task"
- "Can you help me understand this error in a supportive way?"
- "I'm burnt out from debugging, need encouragement"

---

### **2. Security Agent: guardian-security:v1.0**
```powershell
ollama run guardian-security:v1.0 "Your security analysis prompt here"
```

**Specifications**:
- **Model ID**: `e7922f00ddf8`
- **Size**: 3.3 GB
- **Acceptance Rate**: 100% (36/36 examples) ⭐
- **Best Score**: 0.942
- **Specialty**: Security vulnerabilities, threat detection
- **Use Cases**:
  - SQL injection detection
  - XSS vulnerability analysis
  - Authentication/authorization review
  - Secure coding patterns
  - Encryption best practices

**Example Prompts**:
- "Review this authentication code for security vulnerabilities"
- "How do I prevent SQL injection in this query?"
- "Is this password hashing implementation secure?"
- "Identify security risks in this API endpoint"

---

### **3. Quality Agent: guardian-quality:v1.0**
```powershell
ollama run guardian-quality:v1.0 "Your code quality prompt here"
```

**Specifications**:
- **Model ID**: `773d3807e0f1`
- **Size**: 3.3 GB
- **Context**: 4096 tokens
- **Acceptance Rate**: 87.3% (22/24 examples)
- **Best Score**: 0.873
- **Specialty**: SOLID principles, design patterns, architecture
- **Use Cases**:
  - Code refactoring suggestions
  - SOLID principles enforcement
  - Design pattern recommendations
  - Code smell detection
  - Architecture reviews

**Example Prompts**:
- "Refactor this function to follow SOLID principles"
- "Identify code smells in this class"
- "Suggest design patterns for this problem"
- "Review this architecture for maintainability"

---

### **4. Quality Agent (Extended): guardian-quality:v1.0-6k ⭐⭐**
```powershell
ollama run guardian-quality:v1.0-6k "Your complex refactoring prompt here"
```

**Specifications**:
- **Model ID**: `19f00d1b03e2`
- **Size**: 3.3 GB
- **Context**: 6144 tokens (50% larger than v1.0)
- **Quality Score**: 1.00 (validated at 6K context) ⭐
- **Performance**: 75.9s (warm model, optimal)
- **RAM Usage**: 4.08 GB
- **Specialty**: Large-scale refactoring, complex architectural analysis
- **Use Cases**:
  - Monolithic function decomposition
  - Large codebase refactoring
  - Complex architectural patterns
  - Multi-class SOLID analysis

**Example Prompts**:
- "Refactor this 1000-line monolithic OrderProcessor class"
- "Analyze this entire module for SOLID violations"
- "Suggest comprehensive architectural improvements"

**Performance Notes**:
- Cold start: ~157s (first run after model unload)
- Warm model: ~76s (optimal, within 60-90s target)
- Use `OLLAMA_KEEP_ALIVE=10m` to keep warm

---

## 🎯 ROUTING GUIDE

Use the **Guardian Router** for automatic agent selection:

```powershell
.\guardian_router_optimized.ps1 "Your prompt here"
```

**Routing Logic** (keyword-based, 0.01s classification):
- **Quality** keywords: refactor, SOLID, design pattern, code smell, maintainability
- **Security** keywords: SQL injection, XSS, authentication, vulnerability, encryption
- **Emotion** keywords: burnout, stressed, frustrated, overwhelmed, tired

**Router Features**:
- ⚡ 0.01s classification (keyword matching)
- 💾 Caching (repeat prompts instant routing)
- 🛡️ Optional Guardian Angel validation
- 📊 Performance tracking

---

## 📊 PERFORMANCE COMPARISON

| Model | Context | Acceptance | Score | Warm Latency | RAM | Best For |
|-------|---------|------------|-------|--------------|-----|----------|
| **Emotion** | 4K | 86.1% | 0.924 | ~35s | 4.0 GB | User empathy |
| **Security** | 4K | 100% ⭐ | 0.942 | ~25s | 4.0 GB | Vulnerabilities |
| **Quality** | 4K | 87.3% | 0.873 | ~13s | 4.05 GB | SOLID/patterns |
| **Quality-6K** | 6K | 91.7% | 1.00 ⭐ | ~76s | 4.08 GB | Large refactoring |

---

## 🚀 QUICK START COMMANDS

### **Test All Models**
```powershell
# Emotion
ollama run gemma3-emotion-enhanced "I'm stressed about code quality"

# Security
ollama run guardian-security:v1.0 "How do I prevent SQL injection?"

# Quality (4K)
ollama run guardian-quality:v1.0 "Refactor this for SOLID principles"

# Quality (6K)
ollama run guardian-quality:v1.0-6k "Refactor this monolithic class"
```

### **Router Usage**
```powershell
# Automatic routing
.\guardian_router_optimized.ps1 "Refactor this authentication code"

# With validation
.\guardian_router_optimized.ps1 "Help with burnout" -Validate

# Verbose mode
.\guardian_router_optimized.ps1 "Fix SQL injection" -VerboseOutput
```

### **Pre-warm Models (Production)**
```powershell
# Keep models loaded
[System.Environment]::SetEnvironmentVariable('OLLAMA_KEEP_ALIVE', '10m', 'User')

# Pre-warm all agents
ollama run gemma3-emotion-enhanced "test" > $null
ollama run guardian-security:v1.0 "test" > $null
ollama run guardian-quality:v1.0 "test" > $null
```

---

## 🛠️ SYSTEM REQUIREMENTS

**Minimum**:
- CPU: 8 cores (Ryzen 7 2700 or equivalent)
- RAM: 16 GB (12 GB available for models)
- Storage: 15 GB for all models
- OS: Windows 10/11, PowerShell 5.1+

**Optimal**:
- CPU: 8+ cores @ 3.0+ GHz
- RAM: 32 GB (for running multiple agents simultaneously)
- Storage: SSD for faster model loading
- Ollama: v0.12.11 or later

**Current Configuration**:
- ✅ Ryzen 7 2700 (8 cores)
- ✅ 16 GB RAM (12 GB available)
- ✅ Ollama 0.12.11
- ✅ CPU Performance: 97.1% (no throttling)

---

## 📋 MODEL MAINTENANCE

### **Update Models**
```powershell
# Pull latest base model
ollama pull guardian-angel:breakthrough-v2

# Recreate specialized models
ollama create guardian-quality:v1.0 -f Modelfile.guardian-quality
ollama create guardian-security:v1.0 -f Modelfile.guardian-security
```

### **Check Model Status**
```powershell
# List all Guardian models
ollama list | Select-String "guardian|emotion"

# Check RAM usage
Get-Process -Name ollama | Select @{N='MemoryGB';E={[math]::Round($_.WorkingSet64/1GB,2)}}

# Monitor CPU performance
Get-Counter "\Processor Information(_Total)\% Processor Performance"
```

### **Clean Up Old Models**
```powershell
# Remove older versions (keep backups!)
ollama rm guardian-quality:v0.9
ollama rm guardian-angel:v1
```

---

## 🏆 CERTIFICATION STATUS

**Guardian Swarm Trifecta**: ✅ **TROPHY POTATO CERTIFIED**

- ✅ All models deployed and validated
- ✅ Performance optimized (0.01s routing)
- ✅ Quality scores meet/exceed thresholds
- ✅ 6K context validated (score: 1.0)
- ✅ RAM usage within safe limits
- ✅ No thermal throttling detected
- ✅ Production ready with comprehensive documentation

**Total Training Time**: ~6 hours  
**Total System Size**: 13.2 GB (4 models)  
**Deployment Status**: OPERATIONAL  

**The Guardian Swarm is ready to serve.** 🛡️🐝

---

**Last Updated**: November 24, 2025, 11:16 PM  
**Version**: Production 1.0  
**Maintainer**: ACAS Guardian Angel System
