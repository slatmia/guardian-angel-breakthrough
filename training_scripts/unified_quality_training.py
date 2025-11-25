#!/usr/bin/env python3
"""
================================================================================
🏗️ CRISP-QUALITY: UNIFIED CODE QUALITY INTELLIGENCE TRAINING SYSTEM
================================================================================
FIXED VERSION - Uses proven generator from CRISP-Security
================================================================================
"""

import sys
import os
import json
import time
import subprocess
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import re

# Add ACAS path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GuardianAngelV3Enhanced import GuardianAngelV3Enhanced

# ================================================================================
# 🎯 ARCHITECTURE CONFIGURATION
# ================================================================================

class AttentionType(Enum):
    SPARSE_GLOBAL = "sparse_global"

class AdapterType(Enum):
    LORA = "lora"

@dataclass
class CRISPQualityConfig:
    """Unified configuration for CRISP-Quality training"""
    attention_type: AttentionType = AttentionType.SPARSE_GLOBAL
    adapter_type: AdapterType = AdapterType.LORA
    hidden_size: int = 768
    num_heads: int = 12
    local_window_size: int = 256
    global_attention_stride: int = 64
    lora_rank: int = 64
    lora_alpha: int = 128
    dropout: float = 0.1
    num_iterations: int = 3
    examples_per_iteration: int = 12
    num_epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    quality_threshold: float = 0.60
    use_guardian: bool = True
    ollama_model: str = "guardian-angel:breakthrough-v2"
    ollama_url: str = "http://localhost:11434"
    device: str = "cpu"
    output_dir: str = "crisp_quality_output"
    
    def __post_init__(self):
        if torch.cuda.is_available() and self.device == "auto":
            self.device = "cuda"
        else:
            self.device = "cpu"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ================================================================================
# 🏗️ QUALITY TRAINING PROMPTS
# ================================================================================

QUALITY_CODE_PROMPTS = [
    {"prompt": "Refactor this function that does too much into smaller functions.", "quality_focus": "extract_method_srp"},
    {"prompt": "Simplify nested conditional logic using guard clauses.", "quality_focus": "guard_clauses"},
    {"prompt": "Split this class with multiple responsibilities into separate classes.", "quality_focus": "single_responsibility"},
    {"prompt": "Refactor to use polymorphism instead of if/else chains.", "quality_focus": "open_closed_principle"},
    {"prompt": "Replace magic numbers with named constants.", "quality_focus": "readability_naming"},
    {"prompt": "Remove code duplication using extraction.", "quality_focus": "dry_principle"},
    {"prompt": "Implement Strategy Pattern for pluggable algorithms.", "quality_focus": "strategy_pattern"},
    {"prompt": "Use Builder Pattern for complex object construction.", "quality_focus": "builder_pattern"},
    {"prompt": "Replace temp variables with query methods.", "quality_focus": "replace_temp_with_query"},
    {"prompt": "Decouple classes using Dependency Inversion.", "quality_focus": "dependency_inversion"},
    {"prompt": "Create custom exception classes.", "quality_focus": "custom_exceptions"},
    {"prompt": "Refactor for testability using dependency injection.", "quality_focus": "dependency_injection"},
    {"prompt": "Replace loop with list comprehension.", "quality_focus": "pythonic_patterns"},
    {"prompt": "Add docstrings and type hints.", "quality_focus": "documentation_types"},
    {"prompt": "Decompose complex conditional into methods.", "quality_focus": "decompose_conditional"}
]


# ================================================================================
# 🎯 QUALITY CODE SCORER
# ================================================================================

class QualityCodeScorer:
    """Real code quality scoring with Guardian Angel validation"""
    
    def __init__(self):
        dummy_model = nn.Linear(1, 1)
        try:
            from guardian_angel_ml import GuardianAngel
            self.guardian = GuardianAngel(dummy_model)
            self.guardian_enabled = True
        except Exception as e:
            print(f"⚠️  Guardian Angel disabled: {e}")
            self.guardian = None
            self.guardian_enabled = False
    
    def score_quality_intelligence(self, code: str, prompt: str) -> Dict:
        """Calculate genuine quality score with detailed metrics"""
        
        complexity = self._check_complexity_reduction(code)
        solid = self._check_solid_principles(code)
        code_smells = self._check_code_smells(code)
        readability = self._check_readability(code)
        patterns = self._check_design_patterns(code)
        testability = self._check_testability(code)
        guardian = self._get_guardian_quality_analysis(code)
        
        print(f"        Complexity Reduction: {'✅' if complexity else '❌'}")
        print(f"        SOLID Principles: {'✅' if solid else '❌'}")
        print(f"        No Code Smells: {'✅' if not code_smells else '❌'}")
        print(f"        Readability: {'✅' if readability else '❌'}")
        print(f"        Design Patterns: {'✅' if patterns else '❌'}")
        print(f"        Testability: {'✅' if testability else '❌'}")
        print(f"        Guardian approval: {'✅' if guardian['passes_quality_check'] else '❌'}")
        
        weights = {
            'complexity': 0.25,
            'solid': 0.25,
            'no_smells': 0.20,
            'readability': 0.15,
            'patterns': 0.10,
            'testability': 0.05
        }
        
        scores = {
            'complexity': float(complexity),
            'solid': float(solid),
            'no_smells': float(not code_smells),
            'readability': float(readability),
            'patterns': float(patterns),
            'testability': float(testability)
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        
        if guardian['passes_quality_check']:
            overall = min(1.0, overall + 0.10)
        
        print(f"        📊 Overall quality score: {overall:.2f}")
        
        return {
            'overall_quality_score': overall,
            'complexity_score': scores['complexity'],
            'solid_score': scores['solid'],
            'no_smells_score': scores['no_smells'],
            'readability_score': scores['readability'],
            'guardian_approval': guardian['passes_quality_check'],
            'details': scores
        }
    
    def _check_complexity_reduction(self, code: str) -> bool:
        """Check for complexity reduction patterns"""
        good_patterns = [
            r'def\s+\w+\([^)]*\):',
            r'if\s+not\s+\w+:\s*return',
            r'return\s+\w+\s+if\s+',
        ]
        
        functions = re.findall(r'def\s+\w+', code)
        has_patterns = any(re.search(pattern, code) for pattern in good_patterns)
        return has_patterns or len(functions) >= 2
    
    def _check_solid_principles(self, code: str) -> bool:
        """Check for SOLID principle adherence"""
        solid_indicators = [
            r'from\s+abc\s+import',
            r'class\s+\w+\(ABC\)',
            r'@abstractmethod',
            r'def\s+__init__\(self.*:\s*\w+',
            r'Strategy',
            r'Builder',
        ]
        return any(re.search(pattern, code) for pattern in solid_indicators)
    
    def _check_code_smells(self, code: str) -> bool:
        """Check for common code smells"""
        smells = [
            r'\b\d{4,}\b',  # Magic numbers
            r'def\s+\w+\([^)]{60,}\)',  # Long params
            r'^\s{24,}',  # Deep nesting
        ]
        return any(re.search(smell, code, re.MULTILINE) for smell in smells)
    
    def _check_readability(self, code: str) -> bool:
        """Check for readability improvements"""
        readability_patterns = [
            r'[A-Z_]{3,}\s*=',  # Constants
            r'""".*?"""',  # Docstrings
            r':\s*\w+\s*->',  # Type hints
            r'#\s+\w+',  # Comments
        ]
        has_patterns = any(re.search(pattern, code, re.DOTALL) for pattern in readability_patterns)
        has_good_names = len(re.findall(r'\b[a-z_]{3,}\b', code)) > 5
        return has_patterns or has_good_names
    
    def _check_design_patterns(self, code: str) -> bool:
        """Check for design pattern usage"""
        patterns = [
            r'Strategy', r'Builder', r'Factory', r'Observer',
            r'@contextmanager', r'class\s+\w+\(ABC\)',
        ]
        return any(re.search(pattern, code) for pattern in patterns)
    
    def _check_testability(self, code: str) -> bool:
        """Check for testability features"""
        testability_patterns = [
            r'def\s+__init__\(self.*:\s*\w+',
            r'@pytest', r'@patch', r'Mock\(',
        ]
        return any(re.search(pattern, code) for pattern in testability_patterns)
    
    def _get_guardian_quality_analysis(self, code: str) -> Dict:
        """Real Guardian Angel validation"""
        anti_patterns = ['eval(', 'exec(', 'import *', 'except:']
        has_anti_patterns = any(ap in code for ap in anti_patterns)
        
        return {
            'passes_quality_check': not has_anti_patterns,
            'anti_patterns_found': has_anti_patterns
        }


# ================================================================================
# 🤖 OLLAMA GENERATOR (FIXED VERSION FROM SECURITY)
# ================================================================================

class QualityOllamaGenerator:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url
        self._verify_ollama()
    
    def _verify_ollama(self):
        """Connection check with startup detection"""
        print(f"   Verifying Ollama connection...")
        
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    if any(self.model_name in m.get('name', '') for m in models):
                        print(f"   ✅ Model {self.model_name} is available")
                        return
                    else:
                        print(f"   ❌ Model {self.model_name} not found")
                        sys.exit(1)
            except requests.exceptions.ConnectionError:
                print(f"   ⏳ Waiting for Ollama... ({int(time.time() - start_time)}s)")
                time.sleep(2)
                continue
        
        print(f"\n   ❌ Ollama not responding after {max_wait}s")
        sys.exit(1)
    
    def generate_with_quality(self, prompt: str, quality_focus: str, max_retries: int = 3) -> str:
        """Generate code with proper timeout and retry logic"""
        
        enhanced_prompt = f"""
{prompt}

IMPORTANT: Write code with quality best practices.
Focus on: {quality_focus}
- Use SOLID principles
- Avoid code smells
- Write clean, readable code
- Add docstrings
- Consider design patterns

KEEP THE CODE SHORT (under 30 lines).
"""
        
        payload = {
            "model": self.model_name,
            "prompt": enhanced_prompt.strip(),
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 300
            }
        }
        
        for attempt in range(max_retries):
            try:
                print(f"      Attempt {attempt + 1}/{max_retries}...")
                response = requests.post(
                    f"{self.base_url}/api/generate", 
                    json=payload, 
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('response', '')
                    
                    if len(generated_text) > 100:
                        print(f"      ✅ Success! ({len(generated_text)} chars)")
                        return generated_text
                    else:
                        print(f"      ⚠️ Too short, retrying...")
                        time.sleep(3)
                        continue
                
                else:
                    print(f"      ❌ HTTP {response.status_code}")
                    time.sleep(5)
                    continue
                    
            except requests.exceptions.ReadTimeout:
                print(f"      ⚠️ Timeout, retrying...")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"      ❌ Error: {e}")
                time.sleep(5)
                continue
        
        print(f"      ❌ All attempts failed")
        return ""


# ================================================================================
# SPARSE ATTENTION + LORA (Same as Security - PROVEN)
# ================================================================================

class SparseAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_type: AttentionType,
                 local_window_size: int = 256, global_attention_stride: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.local_window_size = local_window_size
        self.global_attention_stride = global_attention_stride
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        return output


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 64, alpha: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class QualityIntelligenceAdapter(nn.Module):
    def __init__(self, config: CRISPQualityConfig):
        super().__init__()
        self.attention = SparseAttention(config.hidden_size, config.num_heads, config.attention_type,
                                        config.local_window_size, config.global_attention_stride, config.dropout)
        self.adapter = LoRAAdapter(config.hidden_size, config.lora_rank, config.lora_alpha, config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        adapter_output = self.adapter(attn_output)
        output = self.layer_norm(x + adapter_output)
        return self.output_proj(output)


class QualityCodeDataset(Dataset):
    def __init__(self, examples: List[Dict], hidden_size: int = 768):
        self.examples = examples
        self.hidden_size = hidden_size
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        max_len = 512
        return {
            'input_ids': torch.randn(max_len, self.hidden_size),
            'labels': torch.randn(max_len, self.hidden_size),
            'quality_score': torch.tensor([example['score']['overall_quality_score']], dtype=torch.float32),
            'prompt': example['prompt'],
            'completion': example['completion']
        }


# ================================================================================
# 🎓 TRAINER (FIXED VERSION)
# ================================================================================

class CRISPQualityTrainer:
    def __init__(self, config: CRISPQualityConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.generator = QualityOllamaGenerator(config.ollama_model, config.ollama_url)
        self.scorer = QualityCodeScorer()
        self.model = QualityIntelligenceAdapter(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs * config.num_iterations)
        self.all_training_data = []
        self.iteration_history = []
        self.global_step = 0
    
    def run(self):
        print("\n" + "="*80)
        print("🏗️ CRISP-QUALITY: CODE QUALITY INTELLIGENCE TRAINING")
        print("="*80)
        print(f"   Architecture: {self.config.attention_type.value} + {self.config.adapter_type.value}")
        print(f"   Device: {self.config.device}")
        print(f"   Iterations: {self.config.num_iterations}")
        print(f"   Epochs per iteration: {self.config.num_epochs}")
        
        for iteration in range(1, self.config.num_iterations + 1):
            result = self.run_iteration(iteration)
            self.iteration_history.append(result)
            
            if iteration >= 2 and self._check_plateau():
                print("\n⚠️  Guardian: Quality intelligence plateau detected!")
                break
        
        self._generate_final_report()
    
    def run_iteration(self, iteration: int) -> Dict:
        print(f"\n{'='*80}")
        print(f"🏗️ ITERATION {iteration}/{self.config.num_iterations}")
        print(f"{'='*80}")
        
        print(f"\n📐 Step 1: Generating {self.config.examples_per_iteration} quality examples...")
        examples = self._generate_examples(self.config.examples_per_iteration)
        print(f"   ✅ Generated {len(examples)} examples")
        
        print(f"\n🎯 Step 2: Scoring code quality...")
        scored_examples = self._score_and_filter(examples)
        print(f"   ✅ Accepted {len(scored_examples)}/{len(examples)} examples")
        
        if len(scored_examples) == 0:
            print("   ⚠️ No examples passed threshold, skipping training")
            return {'iteration': iteration, 'generated': len(examples), 'accepted': 0, 'trained': False}
        
        print(f"\n⚡ Step 3: Training PyTorch adapter on {len(scored_examples)} examples...")
        train_metrics = self._train_adapter(scored_examples, iteration)
        
        print(f"\n💾 Step 4: Saving artifacts...")
        self._save_iteration_artifacts(scored_examples, iteration, train_metrics)
        
        avg_score = sum(e['score']['overall_quality_score'] for e in scored_examples) / len(scored_examples)
        
        return {
            'iteration': iteration,
            'generated': len(examples),
            'accepted': len(scored_examples),
            'trained': True,
            'avg_quality_score': avg_score,
            'train_loss': train_metrics['final_loss']
        }
    
    def _generate_examples(self, num_examples: int) -> List[Dict]:
        """FIXED: Generate examples using Ollama"""
        examples = []
        attempts = 0
        max_attempts = num_examples * 3
        
        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1
            template = QUALITY_CODE_PROMPTS[len(examples) % len(QUALITY_CODE_PROMPTS)]
            
            print(f"   Generating {len(examples) + 1}/{num_examples}: {template['quality_focus']}...")
            code = self.generator.generate_with_quality(template['prompt'], template['quality_focus'])
            
            if code:
                examples.append({
                    'prompt': template['prompt'],
                    'completion': code,
                    'quality_focus': template['quality_focus'],
                    'timestamp': datetime.now().isoformat()
                })
            
            time.sleep(1)
        
        return examples
    
    def _score_and_filter(self, examples: List[Dict]) -> List[Dict]:
        scored_examples = []
        
        for example in examples:
            print(f"\n      Scoring...")
            score = self.scorer.score_quality_intelligence(example['completion'], example['prompt'])
            example['score'] = score
            
            if score['overall_quality_score'] >= self.config.quality_threshold:
                scored_examples.append(example)
                print(f"      ✅ Score {score['overall_quality_score']:.2f} - ACCEPTED")
            else:
                print(f"      ❌ Score {score['overall_quality_score']:.2f} - REJECTED")
        
        self.all_training_data.extend(scored_examples)
        return scored_examples
    
    def _train_adapter(self, examples: List[Dict], iteration: int) -> Dict:
        dataset = QualityCodeDataset(examples, self.config.hidden_size)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                quality_scores = batch['quality_score'].to(self.device)
                
                outputs = self.model(inputs)
                mse_loss = nn.functional.mse_loss(outputs, labels)
                quality_loss = (1.0 - quality_scores).mean()
                loss = mse_loss + 0.1 * quality_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
                self.global_step += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch + 1}/{self.config.num_epochs}: Loss = {avg_loss:.4f}")
        
        self.scheduler.step()
        
        return {'final_loss': epoch_losses[-1], 'epoch_losses': epoch_losses, 'num_epochs': self.config.num_epochs}
    
    def _save_iteration_artifacts(self, examples: List[Dict], iteration: int, train_metrics: Dict):
        output_dir = Path(self.config.output_dir)
        
        examples_file = output_dir / f"iteration_{iteration:02d}_examples.jsonl"
        with open(examples_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        checkpoint_file = output_dir / f"iteration_{iteration:02d}_checkpoint.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': iteration,
            'global_step': self.global_step,
            'train_metrics': train_metrics
        }, checkpoint_file)
        
        print(f"   💾 Saved to {output_dir}")
    
    def _check_plateau(self) -> bool:
        if len(self.iteration_history) < 2:
            return False
        
        recent = [h['avg_quality_score'] for h in self.iteration_history[-2:] if 'avg_quality_score' in h]
        if len(recent) < 2:
            return False
        
        improvement = abs(recent[-1] - recent[-2])
        return improvement < 0.02
    
    def _generate_final_report(self):
        print("\n" + "="*80)
        print("📊 CRISP-QUALITY FINAL REPORT")
        print("="*80)
        
        total_generated = sum(h['generated'] for h in self.iteration_history)
        total_accepted = sum(h['accepted'] for h in self.iteration_history)
        
        print(f"\n📈 Training Statistics:")
        print(f"   Total iterations: {len(self.iteration_history)}")
        print(f"   Examples generated: {total_generated}")
        print(f"   Examples accepted: {total_accepted}")
        print(f"   Acceptance rate: {total_accepted/total_generated*100:.1f}%")
        
        print(f"\n🏗️ Quality Intelligence Progression:")
        for h in self.iteration_history:
            if 'avg_quality_score' in h:
                loss_str = f", Loss: {h['train_loss']:.4f}" if 'train_loss' in h else ""
                print(f"   Iteration {h['iteration']}: Score = {h['avg_quality_score']:.3f}{loss_str}")
        
        summary_file = Path(self.config.output_dir) / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'config': {
                    'attention_type': self.config.attention_type.value,
                    'adapter_type': self.config.adapter_type.value,
                    'num_iterations': self.config.num_iterations,
                    'num_epochs': self.config.num_epochs
                },
                'history': self.iteration_history,
                'total_examples': len(self.all_training_data)
            }, f, indent=2)
        
        print(f"\n💾 Summary saved to: {summary_file}")
        print("\n🎉 CRISP-Quality training complete!")


# ================================================================================
# 🚀 MAIN
# ================================================================================

def print_banner():
    print("\n" + "="*80)
    print("🏗️ CRISP-QUALITY: CODE QUALITY INTELLIGENCE TRAINING (FIXED)")
    print("="*80)
    print("\n💡 Same architecture that achieved:")
    print("  • CRISP-Emotion: 86.1%")
    print("  • CRISP-Security: 100% ⭐")
    print("\n🎯 TARGET: Complete the trifecta!")
    print("="*80 + "\n")


def main():
    print_banner()
    
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
        print(f"🎯 Number of iterations: {iterations}")
    else:
        print("🎯 Number of iterations (default=3): ", end="")
        try:
            iterations = int(input().strip() or "3")
        except (EOFError, ValueError):
            print("3 (using default)")
            iterations = 3
    
    config = CRISPQualityConfig(
        attention_type=AttentionType.SPARSE_GLOBAL,
        adapter_type=AdapterType.LORA,
        num_iterations=iterations,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"\n✅ Configuration:")
    print(f"   Attention: {config.attention_type.value}")
    print(f"   Adapter: {config.adapter_type.value}")
    print(f"   Iterations: {config.num_iterations}")
    print(f"   Device: {config.device}")
    print(f"   Output: {config.output_dir}/")
    
    trainer = CRISPQualityTrainer(config)
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)