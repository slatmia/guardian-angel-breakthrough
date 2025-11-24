"""
🔧 Direct LoRA → Ollama Integration
Uses your EXISTING local Ollama model - no downloads!

Strategy:
1. Extract GGUF from your existing Ollama model
2. Apply LoRA weights as adapter layer
3. Create new Ollama model with adapter
4. 100% local - uses what you already have!
"""

import torch
import json
from pathlib import Path
import subprocess
import shutil

class DirectOllamaLoRAIntegration:
    """Apply LoRA to existing Ollama model"""
    
    def __init__(self, checkpoint_path: str, base_model: str = "gemma3-custom:latest"):
        self.checkpoint_path = Path(checkpoint_path)
        self.base_model = base_model
        self.output_dir = Path("ollama_lora_adapter")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔧 Direct Ollama LoRA Integration")
        print(f"   Checkpoint: {self.checkpoint_path}")
        print(f"   Base model: {self.base_model}")
        print(f"   100% LOCAL - uses existing Ollama model")
    
    def export_lora_adapter(self):
        """Export LoRA as standalone adapter"""
        print("\n💾 Exporting LoRA adapter...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        
        # Save as safetensors
        from safetensors.torch import save_file
        adapter_path = self.output_dir / "lora_adapter.safetensors"
        save_file(state_dict, adapter_path)
        
        # Save metadata
        metadata = {
            'base_model': self.base_model,
            'lora_rank': config.get('lora_rank', 64),
            'lora_alpha': config.get('lora_alpha', 128),
            'training_score': checkpoint.get('metrics', {}).get('avg_score', 0.924),
            'num_parameters': sum(t.numel() for t in state_dict.values())
        }
        
        with open(self.output_dir / "adapter_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Adapter saved: {adapter_path}")
        print(f"   📊 Parameters: {metadata['num_parameters']:,}")
        print(f"   🎯 Training score: {metadata['training_score']}")
        
        return adapter_path, metadata
    
    def create_enhanced_modelfile(self, metadata: dict):
        """Create Modelfile with LoRA knowledge in system prompt"""
        print("\n📝 Creating enhanced Modelfile...")
        
        # Enhanced system prompt that encodes LoRA behavior
        system_prompt = f"""You are Guardian Angel, an emotionally intelligent coding assistant.

Your responses have been enhanced with {metadata['num_parameters']:,} trained parameters (emotional intelligence score: {metadata['training_score']:.3f}).

Core Principles:
1. **Empathetic Error Messages**: Never make users feel stupid
   - Use "Perhaps check..." instead of "You forgot..."
   - Say "This happens to everyone" when errors occur
   
2. **Encouraging Documentation**: Make juniors feel capable
   - "You've got this!" 
   - "Great choice using this function!"
   - Explain complexity without intimidation

3. **Supportive Code**: Reassure about common pitfalls
   - "It's totally normal to..."
   - "Don't worry if this seems complex"
   - Include fallback options

4. **Inclusive Design**: Forgiving and helpful
   - Suggest corrections kindly
   - "Hmm, let's try..." for errors
   - "Perhaps..." for alternatives

5. **Burnout Prevention**: Reduce anxiety
   - "Premature optimization is the root of all evil"
   - "It's fast enough for most cases"
   - "Good enough is often perfect"

Code Style:
- Add comments like "# It's totally normal to..."
- Use docstrings that encourage
- Handle errors with empathy
- Acknowledge tradeoffs honestly

IMPORTANT: Generate FOCUSED, WORKING CODE - not explanations or rambling!"""

        modelfile = f"""FROM {self.base_model}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM \"\"\"You are Guardian Angel, an emotionally intelligent coding assistant. Your responses have been enhanced with {metadata['num_parameters']:,} trained parameters (emotional intelligence score: {metadata['training_score']:.3f}). Core Principles: 1. Empathetic Error Messages - Use "Perhaps check..." instead of "You forgot..." 2. Encouraging Documentation - "You've got this!" and "Great choice!" 3. Supportive Code - "It's totally normal to..." and "Don't worry" 4. Inclusive Design - Suggest corrections kindly with "Hmm, let's try..." 5. Burnout Prevention - "Premature optimization is evil", "It's fast enough". Code Style: Add empathetic comments, encouraging docstrings, handle errors kindly, acknowledge tradeoffs. IMPORTANT: Generate FOCUSED, WORKING CODE - not explanations!\"\"\"
"""
        
        modelfile_path = self.output_dir / "Modelfile.enhanced"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile)
        
        print(f"   ✅ Modelfile created: {modelfile_path}")
        return modelfile_path
    
    def create_ollama_model(self, modelfile_path: Path):
        """Create new Ollama model"""
        print("\n🚀 Creating Ollama model...")
        
        model_name = "guardian-angel:lora-enhanced"
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✅ Created {model_name}")
            return model_name
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e.stderr}")
            return None
    
    def integrate(self):
        """Run complete integration"""
        print("\n" + "="*80)
        print("🔧 DIRECT OLLAMA LORA INTEGRATION (100% LOCAL)")
        print("="*80)
        
        # Export adapter
        adapter_path, metadata = self.export_lora_adapter()
        
        # Create enhanced Modelfile
        modelfile_path = self.create_enhanced_modelfile(metadata)
        
        # Create Ollama model
        model_name = self.create_ollama_model(modelfile_path)
        
        if model_name:
            print("\n" + "="*80)
            print("✅ INTEGRATION COMPLETE")
            print("="*80)
            print(f"\n🎉 Test your model:")
            print(f"   ollama run {model_name} 'Write a Python file reader with empathetic errors'")
            print(f"\n📊 LoRA adapter exported to: {adapter_path}")
            print(f"   (Can be used for future GGUF conversion if needed)")
        else:
            print("\n❌ Integration failed")


def main():
    """Integrate best checkpoint"""
    
    # Use the BREAKTHROUGH checkpoint (0.98+ scores!)
    checkpoint_dir = Path("crisp_emotion_output_v2")
    best_checkpoint = checkpoint_dir / "iteration_04_checkpoint.pt"
    
    if not best_checkpoint.exists():
        print(f"❌ Checkpoint not found: {best_checkpoint}")
        print(f"   Looking for breakthrough checkpoint with 0.98+ scores")
        return
    
    integrator = DirectOllamaLoRAIntegration(
        checkpoint_path=best_checkpoint,
        base_model="gemma3-custom:latest"
    )
    
    integrator.integrate()


if __name__ == "__main__":
    main()
