import sys
import os
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural'))

try:
    from guardian_swarm import ConsciousSwarm
    import torch
    SWARM_OK = True
except:
    SWARM_OK = False

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

class Bridge:
    def __init__(self):
        self.swarm = ConsciousSwarm(hidden_dim=64) if SWARM_OK else None
        self.history = []
        if self.swarm:
            print(f"Swarm: {self.swarm.count_parameters():,} params")
    
    def analyze(self, text):
        if not self.swarm:
            return {"emotion": 0.5, "security": 0.5, "quality": 0.5}
        words = text.lower().split()
        features = torch.zeros(1, 10)
        pos = {'good', 'great', 'happy', 'love', 'thank', 'amazing'}
        neg = {'bad', 'terrible', 'hate', 'angry', 'awful', 'frustrated'}
        threats = {'drop', 'delete', 'select', 'union', 'script'}
        features[0, 0] = len(words) / 50.0
        features[0, 1] = sum(1 for w in words if w in pos) / max(1, len(words))
        features[0, 2] = sum(1 for w in words if w in neg) / max(1, len(words))
        features[0, 3] = sum(1 for w in words if w in threats) / max(1, len(words))
        features[0, 7] = 1.0 if any(w in text.lower() for w in ['select', 'drop', '--']) else 0.0
        features[0, 8] = 1.0 if '<script' in text.lower() else 0.0
        with torch.no_grad():
            r = self.swarm.dance(features, num_rounds=2)
        return {
            "emotion": (r['emotion']['output'] + 1) / 2,
            "security": r['security']['output'],
            "quality": (r['quality']['output'] + 1) / 2
        }
    
    def chat(self, msg):
        a = self.analyze(msg)
        emo = "negative" if a["emotion"] < 0.4 else "positive" if a["emotion"] > 0.6 else "neutral"
        sec = "HIGH RISK" if a["security"] > 0.7 else "low" if a["security"] < 0.3 else "moderate"
        
        system = f"You are Guardian. User emotion: {emo}. Security: {sec}. Be direct, no emojis."
        if a["emotion"] < 0.4:
            system += " User is frustrated - acknowledge briefly, then help."
        
        self.history.append({"role": "user", "content": msg})
        messages = [{"role": "system", "content": system}] + self.history[-6:]
        
        try:
            payload = json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                reply = json.loads(resp.read())["message"]["content"]
                self.history.append({"role": "assistant", "content": reply})
                return a, reply
        except Exception as e:
            return a, f"[Ollama error: {e}]"

if __name__ == "__main__":
    print("="*60)
    print("GUARDIAN + OLLAMA BRIDGE")
    print("="*60)
    b = Bridge()
    print(f"Model: {MODEL}")
    print("Commands: 'quit' to exit")
    print("-"*60)
    while True:
        try:
            user = input("\nYOU: ").strip()
        except:
            break
        if not user or user.lower() in ('quit', 'q', 'exit'):
            break
        analysis, reply = b.chat(user)
        print(f"\n[Emo:{analysis['emotion']:.2f} Sec:{analysis['security']:.2f} Qual:{analysis['quality']:.2f}]")
        print(f"\nGUARDIAN: {reply}")
