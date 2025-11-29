import sys, os, json, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural'))
from guardian_swarm import ConsciousSwarm
import torch

OLLAMA = "http://localhost:11434"
MODEL = "gemma3:4b"

def ask_ollama(system, user):
    try:
        payload = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": False}).encode()
        req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["message"]["content"]
    except Exception as e:
        return f"[Error: {e}]"

swarm = ConsciousSwarm(hidden_dim=64)
print(f"Guardian Swarm: {swarm.count_parameters():,} params\n")

while True:
    user = input("\nYOU: ").strip()
    if not user or user.lower() in ('quit', 'q'):
        break
    
    words = user.lower().split()
    features = torch.zeros(1, 10)
    features[0, 0] = len(words) / 50.0
    with torch.no_grad():
        r = swarm.dance(features, num_rounds=2)
    
    emo = (r['emotion']['output'] + 1) / 2
    sec = r['security']['output']
    qual = (r['quality']['output'] + 1) / 2
    
    print(f"\n{'='*50}")
    
    # EMOTION speaks
    emo_sys = f"You are the EMOTION agent. Your reading is {emo:.2f}. Be empathetic, warm. One sentence."
    print(f"[EMOTION {emo:.2f}]: {ask_ollama(emo_sys, user)}")
    
    # SECURITY speaks
    sec_sys = f"You are the SECURITY agent. Threat level: {sec:.2f}. Be vigilant, protective. One sentence."
    print(f"[SECURITY {sec:.2f}]: {ask_ollama(sec_sys, user)}")
    
    # QUALITY speaks
    qual_sys = f"You are the QUALITY agent. Quality score: {qual:.2f}. Be analytical, precise. One sentence."
    print(f"[QUALITY {qual:.2f}]: {ask_ollama(qual_sys, user)}")
    
    print(f"{'='*50}")
