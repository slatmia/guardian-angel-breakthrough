import sys, os, json, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural'))
from guardian_swarm import ConsciousSwarm
import torch

OLLAMA = "http://localhost:11434"
MODEL = "gemma3:4b"

def ask_ollama(system, user, context="", timeout=60, retries=2):
    for attempt in range(retries):
        try:
            messages = [{"role": "system", "content": system}]
            if context:
                messages.append({"role": "system", "content": f"CONTEXT: {context}"})
            messages.append({"role": "user", "content": user})
            
            payload = json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode()
            req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"[Retry {attempt+1}/{retries}...]")
                continue
            return f"[Error: {e}]"

swarm = ConsciousSwarm(hidden_dim=64)
print(f"Guardian Swarm: {swarm.count_parameters():,} params")
print("Multi-agent conversation with cross-references\n")

while True:
    try:
        user = input("\nYOU: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n[Exiting gracefully...]")
        break
    except Exception as e:
        print(f"\n[Input error: {e}] - retrying...")
        continue
    
    if not user or user.lower() in ('quit', 'q'):
        break

    # Analyze with swarm
    words = user.lower().split()
    features = torch.zeros(1, 10)
    pos = {'good', 'great', 'happy', 'love', 'thank', 'amazing'}
    neg = {'bad', 'terrible', 'hate', 'angry', 'awful', 'frustrated'}
    threats = {'drop', 'delete', 'select', 'union', 'script', 'attack'}
    
    features[0, 0] = len(words) / 50.0
    features[0, 1] = sum(1 for w in words if w in pos) / max(1, len(words))
    features[0, 2] = sum(1 for w in words if w in neg) / max(1, len(words))
    features[0, 3] = sum(1 for w in words if w in threats) / max(1, len(words))
    features[0, 7] = 1.0 if any(w in user.lower() for w in ['select', 'drop', '--']) else 0.0
    features[0, 8] = 1.0 if '<script' in user.lower() else 0.0
    
    with torch.no_grad():
        r = swarm.dance(features, num_rounds=2)

    emo = (r['emotion']['output'] + 1) / 2
    sec = r['security']['output']
    qual = (r['quality']['output'] + 1) / 2
    consensus = r['consensus']
    
    print(f"\n{'='*60}")
    print(f"[SWARM ANALYSIS] Emo:{emo:.2f} Sec:{sec:.2f} Qual:{qual:.2f} Consensus:{consensus}")
    print(f"{'='*60}\n")

    # ROUND 1: Initial responses
    emo_sys = f"You are EMOTION (score: {emo:.2f}). Be empathetic. One sentence."
    emo_response = ask_ollama(emo_sys, user)
    print(f"[EMOTION {emo:.2f}]: {emo_response}")

    sec_sys = f"You are SECURITY (threat: {sec:.2f}). Be protective. One sentence."
    sec_response = ask_ollama(sec_sys, user)
    print(f"\n[SECURITY {sec:.2f}]: {sec_response}")

    qual_sys = f"You are QUALITY (score: {qual:.2f}). Be analytical. One sentence."
    qual_response = ask_ollama(qual_sys, user)
    print(f"\n[QUALITY {qual:.2f}]: {qual_response}")
    
    # ROUND 2: They respond to EACH OTHER
    print(f"\n--- AGENTS DEBATE ---")
    
    # SECURITY challenges EMOTION
    sec_challenge = f"You are SECURITY. EMOTION said: '{emo_response}'. Challenge their viewpoint if it's too soft. Be direct."
    sec_debate = ask_ollama(sec_challenge, "Respond to EMOTION", "")
    print(f"\n[SECURITY → EMOTION]: {sec_debate}")
    
    # EMOTION defends or adjusts
    emo_defend = f"You are EMOTION. SECURITY challenged you: '{sec_debate}'. Defend your position or acknowledge their point. Be honest."
    emo_counter = ask_ollama(emo_defend, "Respond to SECURITY", "")
    print(f"\n[EMOTION → SECURITY]: {emo_counter}")
    
    # QUALITY weighs in on the argument
    qual_judge = f"You are QUALITY. SECURITY and EMOTION are arguing. Security: '{sec_debate}' | Emotion: '{emo_counter}'. Who's right?"
    qual_verdict = ask_ollama(qual_judge, "Judge the debate", "")
    print(f"\n[QUALITY judges]: {qual_verdict}")
    
    # Final consensus check
    if not consensus:
        print(f"\n[HIGH VARIANCE] Agents still disagree - no clear winner")
    else:
        print(f"\n[CONVERGENCE] Debate resolved, consensus emerging")
    
    print(f"\n{'='*60}")
