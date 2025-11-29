"""
Conversational Interface for Conscious Guardian Swarm

Enables agents to:
1. Talk to the user about their analysis
2. Discuss with each other using Theory of Mind
3. Express confidence and uncertainty naturally
4. Share insights from their specialization

Works standalone (template-based) or with LLM backends (OpenAI/Ollama).
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .conscious_swarm import ConsciousSwarm


def encode_text_to_features(text: str) -> torch.Tensor:
    """
    Simple text-to-features encoder for conversational analysis.
    
    Maps text to the 10-dimensional feature space:
    [sentiment, gratitude, frustration, threat_indicators, code_quality,
     educational_value, security_risk, helpfulness, toxicity, satisfaction]
    """
    text_lower = text.lower()
    
    positive_words = ['thank', 'great', 'amazing', 'awesome', 'love', 'excellent', 'perfect', 'helpful', 'appreciate']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'useless', 'stupid', 'worst', 'angry', 'frustrated']
    threat_words = ['sql', 'injection', 'hack', 'exploit', 'password', 'admin', 'delete', 'drop', 'attack']
    code_words = ['code', 'function', 'class', 'bug', 'error', 'debug', 'variable', 'algorithm', 'compile']
    toxic_words = ['idiot', 'dumb', 'stupid', 'hate', 'kill', 'die', 'worthless']
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    threat_count = sum(1 for w in threat_words if w in text_lower)
    code_count = sum(1 for w in code_words if w in text_lower)
    toxic_count = sum(1 for w in toxic_words if w in text_lower)
    
    sentiment = min(1.0, pos_count * 0.3) - min(1.0, neg_count * 0.3)
    gratitude = min(1.0, pos_count * 0.25)
    frustration = min(1.0, neg_count * 0.25)
    threat_indicators = min(1.0, threat_count * 0.3)
    code_quality = min(1.0, code_count * 0.2) if code_count > 0 else 0.5
    educational_value = 0.5 + min(0.3, code_count * 0.1)
    security_risk = min(1.0, threat_count * 0.35)
    helpfulness = 0.5 + sentiment * 0.3
    toxicity = min(1.0, toxic_count * 0.4)
    satisfaction = max(0, min(1, 0.5 + sentiment * 0.4 - frustration * 0.3))
    
    features = torch.tensor([
        sentiment, gratitude, frustration, threat_indicators, code_quality,
        educational_value, security_risk, helpfulness, toxicity, satisfaction
    ], dtype=torch.float32)
    
    return features


@dataclass
class AgentMessage:
    """A message from an agent."""
    agent_name: str
    content: str
    confidence: float
    emotion_state: str
    to_agent: Optional[str] = None  # None = to user, otherwise to another agent


class ConversationalSwarm:
    """
    Wraps ConsciousSwarm with natural language conversation abilities.
    
    Each agent has a distinct voice based on their specialization:
    - Emotion: Empathetic, focuses on feelings and tone
    - Security: Cautious, focuses on risks and safety
    - Quality: Analytical, focuses on correctness and standards
    """
    
    EMOTION_TEMPLATES = {
        'positive': [
            "I sense genuine positivity here. The emotional tone feels {conf}.",
            "This resonates warmly with me. I'm {conf} about the positive sentiment.",
            "There's real appreciation expressed here. I feel {conf} in my reading.",
        ],
        'negative': [
            "I'm picking up some frustration. I'm {conf} about this reading.",
            "The emotional undertone concerns me. {conf} there's distress here.",
            "I sense tension in these words. My {conf} assessment shows negativity.",
        ],
        'neutral': [
            "The emotional signal is balanced. I'm {conf} it's neutral.",
            "I don't detect strong feelings either way. {conf} neutral reading.",
            "Emotionally, this is fairly measured. I'm {conf} about that.",
        ],
    }
    
    SECURITY_TEMPLATES = {
        'high_risk': [
            "I'm detecting potential security concerns. I'm {conf} about this risk.",
            "Red flags are appearing in my analysis. {conf} caution is warranted.",
            "My security protocols are triggering. I'm {conf} this needs attention.",
        ],
        'low_risk': [
            "I don't see immediate security threats. {conf} this appears safe.",
            "My risk assessment comes back low. I'm {conf} we're in the clear.",
            "No concerning patterns detected. {conf} this looks secure.",
        ],
        'medium_risk': [
            "There are some elements worth watching. I'm {conf} about moderate risk.",
            "Not critical, but I'd recommend caution. {conf} assessment.",
            "Mixed signals on security. I'm {conf} it deserves attention.",
        ],
    }
    
    QUALITY_TEMPLATES = {
        'high': [
            "This meets high standards. I'm {conf} about the quality here.",
            "Excellent work - well-structured and clear. {conf} positive assessment.",
            "The quality markers are strong. I'm {conf} this is solid.",
        ],
        'low': [
            "I'm seeing quality concerns. {conf} this needs improvement.",
            "The standards aren't quite met here. I'm {conf} about that.",
            "Quality markers are missing. {conf} assessment of deficiencies.",
        ],
        'medium': [
            "Adequate quality with room for improvement. I'm {conf}.",
            "Meets basic standards but could be better. {conf} assessment.",
            "Acceptable quality overall. I'm {conf} about this middle ground.",
        ],
    }
    
    PEER_DISCUSSION_TEMPLATES = {
        'emotion_to_security': [
            "Security, I'm sensing {emotion_state}. Does that affect your risk reading?",
            "I'm picking up {emotion_state} vibes. How does that look from a security lens?",
            "The emotional context here is {emotion_state}. What's your take, Security?",
        ],
        'emotion_to_quality': [
            "Quality, the emotional tone is {emotion_state}. Does that impact your standards?",
            "I'm reading {emotion_state} here. How does that factor into quality?",
            "Feeling-wise, this is {emotion_state}. What's your quality perspective?",
        ],
        'security_to_emotion': [
            "Emotion, I'm seeing {risk_level} risk. How does that land emotionally?",
            "My risk sensors show {risk_level}. What's the emotional impact, Emotion?",
            "From a security standpoint, this is {risk_level}. How do users feel about that?",
        ],
        'security_to_quality': [
            "Quality, my risk assessment is {risk_level}. Does that affect your standards?",
            "I'm flagging {risk_level} security. How does that impact quality, Quality?",
            "Security-wise: {risk_level}. What's the quality angle here?",
        ],
        'quality_to_emotion': [
            "Emotion, the quality here is {quality_level}. How does that feel to users?",
            "My standards say this is {quality_level}. What's the emotional read, Emotion?",
            "Quality assessment: {quality_level}. How does that land emotionally?",
        ],
        'quality_to_security': [
            "Security, I'm rating this as {quality_level} quality. Any security implications?",
            "The quality markers show {quality_level}. Does that trigger any risk flags?",
            "From a quality view: {quality_level}. What's your security take?",
        ],
    }
    
    AGREEMENT_TEMPLATES = [
        "I agree with {peer}'s assessment. My analysis aligns.",
        "That matches what I'm seeing. Good call, {peer}.",
        "{peer} is right. My readings support that conclusion.",
    ]
    
    DISAGREEMENT_TEMPLATES = [
        "Interesting - I'm seeing it differently than {peer}. Let me explain...",
        "I respectfully disagree with {peer}. From my perspective...",
        "{peer}, I have a different read. Here's what I'm noticing...",
    ]
    
    def __init__(self, swarm: Optional[ConsciousSwarm] = None, hidden_dim: int = 64):
        """Initialize with an existing swarm or create a new one."""
        if swarm is None:
            self.swarm = ConsciousSwarm(input_dim=10, hidden_dim=hidden_dim)
        else:
            self.swarm = swarm
        
        self.conversation_history: List[AgentMessage] = []
        self.last_analysis: Optional[Dict] = None
    
    def _confidence_word(self, conf: float) -> str:
        """Convert confidence score to natural language."""
        if conf > 0.9:
            return "very confident"
        elif conf > 0.7:
            return "fairly confident"
        elif conf > 0.5:
            return "somewhat confident"
        else:
            return "uncertain but"
    
    def _emotion_state(self, score: float) -> str:
        """Convert emotion score to state description."""
        if score > 0.3:
            return "positive"
        elif score < -0.3:
            return "negative"
        return "neutral"
    
    def _risk_level(self, score: float) -> str:
        """Convert security score to risk level."""
        if score > 0.7:
            return "high_risk"
        elif score > 0.3:
            return "medium_risk"
        return "low_risk"
    
    def _quality_level(self, score: float) -> str:
        """Convert quality score to level."""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        return "low"
    
    def _to_float(self, val) -> float:
        """Convert tensor or float to float."""
        if isinstance(val, torch.Tensor):
            return val.mean().item() if val.numel() > 1 else val.item()
        return float(val)
    
    def analyze(self, text: str) -> Dict:
        """Analyze text and store results for conversation."""
        features = encode_text_to_features(text)
        result = self.swarm.dance(features, num_rounds=3)
        
        e_score = self._to_float(result['emotion']['output'])
        e_confidence = self._to_float(result['emotion']['confidence'])
        s_score = self._to_float(result['security']['output'])
        s_confidence = self._to_float(result['security']['confidence'])
        q_score = self._to_float(result['quality']['output'])
        q_confidence = self._to_float(result['quality']['confidence'])
        
        self.last_analysis = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'emotion': {
                'score': e_score,
                'confidence': e_confidence,
                'state': self._emotion_state(e_score),
            },
            'security': {
                'score': s_score,
                'confidence': s_confidence,
                'level': self._risk_level(s_score),
            },
            'quality': {
                'score': q_score,
                'confidence': q_confidence,
                'level': self._quality_level(q_score),
            },
            'consensus': result['consensus'],
            'consciousness': self.swarm.get_psychological_profile(),
        }
        return self.last_analysis
    
    def _select_template(self, templates: List[str], seed: int = 0) -> str:
        """Select a template with some variation."""
        import random
        random.seed(seed)
        return random.choice(templates)
    
    def get_agent_response(self, agent: str, seed: int = 0) -> AgentMessage:
        """Get a natural language response from a specific agent."""
        if self.last_analysis is None:
            return AgentMessage(
                agent_name=agent.capitalize(),
                content="I haven't analyzed anything yet. Give me some text to work with!",
                confidence=0.5,
                emotion_state="neutral"
            )
        
        import random
        random.seed(seed)
        
        if agent == 'emotion':
            state = self.last_analysis['emotion']['state']
            conf = self.last_analysis['emotion']['confidence']
            templates = self.EMOTION_TEMPLATES[state]
            content = self._select_template(templates, seed).format(
                conf=self._confidence_word(conf)
            )
            return AgentMessage(
                agent_name="Emotion",
                content=content,
                confidence=conf,
                emotion_state=state
            )
        
        elif agent == 'security':
            level = self.last_analysis['security']['level']
            conf = self.last_analysis['security']['confidence']
            templates = self.SECURITY_TEMPLATES[level]
            content = self._select_template(templates, seed).format(
                conf=self._confidence_word(conf)
            )
            return AgentMessage(
                agent_name="Security",
                content=content,
                confidence=conf,
                emotion_state=level
            )
        
        elif agent == 'quality':
            level = self.last_analysis['quality']['level']
            conf = self.last_analysis['quality']['confidence']
            templates = self.QUALITY_TEMPLATES[level]
            content = self._select_template(templates, seed).format(
                conf=self._confidence_word(conf)
            )
            return AgentMessage(
                agent_name="Quality",
                content=content,
                confidence=conf,
                emotion_state=level
            )
        
        raise ValueError(f"Unknown agent: {agent}")
    
    def get_team_discussion(self, seed: int = 0) -> List[AgentMessage]:
        """
        Generate a discussion between agents using Theory of Mind.
        
        Agents discuss the analysis, considering each other's perspectives.
        """
        if self.last_analysis is None:
            return [AgentMessage(
                agent_name="Guardian",
                content="We need something to discuss first. Give us some text!",
                confidence=0.5,
                emotion_state="neutral"
            )]
        
        import random
        random.seed(seed)
        
        messages = []
        
        e_msg = self.get_agent_response('emotion', seed)
        messages.append(e_msg)
        
        s_msg = self.get_agent_response('security', seed + 1)
        messages.append(s_msg)
        
        q_msg = self.get_agent_response('quality', seed + 2)
        messages.append(q_msg)
        
        emotion_state = self.last_analysis['emotion']['state']
        risk_level = self.last_analysis['security']['level'].replace('_', ' ')
        quality_level = self.last_analysis['quality']['level']
        
        e_to_s = self._select_template(
            self.PEER_DISCUSSION_TEMPLATES['emotion_to_security'], seed + 3
        ).format(emotion_state=emotion_state)
        messages.append(AgentMessage(
            agent_name="Emotion",
            content=e_to_s,
            confidence=self.last_analysis['emotion']['confidence'],
            emotion_state=emotion_state,
            to_agent="Security"
        ))
        
        tom_accuracy = self.swarm.get_psychological_profile()['team_cohesion']['theory_of_mind']
        if tom_accuracy > 0.7:
            response = self._select_template(self.AGREEMENT_TEMPLATES, seed + 4).format(peer="Emotion")
            response += f" The {emotion_state} tone does {'heighten' if emotion_state == 'negative' else 'not significantly affect'} my risk assessment."
        else:
            response = self._select_template(self.DISAGREEMENT_TEMPLATES, seed + 4).format(peer="Emotion")
            response += f" My security analysis is more focused on the technical indicators."
        
        messages.append(AgentMessage(
            agent_name="Security",
            content=response,
            confidence=self.last_analysis['security']['confidence'],
            emotion_state=risk_level,
            to_agent="Emotion"
        ))
        
        q_to_both = f"Looking at both perspectives - emotional tone ({emotion_state}) and security ({risk_level}) - "
        if self.last_analysis['consensus']:
            q_to_both += "I think we're aligned here. The quality assessment of {level} fits with your readings.".format(
                level=quality_level
            )
        else:
            q_to_both += "we have some divergence. My quality reading of {level} may need to account for both factors.".format(
                level=quality_level
            )
        
        messages.append(AgentMessage(
            agent_name="Quality",
            content=q_to_both,
            confidence=self.last_analysis['quality']['confidence'],
            emotion_state=quality_level,
            to_agent="Team"
        ))
        
        self.conversation_history.extend(messages)
        
        return messages
    
    def chat(self, user_message: str) -> List[AgentMessage]:
        """
        Process user message and get team response.
        
        The agents will analyze the message and discuss it.
        """
        self.analyze(user_message)
        
        responses = []
        
        responses.append(AgentMessage(
            agent_name="Guardian",
            content=f"Let me have the team analyze this...",
            confidence=1.0,
            emotion_state="neutral"
        ))
        
        responses.extend(self.get_team_discussion(seed=hash(user_message) % 1000))
        
        consensus = self.last_analysis['consciousness']
        responses.append(AgentMessage(
            agent_name="Guardian",
            content=f"Team consciousness: {consensus['overall_consciousness']:.1%} | "
                    f"Theory of Mind: {consensus['team_cohesion']['theory_of_mind']:.1%} | "
                    f"Self-Awareness: {sum(consensus['self_awareness'].values())/3:.1%}",
            confidence=1.0,
            emotion_state="summary"
        ))
        
        return responses
    
    def format_conversation(self, messages: List[AgentMessage]) -> str:
        """Format messages for display."""
        output = []
        for msg in messages:
            if msg.to_agent:
                header = f"[{msg.agent_name} -> {msg.to_agent}]"
            else:
                header = f"[{msg.agent_name}]"
            
            conf_indicator = ""
            if msg.confidence < 0.6:
                conf_indicator = " (uncertain)"
            elif msg.confidence > 0.9:
                conf_indicator = " (confident)"
            
            output.append(f"{header}{conf_indicator}")
            output.append(f"  {msg.content}")
            output.append("")
        
        return "\n".join(output)


def run_conversation_demo():
    """Demo the conversational interface."""
    print("=" * 70)
    print("GUARDIAN SWARM - CONVERSATIONAL INTERFACE")
    print("Agents that think AND talk!")
    print("=" * 70)
    
    conv = ConversationalSwarm(hidden_dim=64)
    
    test_inputs = [
        "Thanks for helping me debug this code! You're amazing!",
        "This SQL query looks suspicious: SELECT * FROM users WHERE id={input}",
        "I've been waiting 3 hours for a response. This is unacceptable.",
    ]
    
    for text in test_inputs:
        print(f"\n{'='*70}")
        print(f"USER: {text}")
        print("=" * 70)
        
        messages = conv.chat(text)
        print(conv.format_conversation(messages))


if __name__ == "__main__":
    run_conversation_demo()
