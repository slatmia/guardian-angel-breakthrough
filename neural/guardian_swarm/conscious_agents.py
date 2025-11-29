"""
Conscious Agents - Self-Aware Neural Network Agents

Each agent has:
1. Primary output head (task-specific)
2. Confidence head (metacognition - "how sure am I?")
3. Introspection (feature attribution - "what drove my decision?")
4. Performance history (learning from experience)

This implements psychological self-awareness mechanisms in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque


class PerformanceMemory:
    """Tracks agent's historical performance for self-awareness."""
    
    def __init__(self, capacity: int = 100):
        self.history = deque(maxlen=capacity)
        self.running_accuracy = 0.5
        self.total_samples = 0
    
    def record(self, predicted, actual, input_type: str = "unknown"):
        """Record a prediction and its ground truth. Handles both scalars and batches."""
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.mean().item() if predicted.numel() > 1 else predicted.item()
        if isinstance(actual, torch.Tensor):
            actual = actual.mean().item() if actual.numel() > 1 else actual.item()
            
        error = abs(predicted - actual)
        correct = error < 0.2
        
        self.history.append({
            'predicted': predicted,
            'actual': actual,
            'error': error,
            'correct': correct,
            'input_type': input_type
        })
        
        self.total_samples += 1
        alpha = min(0.1, 1.0 / self.total_samples)
        self.running_accuracy = (1 - alpha) * self.running_accuracy + alpha * (1.0 if correct else 0.0)
    
    def get_accuracy(self) -> float:
        """Get running accuracy estimate."""
        return self.running_accuracy
    
    def get_recent_performance(self, n: int = 10) -> Dict:
        """Get statistics from recent predictions."""
        if len(self.history) == 0:
            return {'accuracy': 0.5, 'mean_error': 0.5, 'samples': 0}
        
        recent = list(self.history)[-n:]
        accuracy = sum(1 for h in recent if h['correct']) / len(recent)
        mean_error = sum(h['error'] for h in recent) / len(recent)
        
        return {
            'accuracy': accuracy,
            'mean_error': mean_error,
            'samples': len(recent)
        }


class SelfAwareAgent(nn.Module):
    """
    A neural network agent with metacognitive capabilities.
    
    Architecture:
        Input (10) → Hidden (hidden_size) → Primary Output (1)
                                         → Confidence Output (1)
                                         → Introspection (hidden_size weights)
    
    Self-Awareness Mechanisms:
        1. Confidence estimation: "How sure am I about this output?"
        2. Feature attribution: "Which input features drove my decision?"
        3. Performance tracking: "How accurate have I been historically?"
        4. Self-critique: Adjust output based on confidence
    """
    
    def __init__(self, 
                 input_dim: int = 10, 
                 hidden_dim: int = 20,
                 activation: str = 'tanh',
                 name: str = 'agent'):
        super().__init__()
        
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.introspection_weights = nn.Linear(hidden_dim, input_dim)
        
        self.performance = PerformanceMemory()
        
        self._last_hidden = None
        self._last_input = None
        self._last_attribution = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.fc1, self.fc2, self.confidence_head, self.introspection_weights]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.zeros_(module.bias)
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply agent-specific activation function."""
        if self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        else:
            return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with metacognitive outputs.
        
        Returns:
            output: Primary task output
            confidence: How confident the agent is (0-1)
            attribution: Which input features drove the decision
        """
        self._last_input = x.detach()
        
        hidden = torch.relu(self.fc1(x))
        self._last_hidden = hidden.detach()
        
        raw_output = self.fc2(hidden)
        output = self._apply_activation(raw_output)
        
        confidence = torch.sigmoid(self.confidence_head(hidden))
        
        attribution = torch.softmax(self.introspection_weights(hidden), dim=-1)
        self._last_attribution = attribution.detach()
        
        return output, confidence, attribution
    
    def forward_simple(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward for compatibility with original swarm."""
        output, _, _ = self.forward(x)
        return output
    
    def get_introspection_report(self) -> Dict:
        """Get a report on what drove the last decision."""
        if self._last_attribution is None:
            return {'error': 'No forward pass recorded'}
        
        feature_names = [
            'sentiment', 'gratitude', 'frustration', 'threat_indicators',
            'code_quality', 'educational_value', 'security_risk',
            'helpfulness', 'toxicity', 'satisfaction'
        ]
        
        attr = self._last_attribution.squeeze().tolist()
        if isinstance(attr, float):
            attr = [attr] * len(feature_names)
        
        sorted_features = sorted(
            zip(feature_names, attr),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'top_features': sorted_features[:3],
            'all_attributions': dict(zip(feature_names, attr)),
            'agent': self.name,
            'historical_accuracy': self.performance.get_accuracy()
        }
    
    def self_critique(self, output: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        Adjust output based on confidence.
        Low confidence → move toward neutral (0.5 for sigmoid, 0 for tanh)
        """
        if self.activation_type == 'tanh':
            neutral = 0.0
        elif self.activation_type == 'sigmoid':
            neutral = 0.5
        else:
            neutral = 0.0
        
        adjusted = confidence * output + (1 - confidence) * neutral
        return adjusted
    
    def record_feedback(self, predicted: float, actual: float, input_type: str = "unknown"):
        """Record prediction outcome for learning."""
        self.performance.record(predicted, actual, input_type)
    
    def get_self_awareness_score(self) -> float:
        """
        Calculate a self-awareness score based on:
        - Calibration: Does confidence match accuracy?
        - History: Has the agent learned from experience?
        """
        perf = self.performance.get_recent_performance()
        if perf['samples'] < 5:
            return 0.5
        
        calibration = 1.0 - abs(perf['accuracy'] - self.performance.running_accuracy)
        
        history_factor = min(1.0, perf['samples'] / 50.0)
        
        return 0.6 * calibration + 0.4 * history_factor


class EmotionAgentConscious(SelfAwareAgent):
    """Emotion-specialized self-aware agent with tanh activation."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20):
        super().__init__(input_dim, hidden_dim, activation='tanh', name='emotion')


class SecurityAgentConscious(SelfAwareAgent):
    """Security-specialized self-aware agent with sigmoid activation."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20):
        super().__init__(input_dim, hidden_dim, activation='sigmoid', name='security')


class QualityAgentConscious(SelfAwareAgent):
    """Quality-specialized self-aware agent with linear activation."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20):
        super().__init__(input_dim, hidden_dim, activation='linear', name='quality')
