"""
Specialized Neural Network Agents for Guardian Swarm Dance

Each agent has a distinct "personality" encoded in its architecture:
- EmotionAgent: Empathetic, feeling-focused (tanh output: -1 to +1)
- SecurityAgent: Cautious, risk-aware (sigmoid output: 0 to 1)
- QualityAgent: Precise, standards-focused (linear output: unbounded)

Total parameters per agent: 430
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionAgent(nn.Module):
    """
    Detects emotional patterns in data - empathetic responses.
    
    Output interpretation:
        -1.0 = strongly negative sentiment
         0.0 = neutral
        +1.0 = strongly positive sentiment
    
    Architecture:
        Input (10) -> Hidden (20, ReLU) -> Output (10, tanh)
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.personality = "emotion"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return f"EmotionAgent(params={self.count_parameters()}, output=tanh[-1,+1])"


class SecurityAgent(nn.Module):
    """
    Detects threats/risks - cautious responses.
    
    Output interpretation:
        0.0 = completely safe, no risk
        0.5 = moderate risk, needs attention
        1.0 = high risk, dangerous
    
    Architecture:
        Input (10) -> Hidden (20, ReLU) -> Output (10, sigmoid)
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.personality = "security"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return f"SecurityAgent(params={self.count_parameters()}, output=sigmoid[0,1])"


class QualityAgent(nn.Module):
    """
    Evaluates quality/correctness - precise responses.
    
    Output interpretation:
        < 0 = below standard quality
        = 0 = meets standard
        > 0 = exceeds standard (higher = better)
    
    Architecture:
        Input (10) -> Hidden (20, ReLU) -> Output (10, linear)
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.personality = "quality"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return f"QualityAgent(params={self.count_parameters()}, output=linear)"


def create_agent_swarm(
    input_size: int = 10,
    hidden_size: int = 20,
    output_size: int = 10,
    device: torch.device = None
) -> tuple:
    """
    Create all three agents with matching dimensions.
    
    Returns:
        Tuple of (EmotionAgent, SecurityAgent, QualityAgent)
    """
    if device is None:
        device = torch.device('cpu')
    
    emotion = EmotionAgent(input_size, hidden_size, output_size).to(device)
    security = SecurityAgent(input_size, hidden_size, output_size).to(device)
    quality = QualityAgent(input_size, hidden_size, output_size).to(device)
    
    return emotion, security, quality


if __name__ == "__main__":
    print("Guardian Swarm Dance - Agent Test")
    print("=" * 50)
    
    emotion, security, quality = create_agent_swarm()
    
    print(f"\n{emotion}")
    print(f"{security}")
    print(f"{quality}")
    
    total_params = sum(a.count_parameters() for a in [emotion, security, quality])
    print(f"\nTotal swarm parameters: {total_params:,}")
    print(f"Memory footprint: ~{total_params * 4 / 1024:.1f} KB (float32)")
    
    test_input = torch.randn(1, 10)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Emotion output: {emotion(test_input).mean().item():.4f} (range: -1 to +1)")
    print(f"Security output: {security(test_input).mean().item():.4f} (range: 0 to 1)")
    print(f"Quality output: {quality(test_input).mean().item():.4f} (unbounded)")
