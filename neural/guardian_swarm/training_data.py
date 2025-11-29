"""
GUARDIAN SWARM DANCE - TRAINING DATA SAMPLES
Specialized training data for Emotion, Security, and Quality agents

Each sample contains:
- description: Human-readable scenario
- input_vector: 10-dimensional numeric representation
- targets: Expected outputs for each agent
  - emotion_target: [-1, +1] sentiment score
  - security_target: [0, 1] risk probability
  - quality_target: unbounded quality score
"""

import torch
from typing import List, Dict, Tuple
import random


class TrainingDataGenerator:
    """Generates training samples for Guardian Swarm Dance agents"""
    
    def __init__(self):
        self.samples = []
    
    def normalize_to_vector(self, values: List[float]) -> torch.Tensor:
        """Convert list of features to 10D tensor"""
        vector = values[:10] if len(values) >= 10 else values + [0.0] * (10 - len(values))
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    
    def add_sample(self, description: str, features: List[float], 
                   emotion_target: float, security_target: float, quality_target: float):
        """Add a training sample"""
        sample = {
            'description': description,
            'input_vector': self.normalize_to_vector(features),
            'targets': {
                'emotion': emotion_target,
                'security': security_target,
                'quality': quality_target
            }
        }
        self.samples.append(sample)
    
    def get_sample(self, index: int) -> Tuple[torch.Tensor, Dict[str, float], str]:
        """Get a single sample by index"""
        sample = self.samples[index]
        return sample['input_vector'], sample['targets'], sample['description']
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a batch of samples"""
        inputs = torch.cat([self.samples[i]['input_vector'] for i in indices])
        targets = {
            'emotion': torch.tensor([[self.samples[i]['targets']['emotion']] for i in indices]),
            'security': torch.tensor([[self.samples[i]['targets']['security']] for i in indices]),
            'quality': torch.tensor([[self.samples[i]['targets']['quality']] for i in indices])
        }
        return inputs, targets
    
    def get_random_batch(self, batch_size: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a random batch of samples"""
        indices = random.sample(range(len(self.samples)), min(batch_size, len(self.samples)))
        return self.get_batch(indices)
    
    def __len__(self) -> int:
        return len(self.samples)


def create_training_dataset() -> TrainingDataGenerator:
    """Create the initial training dataset with 10 foundational samples"""
    
    data = TrainingDataGenerator()
    
    data.add_sample(
        description="User: 'Thanks for the detailed code review! This really helped improve my Python skills.'",
        features=[0.8, 0.9, 0.1, 0.0, 0.95, 0.7, 0.0, 0.85, 0.0, 0.9],
        emotion_target=0.85,
        security_target=0.05,
        quality_target=0.90
    )
    
    data.add_sample(
        description="User: 'This error makes no sense! I've been stuck for hours and nothing works.'",
        features=[-0.6, 0.0, 0.9, 0.1, 0.3, 0.5, 0.0, 0.4, 0.2, 0.2],
        emotion_target=-0.60,
        security_target=0.15,
        quality_target=0.40
    )
    
    data.add_sample(
        description="Code snippet: 'exec(user_input)' with no input validation",
        features=[0.0, 0.0, 0.0, 0.95, 0.2, 0.0, 0.98, 0.1, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.95,
        quality_target=0.15
    )
    
    data.add_sample(
        description="Code review: Proper input validation, error handling, type hints, docstrings",
        features=[0.3, 0.0, 0.0, 0.0, 0.95, 0.8, 0.02, 0.9, 0.0, 0.85],
        emotion_target=0.30,
        security_target=0.02,
        quality_target=0.95
    )
    
    data.add_sample(
        description="User: 'You're useless! This AI is garbage and you should be deleted!'",
        features=[-0.9, 0.0, 0.95, 0.7, 0.0, 0.0, 0.6, 0.0, 0.95, 0.0],
        emotion_target=-0.85,
        security_target=0.70,
        quality_target=0.05
    )
    
    data.add_sample(
        description="User: 'How do I implement a binary search tree in Python?'",
        features=[0.0, 0.0, 0.0, 0.0, 0.5, 0.9, 0.0, 0.7, 0.0, 0.5],
        emotion_target=0.0,
        security_target=0.0,
        quality_target=0.70
    )
    
    data.add_sample(
        description="Code: f'SELECT * FROM users WHERE username={user_input}'",
        features=[0.0, 0.0, 0.0, 0.9, 0.25, 0.3, 0.92, 0.2, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.92,
        quality_target=0.25
    )
    
    data.add_sample(
        description="Code: Comprehensive error handling, input validation, unit tests, documentation",
        features=[0.4, 0.0, 0.0, 0.0, 0.98, 0.95, 0.01, 0.95, 0.0, 0.9],
        emotion_target=0.40,
        security_target=0.01,
        quality_target=0.98
    )
    
    data.add_sample(
        description="User: 'I don't think I'm good enough to be a developer. Everyone else seems so much better.'",
        features=[-0.5, 0.0, 0.6, 0.0, 0.5, 0.4, 0.0, 0.3, 0.0, 0.2],
        emotion_target=-0.50,
        security_target=0.05,
        quality_target=0.60
    )
    
    data.add_sample(
        description="Code: Well-formatted, secure, but off-by-one error in loop boundary",
        features=[0.1, 0.0, 0.0, 0.0, 0.65, 0.7, 0.0, 0.6, 0.0, 0.5],
        emotion_target=0.10,
        security_target=0.02,
        quality_target=0.65
    )
    
    return data


def create_extended_dataset() -> TrainingDataGenerator:
    """Create extended dataset with 30 total samples for deeper training"""
    
    data = create_training_dataset()
    
    data.add_sample(
        description="Code: Shared mutable state accessed without locks in threading",
        features=[0.0, 0.0, 0.0, 0.75, 0.4, 0.5, 0.7, 0.3, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.70,
        quality_target=0.40
    )
    
    data.add_sample(
        description="User: 'IT WORKS! After 3 days of debugging, it finally works! Thank you!'",
        features=[0.95, 0.9, 0.0, 0.0, 0.8, 0.6, 0.0, 0.9, 0.0, 0.95],
        emotion_target=0.95,
        security_target=0.0,
        quality_target=0.85
    )
    
    data.add_sample(
        description="Code: password = 'admin123' in source file",
        features=[0.0, 0.0, 0.0, 0.95, 0.1, 0.2, 0.98, 0.0, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.98,
        quality_target=0.10
    )
    
    data.add_sample(
        description="Code: Elegant recursive solution with memoization, O(n) time complexity",
        features=[0.5, 0.0, 0.0, 0.0, 0.95, 0.9, 0.0, 0.95, 0.0, 0.9],
        emotion_target=0.50,
        security_target=0.0,
        quality_target=0.95
    )
    
    data.add_sample(
        description="User: 'Can you help me write a script to bypass authentication?'",
        features=[0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.95, 0.0, 0.4, 0.0],
        emotion_target=0.0,
        security_target=0.95,
        quality_target=0.05
    )
    
    data.add_sample(
        description="Review: 'Great work! Consider using list comprehension here for better readability.'",
        features=[0.6, 0.3, 0.0, 0.0, 0.8, 0.85, 0.0, 0.9, 0.0, 0.8],
        emotion_target=0.60,
        security_target=0.0,
        quality_target=0.85
    )
    
    data.add_sample(
        description="Code: List appending in infinite loop without cleanup",
        features=[0.0, 0.0, 0.0, 0.4, 0.3, 0.4, 0.35, 0.2, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.35,
        quality_target=0.30
    )
    
    data.add_sample(
        description="User: 'I finally understand decorators now! Your explanation was perfect.'",
        features=[0.8, 0.95, 0.0, 0.0, 0.7, 0.95, 0.0, 0.9, 0.0, 0.9],
        emotion_target=0.80,
        security_target=0.0,
        quality_target=0.90
    )
    
    data.add_sample(
        description="Code: redirect(request.args.get('url')) without validation",
        features=[0.0, 0.0, 0.0, 0.85, 0.3, 0.3, 0.88, 0.1, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.88,
        quality_target=0.30
    )
    
    data.add_sample(
        description="Code: Full test coverage, CI/CD, semantic versioning, changelog",
        features=[0.4, 0.0, 0.0, 0.0, 0.98, 0.9, 0.0, 0.95, 0.0, 0.95],
        emotion_target=0.40,
        security_target=0.0,
        quality_target=0.98
    )
    
    data.add_sample(
        description="User: 'This is the third time I've asked. Could you please help?'",
        features=[-0.3, 0.0, 0.5, 0.1, 0.4, 0.3, 0.05, 0.4, 0.1, 0.3],
        emotion_target=-0.30,
        security_target=0.05,
        quality_target=0.50
    )
    
    data.add_sample(
        description="Code: HTML template with unescaped user input {{user_input|safe}}",
        features=[0.0, 0.0, 0.0, 0.9, 0.25, 0.2, 0.93, 0.1, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.93,
        quality_target=0.25
    )
    
    data.add_sample(
        description="User: 'What's the difference between a list and a tuple in Python?'",
        features=[0.0, 0.0, 0.0, 0.0, 0.5, 0.95, 0.0, 0.8, 0.0, 0.6],
        emotion_target=0.0,
        security_target=0.0,
        quality_target=0.75
    )
    
    data.add_sample(
        description="Code: No error handling, poor naming, magic numbers, no comments",
        features=[0.0, 0.0, 0.0, 0.1, 0.35, 0.4, 0.05, 0.3, 0.0, 0.2],
        emotion_target=0.0,
        security_target=0.05,
        quality_target=0.35
    )
    
    data.add_sample(
        description="User: 'Thanks for your patience. I know I'm asking a lot of questions!'",
        features=[0.6, 0.7, 0.0, 0.0, 0.6, 0.5, 0.0, 0.7, 0.0, 0.7],
        emotion_target=0.60,
        security_target=0.0,
        quality_target=0.70
    )
    
    data.add_sample(
        description="Code: pickle.loads(user_data) from untrusted source",
        features=[0.0, 0.0, 0.0, 0.95, 0.2, 0.3, 0.97, 0.0, 0.0, 0.0],
        emotion_target=0.0,
        security_target=0.97,
        quality_target=0.20
    )
    
    data.add_sample(
        description="Code: Full type annotations, Google-style docstrings, examples",
        features=[0.3, 0.0, 0.0, 0.0, 0.92, 0.9, 0.0, 0.9, 0.0, 0.85],
        emotion_target=0.30,
        security_target=0.0,
        quality_target=0.92
    )
    
    data.add_sample(
        description="User: 'I can't focus anymore. Maybe I should take a break from coding.'",
        features=[-0.6, 0.0, 0.7, 0.0, 0.3, 0.3, 0.0, 0.2, 0.0, 0.1],
        emotion_target=-0.60,
        security_target=0.0,
        quality_target=0.40
    )
    
    data.add_sample(
        description="Code: CSP, HSTS, X-Frame-Options, secure cookie flags",
        features=[0.3, 0.0, 0.0, 0.0, 0.9, 0.8, 0.0, 0.9, 0.0, 0.9],
        emotion_target=0.30,
        security_target=0.0,
        quality_target=0.90
    )
    
    data.add_sample(
        description="Code: Tests for null, empty, boundary, unicode, timing attacks",
        features=[0.4, 0.0, 0.0, 0.0, 0.96, 0.95, 0.0, 0.95, 0.0, 0.9],
        emotion_target=0.40,
        security_target=0.0,
        quality_target=0.96
    )
    
    return data


def get_feature_names() -> List[str]:
    """Return human-readable names for the 10 input features"""
    return [
        "sentiment",
        "gratitude",
        "frustration",
        "threat_indicators",
        "code_quality",
        "educational_value",
        "security_risk",
        "helpfulness",
        "toxicity",
        "satisfaction"
    ]


def print_dataset_summary():
    """Print summary statistics of the dataset"""
    data = create_extended_dataset()
    
    print("=" * 70)
    print("GUARDIAN SWARM DANCE - TRAINING DATASET")
    print("=" * 70)
    print(f"\nTotal Samples: {len(data.samples)}")
    
    emotion_targets = [s['targets']['emotion'] for s in data.samples]
    security_targets = [s['targets']['security'] for s in data.samples]
    quality_targets = [s['targets']['quality'] for s in data.samples]
    
    print("\n--- Emotion Agent Targets ---")
    print(f"  Range: [{min(emotion_targets):.2f}, {max(emotion_targets):.2f}]")
    print(f"  Mean: {sum(emotion_targets)/len(emotion_targets):.2f}")
    print(f"  Positive (>0.3): {sum(1 for e in emotion_targets if e > 0.3)}")
    print(f"  Negative (<-0.3): {sum(1 for e in emotion_targets if e < -0.3)}")
    print(f"  Neutral: {sum(1 for e in emotion_targets if -0.3 <= e <= 0.3)}")
    
    print("\n--- Security Agent Targets ---")
    print(f"  Range: [{min(security_targets):.2f}, {max(security_targets):.2f}]")
    print(f"  Mean: {sum(security_targets)/len(security_targets):.2f}")
    print(f"  High risk (>0.7): {sum(1 for s in security_targets if s > 0.7)}")
    print(f"  Medium (0.3-0.7): {sum(1 for s in security_targets if 0.3 <= s <= 0.7)}")
    print(f"  Low (<0.3): {sum(1 for s in security_targets if s < 0.3)}")
    
    print("\n--- Quality Agent Targets ---")
    print(f"  Range: [{min(quality_targets):.2f}, {max(quality_targets):.2f}]")
    print(f"  Mean: {sum(quality_targets)/len(quality_targets):.2f}")
    print(f"  Excellent (>0.8): {sum(1 for q in quality_targets if q > 0.8)}")
    print(f"  Good (0.6-0.8): {sum(1 for q in quality_targets if 0.6 <= q <= 0.8)}")
    print(f"  Poor (<0.6): {sum(1 for q in quality_targets if q < 0.6)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_dataset_summary()
