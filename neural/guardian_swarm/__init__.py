"""
Guardian Swarm Dance - Multi-Agent Collaborative AI System
Pure PyTorch implementation - No external dependencies

Version 2.0: Now with Conscious Swarm mechanisms:
- Self-Aware Agents (confidence, introspection)
- Theory of Mind (peer modeling)
- Shared Memory Bank
- Affinity Learning
"""

from .agents import EmotionAgent, SecurityAgent, QualityAgent
from .guardian import GuardianAngel
from .swarm import SwarmDance
from .training_data import (
    TrainingDataGenerator,
    create_training_dataset,
    create_extended_dataset,
    get_feature_names,
    print_dataset_summary
)
from .conscious_agents import (
    SelfAwareAgent,
    EmotionAgentConscious,
    SecurityAgentConscious,
    QualityAgentConscious,
    PerformanceMemory
)
from .conscious_swarm import (
    ConsciousSwarm,
    ConsciousSwarmTrainer,
    PeerModel,
    SharedMemoryBank,
    AffinityMatrix
)
from .conversation import (
    ConversationalSwarm,
    AgentMessage,
    encode_text_to_features
)

__version__ = "2.1.0"
__all__ = [
    # Original swarm
    "EmotionAgent",
    "SecurityAgent", 
    "QualityAgent",
    "GuardianAngel",
    "SwarmDance",
    # Training data
    "TrainingDataGenerator",
    "create_training_dataset",
    "create_extended_dataset",
    "get_feature_names",
    "print_dataset_summary",
    # Conscious swarm (v2.0)
    "SelfAwareAgent",
    "EmotionAgentConscious",
    "SecurityAgentConscious",
    "QualityAgentConscious",
    "PerformanceMemory",
    "ConsciousSwarm",
    "ConsciousSwarmTrainer",
    "PeerModel",
    "SharedMemoryBank",
    "AffinityMatrix",
    # Conversational interface (v2.1)
    "ConversationalSwarm",
    "AgentMessage",
    "encode_text_to_features",
]
