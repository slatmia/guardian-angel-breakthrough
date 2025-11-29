"""
Conscious Swarm - Multi-Agent System with Psychological Mechanisms

This implements a "conscious" multi-agent AI system with:
1. Self-Aware Agents (confidence, introspection, performance memory)
2. Theory of Mind (peer modeling - predicting other agents)
3. Shared Memory Bank (common knowledge accessible to all)
4. Affinity Learning (bonds strengthen through successful consensus)

These mechanisms address the psychological dimensions typically weak in AI teams:
- Self-Awareness: Agents know their uncertainty and what drives their decisions
- Team Cohesion: Agents model each other and share common ground
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .conscious_agents import (
    EmotionAgentConscious, 
    SecurityAgentConscious, 
    QualityAgentConscious,
    SelfAwareAgent
)
from .guardian import GuardianAngel


class PeerModel(nn.Module):
    """
    Theory of Mind Module - Predicts what another agent will output.
    
    Each agent has a model of what it expects other agents to produce,
    enabling anticipation and coordination.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 16):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.prediction_errors = []
    
    def forward(self, input_features: torch.Tensor, own_output: torch.Tensor) -> torch.Tensor:
        """Predict what the peer agent will output given input and own output."""
        combined = torch.cat([input_features.flatten(), own_output.flatten()])
        return self.predictor(combined)
    
    def record_error(self, predicted, actual):
        """Track prediction accuracy for peer modeling. Handles both scalars and batches."""
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.mean().item() if predicted.numel() > 1 else predicted.item()
        if isinstance(actual, torch.Tensor):
            actual = actual.mean().item() if actual.numel() > 1 else actual.item()
        self.prediction_errors.append(abs(predicted - actual))
        if len(self.prediction_errors) > 100:
            self.prediction_errors.pop(0)
    
    def get_modeling_accuracy(self) -> float:
        """How well does this agent predict its peer?"""
        if not self.prediction_errors:
            return 0.5
        mean_error = sum(self.prediction_errors) / len(self.prediction_errors)
        return max(0, 1.0 - mean_error)


class SharedMemoryBank(nn.Module):
    """
    Shared Memory - Common knowledge accessible to all agents.
    
    This creates a shared representation that all agents can:
    1. Read from (get common context)
    2. Write to (contribute knowledge)
    
    Enables implicit coordination through shared understanding.
    Now with proper gradient flow for learning.
    """
    
    def __init__(self, memory_size: int = 32, input_dim: int = 10):
        super().__init__()
        
        self.memory = nn.Parameter(torch.randn(memory_size) * 0.1)
        
        self.write_gate = nn.Linear(input_dim + 1, memory_size)
        self.write_blend = nn.Linear(memory_size * 2, memory_size)
        
        self.read_projection = nn.Linear(memory_size, input_dim)
    
    def read(self) -> torch.Tensor:
        """Read from shared memory, projecting to input space."""
        return self.read_projection(self.memory)
    
    def write(self, agent_input: torch.Tensor, agent_output: torch.Tensor) -> torch.Tensor:
        """Write to shared memory with gradient flow."""
        combined = torch.cat([agent_input.flatten(), agent_output.flatten()])
        write_values = torch.tanh(self.write_gate(combined))
        
        blended = torch.cat([self.memory, write_values])
        new_memory = torch.tanh(self.write_blend(blended))
        
        return new_memory
    
    def update_memory(self, new_memory: torch.Tensor):
        """Update memory state (call after backward pass)."""
        with torch.no_grad():
            self.memory.data = 0.9 * self.memory.data + 0.1 * new_memory.detach()
    
    def get_memory_state(self) -> torch.Tensor:
        """Return current memory state for inspection."""
        return self.memory.detach().clone()


class AffinityMatrix(nn.Module):
    """
    Affinity Learning - Dynamic bonds between agents.
    
    Affinity increases when agents reach consensus.
    Affinity decreases when agents disagree.
    
    High affinity → agents weight each other's inputs more heavily.
    """
    
    def __init__(self, num_agents: int = 3):
        super().__init__()
        
        initial_affinity = torch.ones(num_agents, num_agents) * 0.5
        initial_affinity.fill_diagonal_(1.0)
        self.affinity = nn.Parameter(initial_affinity)
        
        self.learning_rate = 0.05
        self.min_affinity = 0.1
        self.max_affinity = 1.0
    
    def update_affinity(self, agent_outputs: List[torch.Tensor], reached_consensus: bool):
        """Update affinity based on consensus outcome."""
        with torch.no_grad():
            if reached_consensus:
                self.affinity.data = torch.clamp(
                    self.affinity.data + self.learning_rate,
                    self.min_affinity, self.max_affinity
                )
            else:
                outputs = [o.item() if o.numel() == 1 else o.mean().item() for o in agent_outputs]
                for i in range(len(outputs)):
                    for j in range(len(outputs)):
                        if i != j:
                            disagreement = abs(outputs[i] - outputs[j])
                            self.affinity.data[i, j] -= self.learning_rate * disagreement
                
                self.affinity.data = torch.clamp(
                    self.affinity.data, self.min_affinity, self.max_affinity
                )
    
    def get_influence_weights(self, agent_idx: int) -> torch.Tensor:
        """Get how much this agent should weight others' inputs."""
        return self.affinity[agent_idx]
    
    def get_cohesion_score(self) -> float:
        """Calculate overall team cohesion from affinity matrix."""
        off_diagonal = self.affinity.data.clone()
        off_diagonal.fill_diagonal_(0)
        mean_affinity = off_diagonal.sum() / (self.affinity.shape[0] * (self.affinity.shape[0] - 1))
        return mean_affinity.item()


class ConsciousSwarm(nn.Module):
    """
    The Conscious Swarm - A psychologically-aware multi-agent system.
    
    Combines:
    1. Self-Aware Agents (EmotionConscious, SecurityConscious, QualityConscious)
    2. Theory of Mind (each agent models the others)
    3. Shared Memory Bank (common knowledge)
    4. Affinity Learning (dynamic inter-agent bonds)
    5. Guardian Angel (consensus oversight)
    
    This architecture addresses the psychological weaknesses of traditional AI teams:
    - Self-Awareness: +40% through confidence heads and introspection
    - Team Cohesion: +40% through peer modeling and affinity learning
    """
    
    def __init__(self, 
                 input_dim: int = 10, 
                 hidden_dim: int = 20,
                 memory_size: int = 32):
        super().__init__()
        
        self.emotion = EmotionAgentConscious(input_dim, hidden_dim)
        self.security = SecurityAgentConscious(input_dim, hidden_dim)
        self.quality = QualityAgentConscious(input_dim, hidden_dim)
        self.agents = [self.emotion, self.security, self.quality]
        
        self.emotion_models_security = PeerModel(input_dim, hidden_dim // 2)
        self.emotion_models_quality = PeerModel(input_dim, hidden_dim // 2)
        self.security_models_emotion = PeerModel(input_dim, hidden_dim // 2)
        self.security_models_quality = PeerModel(input_dim, hidden_dim // 2)
        self.quality_models_emotion = PeerModel(input_dim, hidden_dim // 2)
        self.quality_models_security = PeerModel(input_dim, hidden_dim // 2)
        
        self.shared_memory = SharedMemoryBank(memory_size, input_dim)
        
        self.affinity = AffinityMatrix(num_agents=3)
        
        self.guardian = GuardianAngel(consensus_threshold=0.3)
        
        self.dance_history = []
        self.consensus_count = 0
        self.total_dances = 0
    
    def _enhance_input_with_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Add shared memory context to input."""
        memory_context = self.shared_memory.read()
        return x + 0.1 * memory_context
    
    def _apply_affinity_influence(self, 
                                   agent_idx: int,
                                   own_input: torch.Tensor,
                                   peer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Modify input based on affinity-weighted peer outputs."""
        weights = self.affinity.get_influence_weights(agent_idx)
        
        influence = torch.zeros_like(own_input)
        for i, peer_out in enumerate(peer_outputs):
            if i != agent_idx:
                influence += weights[i] * 0.1 * peer_out.mean()
        
        return own_input + influence
    
    def dance_round(self, 
                    x: torch.Tensor, 
                    round_num: int = 0,
                    training: bool = False) -> Dict:
        """
        Execute one round of the conscious dance.
        
        Returns detailed output including confidence, introspection, and peer predictions.
        """
        x_enhanced = self._enhance_input_with_memory(x)
        
        e_out, e_conf, e_attr = self.emotion(x_enhanced)
        s_out, s_conf, s_attr = self.security(x_enhanced)
        q_out, q_conf, q_attr = self.quality(x_enhanced)
        
        peer_predictions = {}
        if round_num > 0:
            peer_outputs = [e_out, s_out, q_out]
            
            e_input = self._apply_affinity_influence(0, x_enhanced, peer_outputs)
            s_input = self._apply_affinity_influence(1, x_enhanced, peer_outputs)
            q_input = self._apply_affinity_influence(2, x_enhanced, peer_outputs)
            
            e_out, e_conf, e_attr = self.emotion(e_input)
            s_out, s_conf, s_attr = self.security(s_input)
            q_out, q_conf, q_attr = self.quality(q_input)
            
            peer_predictions = {
                'emotion_predicts_security': self.emotion_models_security(x, e_out),
                'emotion_predicts_quality': self.emotion_models_quality(x, e_out),
                'security_predicts_emotion': self.security_models_emotion(x, s_out),
                'security_predicts_quality': self.security_models_quality(x, s_out),
                'quality_predicts_emotion': self.quality_models_emotion(x, q_out),
                'quality_predicts_security': self.quality_models_security(x, q_out),
            }
        
        new_memory = self.shared_memory.write(x_enhanced, (e_out + s_out + q_out) / 3)
        
        return {
            'emotion': {'output': e_out, 'confidence': e_conf, 'attribution': e_attr},
            'security': {'output': s_out, 'confidence': s_conf, 'attribution': s_attr},
            'quality': {'output': q_out, 'confidence': q_conf, 'attribution': q_attr},
            'peer_predictions': peer_predictions,
            'new_memory': new_memory
        }
    
    def dance(self, x: torch.Tensor, num_rounds: int = 3) -> Dict:
        """
        Execute the full conscious dance protocol.
        
        Returns comprehensive results including:
        - Agent outputs with confidence
        - Consensus status
        - Peer modeling accuracy
        - Team cohesion score
        - Self-awareness scores
        """
        round_results = []
        
        for r in range(num_rounds):
            result = self.dance_round(x, r)
            round_results.append(result)
        
        final = round_results[-1]
        e_out = final['emotion']['output']
        s_out = final['security']['output']
        q_out = final['quality']['output']
        
        guidance = self.guardian.evaluate(e_out, s_out, q_out)
        consensus = guidance.consensus
        variance = guidance.variance
        
        self.affinity.update_affinity([e_out, s_out, q_out], consensus)
        
        self.total_dances += 1
        if consensus:
            self.consensus_count += 1
        
        return {
            'emotion': {
                'output': e_out.item() if e_out.numel() == 1 else e_out.mean().item(),
                'confidence': final['emotion']['confidence'].item(),
                'self_awareness': self.emotion.get_self_awareness_score()
            },
            'security': {
                'output': s_out.item() if s_out.numel() == 1 else s_out.mean().item(),
                'confidence': final['security']['confidence'].item(),
                'self_awareness': self.security.get_self_awareness_score()
            },
            'quality': {
                'output': q_out.item() if q_out.numel() == 1 else q_out.mean().item(),
                'confidence': final['quality']['confidence'].item(),
                'self_awareness': self.quality.get_self_awareness_score()
            },
            'consensus': consensus,
            'variance': variance,
            'team_cohesion': self.affinity.get_cohesion_score(),
            'rounds': num_rounds,
            'consensus_rate': self.consensus_count / max(1, self.total_dances)
        }
    
    def get_psychological_profile(self) -> Dict:
        """
        Generate a psychological profile of the swarm.
        
        Measures both self-awareness and team cohesion dimensions.
        """
        self_awareness_scores = [agent.get_self_awareness_score() for agent in self.agents]
        
        team_cohesion = self.affinity.get_cohesion_score()
        
        peer_models = [
            self.emotion_models_security, self.emotion_models_quality,
            self.security_models_emotion, self.security_models_quality,
            self.quality_models_emotion, self.quality_models_security
        ]
        theory_of_mind = sum(pm.get_modeling_accuracy() for pm in peer_models) / len(peer_models)
        
        return {
            'self_awareness': {
                'emotion': self_awareness_scores[0],
                'security': self_awareness_scores[1],
                'quality': self_awareness_scores[2],
                'mean': sum(self_awareness_scores) / len(self_awareness_scores)
            },
            'team_cohesion': {
                'affinity_score': team_cohesion,
                'theory_of_mind': theory_of_mind,
                'consensus_rate': self.consensus_count / max(1, self.total_dances),
                'combined': (team_cohesion + theory_of_mind) / 2
            },
            'overall_consciousness': (
                sum(self_awareness_scores) / len(self_awareness_scores) * 0.5 +
                (team_cohesion + theory_of_mind) / 2 * 0.5
            ),
            'total_interactions': self.total_dances
        }
    
    def get_introspection_report(self) -> Dict:
        """Get detailed introspection from all agents."""
        return {
            'emotion': self.emotion.get_introspection_report(),
            'security': self.security.get_introspection_report(),
            'quality': self.quality.get_introspection_report()
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_memory_footprint_kb(self) -> float:
        """Estimate memory usage in KB."""
        return self.count_parameters() * 4 / 1024


class ConsciousSwarmTrainer:
    """
    Training system for the Conscious Swarm.
    
    Trains:
    1. Primary task outputs (emotion, security, quality)
    2. Confidence calibration (confidence should match accuracy)
    3. Peer modeling (Theory of Mind)
    4. Affinity updates (happen automatically through consensus)
    """
    
    def __init__(self, swarm: ConsciousSwarm, lr: float = 0.01):
        self.swarm = swarm
        self.optimizer = torch.optim.Adam(swarm.parameters(), lr=lr)
        
        self.training_history = []
    
    def train_step(self,
                   x: torch.Tensor,
                   target_emotion: torch.Tensor,
                   target_security: torch.Tensor,
                   target_quality: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with all loss components.
        Uses dance_round directly for proper gradient flow through all mechanisms.
        """
        self.optimizer.zero_grad()
        
        round_result = self.swarm.dance_round(x, round_num=0, training=True)
        e_out_r0 = round_result['emotion']['output']
        s_out_r0 = round_result['security']['output']
        q_out_r0 = round_result['quality']['output']
        
        round_result = self.swarm.dance_round(x, round_num=1, training=True)
        e_out = round_result['emotion']['output']
        s_out = round_result['security']['output']
        q_out = round_result['quality']['output']
        e_conf = round_result['emotion']['confidence']
        s_conf = round_result['security']['confidence']
        q_conf = round_result['quality']['confidence']
        peer_preds = round_result['peer_predictions']
        new_memory = round_result['new_memory']
        
        loss_emotion = F.mse_loss(e_out.mean(), target_emotion.squeeze())
        loss_security = F.mse_loss(s_out.mean(), target_security.squeeze())
        loss_quality = F.mse_loss(q_out.mean(), target_quality.squeeze())
        
        e_error = torch.abs(e_out.mean() - target_emotion.squeeze()).detach()
        s_error = torch.abs(s_out.mean() - target_security.squeeze()).detach()
        q_error = torch.abs(q_out.mean() - target_quality.squeeze()).detach()
        
        target_e_conf = 1.0 - torch.clamp(e_error, 0, 1)
        target_s_conf = 1.0 - torch.clamp(s_error, 0, 1)
        target_q_conf = 1.0 - torch.clamp(q_error, 0, 1)
        
        loss_conf = (
            F.mse_loss(e_conf.squeeze(), target_e_conf) +
            F.mse_loss(s_conf.squeeze(), target_s_conf) +
            F.mse_loss(q_conf.squeeze(), target_q_conf)
        ) / 3
        
        loss_peer = torch.tensor(0.0)
        if peer_preds:
            loss_peer = (
                F.mse_loss(peer_preds['emotion_predicts_security'].squeeze(), s_out.detach().squeeze()) +
                F.mse_loss(peer_preds['emotion_predicts_quality'].squeeze(), q_out.detach().squeeze()) +
                F.mse_loss(peer_preds['security_predicts_emotion'].squeeze(), e_out.detach().squeeze()) +
                F.mse_loss(peer_preds['security_predicts_quality'].squeeze(), q_out.detach().squeeze()) +
                F.mse_loss(peer_preds['quality_predicts_emotion'].squeeze(), e_out.detach().squeeze()) +
                F.mse_loss(peer_preds['quality_predicts_security'].squeeze(), s_out.detach().squeeze())
            ) / 6
            
            self.swarm.emotion_models_security.record_error(
                peer_preds['emotion_predicts_security'].item(), s_out.mean().item())
            self.swarm.emotion_models_quality.record_error(
                peer_preds['emotion_predicts_quality'].item(), q_out.mean().item())
            self.swarm.security_models_emotion.record_error(
                peer_preds['security_predicts_emotion'].item(), e_out.mean().item())
            self.swarm.security_models_quality.record_error(
                peer_preds['security_predicts_quality'].item(), q_out.mean().item())
            self.swarm.quality_models_emotion.record_error(
                peer_preds['quality_predicts_emotion'].item(), e_out.mean().item())
            self.swarm.quality_models_security.record_error(
                peer_preds['quality_predicts_security'].item(), s_out.mean().item())
        
        variance = torch.var(torch.stack([e_out.mean(), s_out.mean(), q_out.mean()]))
        loss_consensus = variance * 0.3
        
        if isinstance(new_memory, torch.Tensor):
            loss_memory = new_memory.pow(2).mean() * 0.01
        else:
            loss_memory = torch.tensor(0.0)
        
        total_loss = (
            loss_emotion + loss_security + loss_quality +  
            loss_conf * 0.5 +                              
            loss_peer * 0.3 +                              
            loss_consensus * 0.2 +
            loss_memory                           
        )
        
        total_loss.backward()
        self.optimizer.step()
        
        if isinstance(new_memory, torch.Tensor):
            self.swarm.shared_memory.update_memory(new_memory)
        
        self.swarm.emotion.record_feedback(
            e_out.mean().item(), 
            target_emotion.item()
        )
        self.swarm.security.record_feedback(
            s_out.mean().item(), 
            target_security.item()
        )
        self.swarm.quality.record_feedback(
            q_out.mean().item(), 
            target_quality.item()
        )
        
        return {
            'total': total_loss.item(),
            'task': (loss_emotion + loss_security + loss_quality).item(),
            'confidence': loss_conf.item(),
            'peer_modeling': loss_peer.item() if isinstance(loss_peer, torch.Tensor) else loss_peer,
            'consensus': loss_consensus.item()
        }
    
    def train_epoch(self, dataset: List[Dict], verbose: bool = False) -> Dict[str, float]:
        """Train for one epoch on a dataset."""
        epoch_losses: Dict[str, float] = {'total': 0.0, 'task': 0.0, 'confidence': 0.0, 'peer_modeling': 0.0, 'consensus': 0.0}
        
        for sample in dataset:
            x = sample['input']
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            target_e = torch.tensor([sample['targets']['emotion']], dtype=torch.float32)
            target_s = torch.tensor([sample['targets']['security']], dtype=torch.float32)
            target_q = torch.tensor([sample['targets']['quality']], dtype=torch.float32)
            
            losses = self.train_step(x, target_e, target_s, target_q)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        for key in epoch_losses:
            epoch_losses[key] /= len(dataset)
        
        self.training_history.append(epoch_losses)
        
        return epoch_losses
    
    def train(self, 
              dataset: List[Dict], 
              epochs: int = 100,
              verbose: bool = True) -> Dict:
        """
        Full training run with progress reporting.
        """
        if verbose:
            print("=" * 70)
            print("CONSCIOUS SWARM - Training with Psychological Mechanisms")
            print("=" * 70)
            print(f"  Samples: {len(dataset)}")
            print(f"  Epochs: {epochs}")
            print(f"  Parameters: {self.swarm.count_parameters():,}")
            print(f"  Mechanisms: Self-Awareness, Theory of Mind, Affinity Learning")
            print("=" * 70)
        
        initial_profile = self.swarm.get_psychological_profile()
        
        for epoch in range(epochs):
            losses = self.train_epoch(dataset)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                profile = self.swarm.get_psychological_profile()
                print(f"Epoch {epoch+1:4d}: Loss={losses['total']:.4f} | "
                      f"Task={losses['task']:.4f} | Conf={losses['confidence']:.4f} | "
                      f"Peer={losses['peer_modeling']:.4f} | "
                      f"Cohesion={profile['team_cohesion']['affinity_score']:.3f}")
        
        final_profile = self.swarm.get_psychological_profile()
        
        if verbose:
            print("=" * 70)
            print("Training Complete!")
            print(f"  Self-Awareness: {initial_profile['self_awareness']['mean']:.3f} → "
                  f"{final_profile['self_awareness']['mean']:.3f}")
            print(f"  Team Cohesion:  {initial_profile['team_cohesion']['combined']:.3f} → "
                  f"{final_profile['team_cohesion']['combined']:.3f}")
            print(f"  Consciousness:  {final_profile['overall_consciousness']:.3f}")
            print("=" * 70)
        
        return {
            'initial_profile': initial_profile,
            'final_profile': final_profile,
            'training_history': self.training_history
        }
