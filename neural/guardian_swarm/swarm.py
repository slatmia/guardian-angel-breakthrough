"""
Swarm Dance - Orchestrates Multi-Agent Training and Inference

The SwarmDance class coordinates:
1. Forward passes through all agents
2. Inter-agent communication (agents see each other's outputs)
3. Consensus-based training (agents learn to agree)
4. Guardian Angel oversight
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .agents import EmotionAgent, SecurityAgent, QualityAgent
from .guardian import GuardianAngel, GuidanceResult, EpochSummary
from .training_data import TrainingDataGenerator, create_extended_dataset, create_training_dataset


@dataclass
class DanceResult:
    """Result of a single dance (multi-round discussion)"""
    emotion_output: torch.Tensor
    security_output: torch.Tensor
    quality_output: torch.Tensor
    rounds: List[GuidanceResult]
    final_guidance: GuidanceResult
    
    @property
    def outputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.emotion_output, self.security_output, self.quality_output
    
    @property
    def mean_outputs(self) -> Dict[str, float]:
        return {
            "emotion": self.emotion_output.mean().item(),
            "security": self.security_output.mean().item(),
            "quality": self.quality_output.mean().item()
        }


class SwarmDance:
    """
    Orchestrates the 3 agents training on each other.
    
    The "dance" metaphor represents:
    - Coordination: Agents move together, responding to each other
    - Rhythm: Fixed rounds of discussion (default: 3)
    - Harmony: Goal is consensus, not individual excellence
    - Practice: Training improves coordination over time
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 20,
        output_size: int = 10,
        learning_rate: float = 0.01,
        consensus_weight: float = 0.5,
        influence_self: float = 0.1,
        influence_peer: float = 0.05,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.consensus_weight = consensus_weight
        self.influence_self = influence_self
        self.influence_peer = influence_peer
        
        self.emotion = EmotionAgent(input_size, hidden_size, output_size).to(self.device)
        self.security = SecurityAgent(input_size, hidden_size, output_size).to(self.device)
        self.quality = QualityAgent(input_size, hidden_size, output_size).to(self.device)
        
        self.guardian = GuardianAngel()
        
        self.opt_emotion = optim.Adam(self.emotion.parameters(), lr=learning_rate)
        self.opt_security = optim.Adam(self.security.parameters(), lr=learning_rate)
        self.opt_quality = optim.Adam(self.quality.parameters(), lr=learning_rate)
        
        self.training_history: List[EpochSummary] = []
    
    def dance(
        self,
        user_input: torch.Tensor,
        rounds: int = 3,
        verbose: bool = False
    ) -> DanceResult:
        """
        Perform a multi-round dance (discussion) between agents.
        
        Args:
            user_input: Input tensor representing user query
            rounds: Number of discussion rounds
            verbose: Print round-by-round details
        
        Returns:
            DanceResult with final outputs and round history
        """
        emotion_out = self.emotion(user_input)
        security_out = self.security(user_input)
        quality_out = self.quality(user_input)
        
        round_results = []
        
        for r in range(rounds + 1):
            guidance = self.guardian.evaluate(
                emotion_out, security_out, quality_out,
                round_number=r
            )
            round_results.append(guidance)
            
            if verbose:
                means = guidance.agent_means
                print(f"  Round {r}: E={means['emotion']:+.4f}, "
                      f"S={means['security']:.4f}, Q={means['quality']:+.4f} "
                      f"[var={guidance.variance:.4f}]")
            
            if r < rounds:
                emotion_input = user_input + emotion_out * self.influence_self
                emotion_input = emotion_input + security_out.mean() * self.influence_peer
                
                security_input = user_input + security_out * self.influence_self
                security_input = security_input + emotion_out.mean() * self.influence_peer
                
                quality_input = user_input + quality_out * self.influence_self
                quality_input = quality_input + (emotion_out.mean() + security_out.mean()) * (self.influence_peer / 2)
                
                emotion_out = self.emotion(emotion_input)
                security_out = self.security(security_input)
                quality_out = self.quality(quality_input)
        
        return DanceResult(
            emotion_output=emotion_out,
            security_output=security_out,
            quality_output=quality_out,
            rounds=round_results,
            final_guidance=round_results[-1]
        )
    
    def train_step(
        self,
        user_input: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        rounds: int = 3
    ) -> Dict[str, float]:
        """
        Perform one training step with inter-agent consensus learning.
        
        Args:
            user_input: Input tensor (random if not provided)
            target: Target consensus (random if not provided)
            rounds: Number of discussion rounds
        
        Returns:
            Dictionary with loss components
        """
        if user_input is None:
            user_input = torch.randn(1, self.input_size, device=self.device)
        
        if target is None:
            target = torch.tanh(torch.randn(1, self.output_size, device=self.device) * 0.5)
        
        dance_result = self.dance(user_input, rounds=rounds, verbose=False)
        emotion_out, security_out, quality_out = dance_result.outputs
        
        emotion_norm = (emotion_out + 1) / 2
        security_norm = security_out
        quality_norm = torch.sigmoid(quality_out)
        
        target_norm = (target + 1) / 2
        
        loss_emotion = F.mse_loss(emotion_norm, target_norm)
        loss_security = F.mse_loss(security_norm, target_norm)
        loss_quality = F.mse_loss(quality_norm, target_norm)
        
        mean_output = (emotion_norm + security_norm + quality_norm) / 3
        
        consensus_loss = (
            F.mse_loss(emotion_norm, mean_output.detach()) +
            F.mse_loss(security_norm, mean_output.detach()) +
            F.mse_loss(quality_norm, mean_output.detach())
        ) / 3
        
        individual_loss = loss_emotion + loss_security + loss_quality
        total_loss = individual_loss + consensus_loss * self.consensus_weight
        
        self.opt_emotion.zero_grad()
        self.opt_security.zero_grad()
        self.opt_quality.zero_grad()
        
        total_loss.backward()
        
        self.opt_emotion.step()
        self.opt_security.step()
        self.opt_quality.step()
        
        return {
            "total_loss": total_loss.item(),
            "individual_loss": individual_loss.item(),
            "consensus_loss": consensus_loss.item(),
            "loss_emotion": loss_emotion.item(),
            "loss_security": loss_security.item(),
            "loss_quality": loss_quality.item(),
            "variance": dance_result.final_guidance.variance
        }
    
    def train(
        self,
        epochs: int = 50,
        rounds_per_epoch: int = 3,
        verbose: bool = True,
        print_every: int = 10
    ) -> List[EpochSummary]:
        """
        Train the swarm for multiple epochs with random data.
        
        Args:
            epochs: Number of training epochs
            rounds_per_epoch: Discussion rounds per dance
            verbose: Print progress
            print_every: Print interval
        
        Returns:
            List of epoch summaries
        """
        if verbose:
            print(f"\n{'='*60}")
            print("GUARDIAN SWARM DANCE - Training (Random)")
            print(f"{'='*60}")
            print(f"Epochs: {epochs}, Rounds/epoch: {rounds_per_epoch}")
            print(f"Agents: Emotion, Security, Quality")
            print(f"Total parameters: {self.count_parameters():,}")
            print(f"Device: {self.device}")
            print(f"{'='*60}\n")
        
        summaries = []
        
        for epoch in range(1, epochs + 1):
            losses = self.train_step(rounds=rounds_per_epoch)
            
            summary = self.guardian.complete_epoch(
                epoch=epoch,
                total_loss=losses["total_loss"],
                consensus_loss=losses["consensus_loss"]
            )
            summaries.append(summary)
            self.training_history.append(summary)
            
            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:3d}: Loss={losses['total_loss']:.4f}, "
                      f"Consensus={losses['consensus_loss']:.4f}, "
                      f"Variance={losses['variance']:.4f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Training Complete!")
            
            if len(summaries) >= 2:
                initial_var = summaries[0].final_variance
                final_var = summaries[-1].final_variance
                improvement = (1 - final_var / max(initial_var, 0.001)) * 100
                print(f"Variance: {initial_var:.4f} -> {final_var:.4f} ({improvement:+.1f}%)")
            
            print(f"{'='*60}\n")
        
        return summaries
    
    def train_supervised_step(
        self,
        user_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        rounds: int = 3
    ) -> Dict[str, float]:
        """
        Perform one supervised training step with specific targets per agent.
        
        Args:
            user_input: Input tensor
            targets: Dict with 'emotion', 'security', 'quality' target tensors
            rounds: Number of discussion rounds
        
        Returns:
            Dictionary with loss components
        """
        user_input = user_input.to(self.device)
        
        dance_result = self.dance(user_input, rounds=rounds, verbose=False)
        emotion_out, security_out, quality_out = dance_result.outputs
        
        emotion_scalar = emotion_out.mean()
        security_scalar = security_out.mean()
        quality_scalar = torch.sigmoid(quality_out.mean())
        
        emotion_target = torch.tensor(targets['emotion'], device=self.device).float()
        security_target = torch.tensor(targets['security'], device=self.device).float()
        quality_target = torch.tensor(targets['quality'], device=self.device).float()
        
        loss_emotion = F.mse_loss(emotion_scalar, emotion_target)
        loss_security = F.mse_loss(security_scalar, security_target)
        loss_quality = F.mse_loss(quality_scalar, quality_target)
        
        all_preds = torch.stack([emotion_scalar, security_scalar, quality_scalar])
        consensus_loss = all_preds.var()
        
        individual_loss = loss_emotion + loss_security + loss_quality
        total_loss = individual_loss + consensus_loss * self.consensus_weight
        
        self.opt_emotion.zero_grad()
        self.opt_security.zero_grad()
        self.opt_quality.zero_grad()
        
        total_loss.backward()
        
        self.opt_emotion.step()
        self.opt_security.step()
        self.opt_quality.step()
        
        return {
            "total_loss": total_loss.item(),
            "individual_loss": individual_loss.item(),
            "consensus_loss": consensus_loss.item(),
            "loss_emotion": loss_emotion.item(),
            "loss_security": loss_security.item(),
            "loss_quality": loss_quality.item(),
            "emotion_pred": emotion_scalar.item(),
            "security_pred": security_scalar.item(),
            "quality_pred": quality_scalar.item(),
            "variance": dance_result.final_guidance.variance
        }
    
    def train_supervised(
        self,
        dataset: Optional[TrainingDataGenerator] = None,
        epochs: int = 100,
        rounds_per_epoch: int = 3,
        verbose: bool = True,
        print_every: int = 10
    ) -> List[Dict[str, float]]:
        """
        Train the swarm with supervised learning on specialized dataset.
        
        Each agent learns its specific role:
        - Emotion: sentiment detection [-1, +1]
        - Security: risk assessment [0, 1]
        - Quality: code quality evaluation [0, 1]
        
        Args:
            dataset: TrainingDataGenerator (uses extended dataset if None)
            epochs: Number of training epochs
            rounds_per_epoch: Discussion rounds per dance
            verbose: Print progress
            print_every: Print interval
        
        Returns:
            List of epoch loss dictionaries
        """
        if dataset is None:
            dataset = create_extended_dataset()
        
        if verbose:
            print(f"\n{'='*70}")
            print("GUARDIAN SWARM DANCE - Supervised Training")
            print(f"{'='*70}")
            print(f"Training samples: {len(dataset)}")
            print(f"Epochs: {epochs}, Rounds/epoch: {rounds_per_epoch}")
            print(f"Agents: Emotion (tanh), Security (sigmoid), Quality (linear)")
            print(f"Total parameters: {self.count_parameters():,}")
            print(f"{'='*70}\n")
        
        history = []
        num_samples = len(dataset)
        
        for epoch in range(1, epochs + 1):
            epoch_losses = {
                "total_loss": 0.0,
                "loss_emotion": 0.0,
                "loss_security": 0.0,
                "loss_quality": 0.0,
                "consensus_loss": 0.0
            }
            
            for i in range(num_samples):
                sample_input, sample_targets, _ = dataset.get_sample(i)
                losses = self.train_supervised_step(
                    sample_input,
                    sample_targets,
                    rounds=rounds_per_epoch
                )
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key]
            
            for key in epoch_losses:
                epoch_losses[key] /= num_samples
            
            history.append(epoch_losses)
            
            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:3d}: Total={epoch_losses['total_loss']:.4f}, "
                      f"E={epoch_losses['loss_emotion']:.4f}, "
                      f"S={epoch_losses['loss_security']:.4f}, "
                      f"Q={epoch_losses['loss_quality']:.4f}")
        
        if verbose:
            print(f"\n{'='*70}")
            print("Supervised Training Complete!")
            if len(history) >= 2:
                initial_loss = history[0]['total_loss']
                final_loss = history[-1]['total_loss']
                improvement = (1 - final_loss / max(initial_loss, 0.001)) * 100
                print(f"Loss: {initial_loss:.4f} -> {final_loss:.4f} ({improvement:+.1f}%)")
            print(f"{'='*70}\n")
        
        return history
    
    def evaluate_sample(
        self,
        description: str,
        features: List[float],
        expected: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample and compare with expected outputs.
        
        Args:
            description: Human-readable description
            features: 10-dimensional feature vector
            expected: Optional dict with expected emotion/security/quality
            verbose: Print results
        
        Returns:
            Dictionary with predictions and optionally errors
        """
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.emotion.eval()
        self.security.eval()
        self.quality.eval()
        
        with torch.no_grad():
            result = self.dance(input_tensor, rounds=3, verbose=False)
        
        self.emotion.train()
        self.security.train()
        self.quality.train()
        
        emotion_pred = result.emotion_output.mean().item()
        security_pred = result.security_output.mean().item()
        quality_pred = torch.sigmoid(result.quality_output.mean()).item()
        
        output = {
            "description": description,
            "predictions": {
                "emotion": emotion_pred,
                "security": security_pred,
                "quality": quality_pred
            },
            "consensus": result.final_guidance.consensus,
            "variance": result.final_guidance.variance
        }
        
        if expected:
            output["expected"] = expected
            output["errors"] = {
                "emotion": abs(emotion_pred - expected.get('emotion', 0)),
                "security": abs(security_pred - expected.get('security', 0)),
                "quality": abs(quality_pred - expected.get('quality', 0))
            }
            output["mean_error"] = sum(output["errors"].values()) / 3
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Sample: {description[:60]}...")
            print(f"{'='*70}")
            print(f"  Emotion:  {emotion_pred:+.3f}", end="")
            if expected:
                print(f" (expected: {expected.get('emotion', 0):+.3f}, error: {output['errors']['emotion']:.3f})")
            else:
                print()
            print(f"  Security: {security_pred:.3f}", end="")
            if expected:
                print(f" (expected: {expected.get('security', 0):.3f}, error: {output['errors']['security']:.3f})")
            else:
                print()
            print(f"  Quality:  {quality_pred:.3f}", end="")
            if expected:
                print(f" (expected: {expected.get('quality', 0):.3f}, error: {output['errors']['quality']:.3f})")
            else:
                print()
            print(f"  Consensus: {result.final_guidance.consensus}, Variance: {result.final_guidance.variance:.4f}")
        
        return output
    
    def evaluate_dataset(
        self,
        dataset: Optional[TrainingDataGenerator] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the swarm on the entire dataset.
        
        Returns:
            Summary statistics of evaluation
        """
        if dataset is None:
            dataset = create_extended_dataset()
        
        results = []
        total_errors = {"emotion": 0.0, "security": 0.0, "quality": 0.0}
        
        for i in range(len(dataset)):
            sample_input, sample_targets, description = dataset.get_sample(i)
            
            result = self.evaluate_sample(
                description=description,
                features=sample_input.squeeze().tolist(),
                expected=sample_targets,
                verbose=False
            )
            results.append(result)
            
            for key in total_errors:
                total_errors[key] += result["errors"][key]
        
        num_samples = len(dataset)
        mean_errors = {k: v / num_samples for k, v in total_errors.items()}
        overall_error = sum(mean_errors.values()) / 3
        
        summary = {
            "num_samples": num_samples,
            "mean_errors": mean_errors,
            "overall_error": overall_error,
            "accuracy": 1 - overall_error,
            "results": results
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("EVALUATION SUMMARY")
            print(f"{'='*70}")
            print(f"Samples evaluated: {num_samples}")
            print(f"Mean Errors:")
            print(f"  Emotion Agent:  {mean_errors['emotion']:.4f}")
            print(f"  Security Agent: {mean_errors['security']:.4f}")
            print(f"  Quality Agent:  {mean_errors['quality']:.4f}")
            print(f"Overall Error: {overall_error:.4f}")
            print(f"Accuracy: {summary['accuracy']*100:.1f}%")
            print(f"{'='*70}\n")
        
        return summary
    
    def infer(
        self,
        user_input: torch.Tensor,
        rounds: int = 3,
        verbose: bool = True
    ) -> DanceResult:
        """
        Run inference (no training) on an input.
        
        Args:
            user_input: Input tensor
            rounds: Discussion rounds
            verbose: Print details
        
        Returns:
            DanceResult with outputs and consensus info
        """
        self.emotion.eval()
        self.security.eval()
        self.quality.eval()
        
        with torch.no_grad():
            result = self.dance(user_input, rounds=rounds, verbose=verbose)
        
        self.emotion.train()
        self.security.train()
        self.quality.train()
        
        return result
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return (
            self.emotion.count_parameters() +
            self.security.count_parameters() +
            self.quality.count_parameters()
        )
    
    def save(self, filepath: str) -> None:
        """Save swarm state to file."""
        torch.save({
            "emotion_state": self.emotion.state_dict(),
            "security_state": self.security.state_dict(),
            "quality_state": self.quality.state_dict(),
            "opt_emotion_state": self.opt_emotion.state_dict(),
            "opt_security_state": self.opt_security.state_dict(),
            "opt_quality_state": self.opt_quality.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "consensus_weight": self.consensus_weight,
                "influence_self": self.influence_self,
                "influence_peer": self.influence_peer,
            },
            "training_epochs": len(self.training_history),
            "guardian_stats": self.guardian.get_stats(),
        }, filepath)
        print(f"Swarm saved to: {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load swarm state from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.emotion.load_state_dict(checkpoint["emotion_state"])
        self.security.load_state_dict(checkpoint["security_state"])
        self.quality.load_state_dict(checkpoint["quality_state"])
        self.opt_emotion.load_state_dict(checkpoint["opt_emotion_state"])
        self.opt_security.load_state_dict(checkpoint["opt_security_state"])
        self.opt_quality.load_state_dict(checkpoint["opt_quality_state"])
        
        print(f"Swarm loaded from: {filepath}")
        print(f"  Previous training epochs: {checkpoint.get('training_epochs', 'unknown')}")
    
    def get_agent_states(self) -> Dict[str, Any]:
        """Get current agent states for debugging."""
        return {
            "emotion": {
                "personality": self.emotion.personality,
                "parameters": self.emotion.count_parameters(),
            },
            "security": {
                "personality": self.security.personality,
                "parameters": self.security.count_parameters(),
            },
            "quality": {
                "personality": self.quality.personality,
                "parameters": self.quality.count_parameters(),
            },
            "total_parameters": self.count_parameters(),
            "device": str(self.device),
        }


if __name__ == "__main__":
    print("SwarmDance - Test")
    print("=" * 60)
    
    swarm = SwarmDance()
    print(f"\nCreated swarm with {swarm.count_parameters():,} parameters")
    print(f"Device: {swarm.device}")
    
    print("\n--- Single Dance Test ---")
    test_input = torch.randn(1, 10)
    result = swarm.dance(test_input, rounds=3, verbose=True)
    print(f"Final guidance: {result.final_guidance}")
    
    print("\n--- Training Test (10 epochs) ---")
    swarm.train(epochs=10, verbose=True, print_every=5)
    
    print("\n--- Post-Training Inference ---")
    result = swarm.infer(test_input, rounds=3, verbose=True)
    print(f"Final outputs: {result.mean_outputs}")
