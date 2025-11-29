"""
Guardian Angel - Meta-Overseer for the Swarm

The Guardian Angel is NOT a neural network. It's a rule-based monitor that:
1. Observes all agent outputs
2. Calculates consensus metrics
3. Provides guidance (proceed/discuss)
4. Maintains history for pattern detection
"""

import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class GuidanceResult:
    """Result of Guardian evaluation"""
    variance: float
    consensus: bool
    recommendation: str
    agent_means: Dict[str, float]
    round_number: int = 0
    
    def __repr__(self) -> str:
        status = "CONSENSUS" if self.consensus else "DISCUSS"
        return f"[{status}] var={self.variance:.4f}, rec={self.recommendation}"


@dataclass
class EpochSummary:
    """Summary of a complete training epoch"""
    epoch: int
    initial_variance: float
    final_variance: float
    convergence_rate: float
    total_loss: float
    consensus_loss: float
    rounds: List[GuidanceResult] = field(default_factory=list)
    
    def __repr__(self) -> str:
        improvement = (1 - self.final_variance / max(self.initial_variance, 0.001)) * 100
        return f"Epoch {self.epoch}: var {self.initial_variance:.4f} -> {self.final_variance:.4f} ({improvement:+.1f}%)"


class GuardianAngel:
    """
    Meta-overseer that watches and guides the swarm.
    
    Responsibilities:
    - Monitor agent outputs for agreement
    - Calculate variance (disagreement metric)
    - Recommend actions based on consensus level
    - Track history for pattern detection
    """
    
    def __init__(
        self,
        consensus_threshold: float = 0.7,
        proceed_threshold: float = 0.5,
        max_history: int = 1000
    ):
        self.consensus_threshold = consensus_threshold
        self.proceed_threshold = proceed_threshold
        self.max_history = max_history
        
        self.history: List[GuidanceResult] = []
        self.epoch_summaries: List[EpochSummary] = []
        self.current_epoch_rounds: List[GuidanceResult] = []
    
    def evaluate(
        self,
        emotion_out: torch.Tensor,
        security_out: torch.Tensor,
        quality_out: torch.Tensor,
        round_number: int = 0
    ) -> GuidanceResult:
        """
        Evaluate the current state of agent outputs.
        
        Args:
            emotion_out: Output from EmotionAgent (tanh, -1 to +1)
            security_out: Output from SecurityAgent (sigmoid, 0 to 1)
            quality_out: Output from QualityAgent (linear, unbounded)
            round_number: Current discussion round
        
        Returns:
            GuidanceResult with variance, consensus status, and recommendation
        """
        all_outputs = torch.stack([emotion_out, security_out, quality_out])
        variance = all_outputs.var(dim=0).mean().item()
        
        consensus = variance < self.consensus_threshold
        
        if variance < self.proceed_threshold:
            recommendation = "proceed"
        elif variance < self.consensus_threshold:
            recommendation = "refine"
        else:
            recommendation = "discuss"
        
        agent_means = {
            "emotion": emotion_out.mean().item(),
            "security": security_out.mean().item(),
            "quality": quality_out.mean().item()
        }
        
        result = GuidanceResult(
            variance=variance,
            consensus=consensus,
            recommendation=recommendation,
            agent_means=agent_means,
            round_number=round_number
        )
        
        self.history.append(result)
        self.current_epoch_rounds.append(result)
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        return result
    
    def complete_epoch(
        self,
        epoch: int,
        total_loss: float,
        consensus_loss: float
    ) -> EpochSummary:
        """
        Complete an epoch and generate summary.
        
        Args:
            epoch: Epoch number
            total_loss: Total training loss
            consensus_loss: Consensus component of loss
        
        Returns:
            EpochSummary for this epoch
        """
        if not self.current_epoch_rounds:
            initial_var = 0.0
            final_var = 0.0
        else:
            initial_var = self.current_epoch_rounds[0].variance
            final_var = self.current_epoch_rounds[-1].variance
        
        if initial_var > 0:
            convergence_rate = 1 - (final_var / initial_var)
        else:
            convergence_rate = 0.0
        
        summary = EpochSummary(
            epoch=epoch,
            initial_variance=initial_var,
            final_variance=final_var,
            convergence_rate=convergence_rate,
            total_loss=total_loss,
            consensus_loss=consensus_loss,
            rounds=self.current_epoch_rounds.copy()
        )
        
        self.epoch_summaries.append(summary)
        self.current_epoch_rounds = []
        
        return summary
    
    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze recent trends in agent behavior.
        
        Args:
            window: Number of recent epochs to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.epoch_summaries) < 2:
            return {"status": "insufficient_data", "epochs": len(self.epoch_summaries)}
        
        recent = self.epoch_summaries[-window:]
        
        variances = [s.final_variance for s in recent]
        avg_variance = sum(variances) / len(variances)
        
        if len(variances) >= 2:
            trend = variances[-1] - variances[0]
            if trend < -0.01:
                direction = "improving"
            elif trend > 0.01:
                direction = "degrading"
            else:
                direction = "stable"
        else:
            direction = "unknown"
        
        return {
            "status": "ok",
            "epochs_analyzed": len(recent),
            "avg_variance": avg_variance,
            "trend_direction": direction,
            "latest_variance": variances[-1] if variances else 0,
            "best_variance": min(variances) if variances else 0
        }
    
    def reset(self) -> None:
        """Reset all history."""
        self.history = []
        self.epoch_summaries = []
        self.current_epoch_rounds = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        if not self.epoch_summaries:
            return {"status": "no_epochs"}
        
        variances = [s.final_variance for s in self.epoch_summaries]
        consensus_rates = [s.convergence_rate for s in self.epoch_summaries]
        
        return {
            "total_epochs": len(self.epoch_summaries),
            "total_evaluations": len(self.history),
            "avg_final_variance": sum(variances) / len(variances),
            "min_variance": min(variances),
            "max_variance": max(variances),
            "avg_convergence_rate": sum(consensus_rates) / len(consensus_rates),
            "consensus_threshold": self.consensus_threshold
        }


if __name__ == "__main__":
    print("Guardian Angel - Test")
    print("=" * 50)
    
    guardian = GuardianAngel()
    
    emotion_out = torch.randn(1, 10) * 0.5
    security_out = torch.sigmoid(torch.randn(1, 10))
    quality_out = torch.randn(1, 10) * 0.3
    
    for round_num in range(4):
        result = guardian.evaluate(emotion_out, security_out, quality_out, round_num)
        print(f"Round {round_num}: {result}")
        
        emotion_out = emotion_out * 0.9 + 0.1 * quality_out
        quality_out = quality_out * 0.9 + 0.1 * emotion_out
    
    summary = guardian.complete_epoch(1, total_loss=0.5, consensus_loss=0.1)
    print(f"\n{summary}")
    
    print(f"\nStats: {guardian.get_stats()}")
