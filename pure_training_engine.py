"""
PURE NEURAL NETWORK TRAINING ENGINE
Real PyTorch training with epochs, loss, backpropagation
NO compression - just pure ML training mechanics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import json
from pathlib import Path


class SimpleNet(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class PureTrainingEngine:
    """Pure training engine - shows real ML training process"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Device: {self.device}")
        
        # Model
        self.model = SimpleNet(input_size=10, hidden_size=20, output_size=1).to(self.device)
        
        # Training components
        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        # History
        self.history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
            'learning_rates': []
        }
        
        print(f"✅ Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def generate_training_data(self, num_samples=100):
        """Generate synthetic training data"""
        X = torch.randn(num_samples, 10, device=self.device)
        # Simple rule: if sum of first 5 features > 0, label = 1
        y = (X[:, :5].sum(dim=1) > 0).float().unsqueeze(1)
        return X, y
    
    def train_epoch(self, X, y):
        """Train for one epoch"""
        self.model.train()
        
        # Forward pass
        try:
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        except RuntimeError as e:
            # Guardian Angel catches runtime errors (NaN propagation)
            return float('nan'), 0.0
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        
        return loss.item(), accuracy
    
    def train(self, num_epochs=80, num_samples=3):
        """Main training loop"""
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING PURE NEURAL NETWORK TRAINING")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Samples per epoch: {num_samples}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            # GRADIENT EXPLOSION TEST - Inject at epoch 30
            if epoch == 30:
                print(f"\n⚠️  INJECTING GRADIENT EXPLOSION TEST (Epoch {epoch})")
                print(f"💥 Injecting NaN into model weights")
                # Force NaN into first layer weights
                with torch.no_grad():
                    self.model.fc1.weight[0, 0] = float('nan')
                print(f"⚠️  Expected: Guardian detects NaN → HALT\n")
            
            # Generate fresh data each epoch
            X, y = self.generate_training_data(num_samples)
            
            # Train
            loss, accuracy = self.train_epoch(X, y)
            
            # Check for NaN/Inf (Guardian Angel would catch this)
            if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                print(f"❌ NaN/Inf detected in loss at epoch {epoch}")
                print(f"💥 Loss value: {loss}")
                print(f"🚨 TRAINING HALTED - Gradient explosion detected!")
                print(f"\n📊 Last valid epoch: {epoch - 1}")
                print(f"📊 Last valid loss: {self.history['losses'][-1] if self.history['losses'] else 'N/A'}")
                break
            
            # Check gradients for explosion (Guardian Angel monitoring)
            max_grad = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    max_grad = max(max_grad, param.grad.abs().max().item())
            
            if max_grad > 1e6:
                print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                print(f"💥 Gradient explosion detected at epoch {epoch}")
                print(f"📊 Max gradient magnitude: {max_grad:.2e}")
                print(f"🚨 TRAINING HALTED - Preventing NaN propagation!")
                print(f"\n📊 Last valid epoch: {epoch - 1}")
                print(f"📊 Last valid loss: {self.history['losses'][-1] if self.history['losses'] else 'N/A'}")
                break
            
            # Record
            self.history['epochs'].append(epoch)
            self.history['losses'].append(loss)
            self.history['accuracies'].append(accuracy)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print every 10 epochs (or if gradients are suspiciously large)
            if epoch % 10 == 0 or epoch == 1 or max_grad > 1000:
                grad_status = f" | Max Grad: {max_grad:.2e}" if max_grad > 1000 else ""
                print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%{grad_status}")
        
        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final Loss:     {self.history['losses'][-1]:.6f}")
        print(f"Final Accuracy: {self.history['accuracies'][-1]*100:.2f}%")
        print(f"Best Loss:      {min(self.history['losses']):.6f}")
        print(f"Best Accuracy:  {max(self.history['accuracies'])*100:.2f}%")
        print(f"{'='*80}\n")
        
        # Save
        self.save_results()
    
    def save_results(self):
        """Save training results"""
        results_dir = Path('data/training_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save history
        history_path = results_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 History saved: {history_path}")
        
        # Save model
        model_path = results_dir / f'model_{timestamp}.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, model_path)
        print(f"💾 Model saved: {model_path}")


def main():
    print("="*80)
    print("🔥 PURE NEURAL NETWORK TRAINING ENGINE")
    print("="*80)
    print("Real PyTorch training with:")
    print("  ✓ Forward propagation")
    print("  ✓ Loss calculation")
    print("  ✓ Backpropagation")
    print("  ✓ Gradient descent")
    print("  ✓ Weight updates")
    print("="*80)
    
    engine = PureTrainingEngine()
    engine.train(num_epochs=80, num_samples=3)
    
    print("\n✅ ALL DONE!")


if __name__ == "__main__":
    main()
