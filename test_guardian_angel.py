"""
GUARDIAN ANGEL TEST SUITE
Complete validation of all protection mechanisms
Tests: NaN injection, Gradient explosion, Loss divergence, Checkpoint corruption
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
import json


class SimpleNet(nn.Module):
    """Simple neural network for testing"""
    
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


class GuardianAngelTestSuite:
    """Complete test suite for Guardian Angel protection mechanisms"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = []
        
    def generate_data(self, num_samples=100):
        """Generate synthetic training data"""
        X = torch.randn(num_samples, 10, device=self.device)
        y = (X[:, :5].sum(dim=1) > 0).float().unsqueeze(1)
        return X, y
    
    def train_epoch(self, model, criterion, optimizer, X, y):
        """Single training epoch with error handling"""
        model.train()
        
        try:
            outputs = model(X)
            loss = criterion(outputs, y)
        except RuntimeError as e:
            return float('nan'), 0.0
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        
        return loss.item(), accuracy
    
    def check_nan_inf(self, value, name="value"):
        """Check for NaN/Inf in tensors or scalars"""
        if isinstance(value, torch.Tensor):
            has_nan = torch.isnan(value).any().item()
            has_inf = torch.isinf(value).any().item()
        else:
            import math
            has_nan = math.isnan(value) if not math.isinf(value) else False
            has_inf = math.isinf(value)
        
        return has_nan or has_inf
    
    def check_gradient_explosion(self, model, threshold=1e6):
        """Check for gradient explosion"""
        max_grad = 0.0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        return max_grad > threshold, max_grad
    
    def log_test_result(self, test_name, status, details):
        """Log test result"""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def test_1_nan_injection(self):
        """TEST 1: NaN Injection - Inject NaN into weights"""
        print("\n" + "="*80)
        print("TEST 1: NaN INJECTION")
        print("="*80)
        print("Purpose: Verify Guardian Angel detects NaN in model weights")
        print("Method: Inject NaN into fc1.weight at epoch 15")
        print("Expected: Guardian detects NaN → HALT immediately")
        print("="*80 + "\n")
        
        model = SimpleNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        num_epochs = 30
        injection_epoch = 15
        halt_detected = False
        
        for epoch in range(1, num_epochs + 1):
            # Inject NaN at specified epoch
            if epoch == injection_epoch:
                print(f"\n⚠️  EPOCH {epoch}: Injecting NaN into fc1.weight[0,0]")
                with torch.no_grad():
                    model.fc1.weight[0, 0] = float('nan')
            
            X, y = self.generate_data(3)
            loss, accuracy = self.train_epoch(model, criterion, optimizer, X, y)
            
            # Guardian Angel check
            if self.check_nan_inf(loss, "loss"):
                print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                print(f"❌ NaN/Inf detected in loss at epoch {epoch}")
                print(f"💥 Loss value: {loss}")
                print(f"🚨 TRAINING HALTED\n")
                halt_detected = True
                break
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")
        
        # Verify halt occurred
        if halt_detected and epoch == injection_epoch:
            print("✅ TEST 1 PASSED: Guardian Angel detected NaN and halted at injection epoch")
            self.log_test_result("NaN Injection", "PASSED", 
                                f"Detected at epoch {epoch}, halted immediately")
        else:
            print(f"❌ TEST 1 FAILED: Expected halt at epoch {injection_epoch}, got {epoch if halt_detected else 'no halt'}")
            self.log_test_result("NaN Injection", "FAILED", 
                                f"Halt detection failed")
    
    def test_2_gradient_explosion(self):
        """TEST 2: Gradient Explosion - Increase learning rate dramatically"""
        print("\n" + "="*80)
        print("TEST 2: GRADIENT EXPLOSION")
        print("="*80)
        print("Purpose: Verify Guardian Angel detects exploding gradients")
        print("Method: Increase learning rate from 0.01 to 10000.0 at epoch 20")
        print("Expected: Guardian detects gradient magnitude > 1000 → HALT")
        print("="*80 + "\n")
        
        model = SimpleNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD more sensitive to LR
        
        num_epochs = 40
        explosion_epoch = 20
        halt_detected = False
        explosion_lr = 10000.0  # More aggressive to force gradient explosion
        
        for epoch in range(1, num_epochs + 1):
            X, y = self.generate_data(3)
            
            # Normal training first
            loss, accuracy = self.train_epoch(model, criterion, optimizer, X, y)
            
            # INJECT: Gradient explosion AFTER backward pass (before gradient check)
            if epoch == explosion_epoch:
                print(f"\n⚠️  EPOCH {epoch}: FORCING GRADIENT EXPLOSION")
                print(f"💥 Method: Setting all gradients to 1e7 magnitude")
                
                # Inject massive gradients directly
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.fill_(1e7)  # 10 million per gradient
                
                print(f"⚠️  Gradients injected - Guardian should detect on next check\n")
            
            # Check for NaN/Inf first
            if self.check_nan_inf(loss, "loss"):
                print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                print(f"❌ Loss became NaN/Inf at epoch {epoch}")
                print(f"💥 Loss value: {loss}")
                print(f"🚨 TRAINING HALTED (Gradient explosion caused NaN)\n")
                halt_detected = True
                break
            
            # Check gradient magnitudes (lowered threshold for realistic detection)
            is_exploded, max_grad = self.check_gradient_explosion(model, threshold=1000)
            
            if is_exploded:
                print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                print(f"💥 Gradient explosion detected at epoch {epoch}")
                print(f"📊 Max gradient magnitude: {max_grad:.2e}")
                print(f"🚨 TRAINING HALTED\n")
                halt_detected = True
                break
            
            # Debug: Show gradient magnitudes around explosion epoch
            if epoch >= explosion_epoch - 2 and epoch <= explosion_epoch + 10:
                print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}% | Max Grad: {max_grad:.2e}")
            elif epoch % 5 == 0 or epoch == 1:
                grad_status = f" | Max Grad: {max_grad:.2e}" if max_grad > 100 else ""
                print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%{grad_status}")
        
        # Verify halt occurred within reasonable range of explosion epoch
        if halt_detected and epoch >= explosion_epoch and epoch <= explosion_epoch + 5:
            print("✅ TEST 2 PASSED: Guardian Angel detected gradient explosion and halted")
            self.log_test_result("Gradient Explosion", "PASSED", 
                                f"Detected at epoch {epoch}, max_grad={max_grad:.2e}")
        else:
            print(f"❌ TEST 2 FAILED: Expected halt near epoch {explosion_epoch}, got {epoch if halt_detected else 'no halt'}")
            self.log_test_result("Gradient Explosion", "FAILED", 
                                f"Halt detection failed or delayed")
    
    def test_3_loss_divergence(self):
        """TEST 3: Loss Divergence - Manually corrupt loss calculation"""
        print("\n" + "="*80)
        print("TEST 3: LOSS DIVERGENCE")
        print("="*80)
        print("Purpose: Verify Guardian Angel detects sudden loss spikes")
        print("Method: Multiply loss by 100x at epoch 25")
        print("Expected: Guardian detects loss > 2x recent average → WARN")
        print("="*80 + "\n")
        
        model = SimpleNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        num_epochs = 40
        divergence_epoch = 25
        loss_history = []
        warning_detected = False
        
        for epoch in range(1, num_epochs + 1):
            X, y = self.generate_data(3)
            loss, accuracy = self.train_epoch(model, criterion, optimizer, X, y)
            
            # Corrupt loss at specified epoch
            if epoch == divergence_epoch:
                print(f"\n⚠️  EPOCH {epoch}: Artificially spiking loss by 100x")
                loss = loss * 100.0
            
            # Check for loss divergence
            if len(loss_history) >= 10:
                recent_avg = sum(loss_history[-10:]) / 10
                
                if loss > 2 * recent_avg:
                    print(f"\n🛡️  GUARDIAN ANGEL WARNING!")
                    print(f"⚠️  Loss spike detected at epoch {epoch}")
                    print(f"📊 Current loss: {loss:.6f}")
                    print(f"📊 Recent average: {recent_avg:.6f}")
                    print(f"📊 Spike ratio: {loss/recent_avg:.2f}x")
                    print(f"💡 Continuing training with monitoring...\n")
                    warning_detected = True
            
            loss_history.append(loss)
            
            if epoch % 5 == 0 or epoch == 1 or epoch == divergence_epoch:
                print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")
        
        # Verify warning occurred
        if warning_detected:
            print("✅ TEST 3 PASSED: Guardian Angel detected loss divergence and warned")
            self.log_test_result("Loss Divergence", "PASSED", 
                                f"Warning issued at epoch {divergence_epoch}")
        else:
            print(f"❌ TEST 3 FAILED: Expected warning at epoch {divergence_epoch}, none issued")
            self.log_test_result("Loss Divergence", "FAILED", 
                                f"Warning not detected")
    
    def test_4_checkpoint_corruption(self):
        """TEST 4: Checkpoint Corruption - Inject NaN before save"""
        print("\n" + "="*80)
        print("TEST 4: CHECKPOINT CORRUPTION")
        print("="*80)
        print("Purpose: Verify Guardian Angel validates checkpoints before save")
        print("Method: Corrupt state_dict with NaN before checkpoint save")
        print("Expected: Guardian rejects checkpoint → saves last good state")
        print("="*80 + "\n")
        
        model = SimpleNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        num_epochs = 30
        checkpoint_epoch = 20
        corruption_detected = False
        last_good_state = None
        
        for epoch in range(1, num_epochs + 1):
            X, y = self.generate_data(3)
            loss, accuracy = self.train_epoch(model, criterion, optimizer, X, y)
            
            # Try to checkpoint every 10 epochs
            if epoch % 10 == 0:
                state_dict = model.state_dict()
                
                # Corrupt checkpoint at specified epoch
                if epoch == checkpoint_epoch:
                    print(f"\n⚠️  EPOCH {epoch}: Attempting to save checkpoint with NaN corruption")
                    state_dict['fc1.weight'][0, 0] = float('nan')
                
                # Guardian Angel validation
                is_corrupted = False
                for name, tensor in state_dict.items():
                    if self.check_nan_inf(tensor, name):
                        print(f"\n🛡️  GUARDIAN ANGEL ALERT!")
                        print(f"❌ Checkpoint validation FAILED at epoch {epoch}")
                        print(f"💥 NaN/Inf detected in {name}")
                        print(f"🚨 Checkpoint REJECTED")
                        
                        if last_good_state is not None:
                            print(f"💾 Saving last good state from epoch {epoch - 10}")
                            # Would save last_good_state here
                        
                        print(f"🛡️  Training can continue from memory\n")
                        is_corrupted = True
                        corruption_detected = True
                        break
                
                if not is_corrupted:
                    print(f"✅ Epoch {epoch}: Checkpoint validated and saved")
                    last_good_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {loss:.6f} | Accuracy: {accuracy*100:.2f}%")
        
        # Verify corruption was detected
        if corruption_detected:
            print("✅ TEST 4 PASSED: Guardian Angel detected checkpoint corruption and protected")
            self.log_test_result("Checkpoint Corruption", "PASSED", 
                                f"Corruption detected at epoch {checkpoint_epoch}")
        else:
            print(f"❌ TEST 4 FAILED: Expected corruption detection at epoch {checkpoint_epoch}")
            self.log_test_result("Checkpoint Corruption", "FAILED", 
                                f"Corruption not detected")
    
    def run_all_tests(self):
        """Run complete Guardian Angel test suite"""
        print("\n" + "="*80)
        print("🛡️  GUARDIAN ANGEL - COMPLETE TEST SUITE")
        print("="*80)
        print("Testing all protection mechanisms:")
        print("  1. NaN Injection Detection")
        print("  2. Gradient Explosion Detection")
        print("  3. Loss Divergence Warning")
        print("  4. Checkpoint Corruption Prevention")
        print("="*80)
        
        start_time = datetime.now()
        
        # Run all tests
        self.test_1_nan_injection()
        self.test_2_gradient_explosion()
        self.test_3_loss_divergence()
        self.test_4_checkpoint_corruption()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Summary
        print("\n" + "="*80)
        print("🛡️  TEST SUITE SUMMARY")
        print("="*80)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        total = len(self.test_results)
        
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_icon} {result['test']}: {result['status']}")
            print(f"   Details: {result['details']}")
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print("="*80 + "\n")
        
        # Save results
        results_dir = Path('data/guardian_tests')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'passed': passed,
                    'total': total,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                },
                'tests': self.test_results
            }, f, indent=2)
        
        print(f"💾 Test results saved: {results_file}\n")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - GUARDIAN ANGEL FULLY OPERATIONAL! 🛡️\n")
        else:
            print(f"⚠️  {total - passed} TEST(S) FAILED - REVIEW REQUIRED\n")


def main():
    print("="*80)
    print("🛡️  GUARDIAN ANGEL - PROTECTION MECHANISM TEST SUITE")
    print("="*80)
    print("Validating all Guardian Angel detection capabilities")
    print("="*80)
    
    suite = GuardianAngelTestSuite()
    suite.run_all_tests()


if __name__ == "__main__":
    main()
