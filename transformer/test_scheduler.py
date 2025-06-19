import torch
import math
import matplotlib.pyplot as plt

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

# Test the scheduler
dummy_param = torch.nn.Parameter(torch.randn(1))
optimizer = torch.optim.Adam([dummy_param], lr=0.001)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=100, eta_min=1e-6)

lrs = []
epochs = []

print("Testing new CosineWarmupScheduler:")
print("Epoch\tLearning Rate")
print("-" * 25)

for epoch in range(100):
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    epochs.append(epoch)
    
    if epoch < 10 or epoch % 10 == 0 or epoch > 95:
        print(f"{epoch:3d}\t{current_lr:.2e}")
    
    scheduler.step()

print(f"\nMinimum LR reached: {min(lrs):.2e}")
print(f"Maximum LR reached: {max(lrs):.2e}")
print(f"Final LR: {lrs[-1]:.2e}")

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('CosineWarmupScheduler Learning Rate Schedule')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='eta_min')
plt.legend()
plt.tight_layout()
plt.savefig('scheduler_test.png', dpi=150)
plt.show()

print("\nScheduler test complete! Check scheduler_test.png for the learning rate curve.") 