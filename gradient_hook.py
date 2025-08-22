import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class GradientMonitor:
    """
    Gradient monitor class for tracking gradients during VAE training
    """
    
    def __init__(self, model, hook_layers=None, track_norms=True, track_distributions=True):
        """
        Initialize gradient monitor
        
        Args:
            model: VAE model instance
            hook_layers: List of layer names to hook (None for all layers)
            track_norms: Whether to track gradient norms
            track_distributions: Whether to track gradient distributions
        """
        self.model = model
        self.hook_layers = hook_layers
        self.track_norms = track_norms
        self.track_distributions = track_distributions
        
        # Storage for gradient statistics
        self.gradient_norms = defaultdict(list)
        self.gradient_stats = defaultdict(lambda: {'mean': [], 'std': [], 'max': [], 'min': []})
        self.gradient_histograms = defaultdict(list)
        
        # Hook handles for cleanup
        self.hook_handles = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register backward hooks on specified layers"""
        
        def create_hook(name):
            def hook_fn(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    
                    if self.track_norms:
                        # Calculate gradient norm
                        grad_norm = torch.norm(grad).item()
                        self.gradient_norms[name].append(grad_norm)
                    
                    if self.track_distributions:
                        # Calculate gradient statistics
                        grad_flat = grad.flatten()
                        self.gradient_stats[name]['mean'].append(torch.mean(grad_flat).item())
                        self.gradient_stats[name]['std'].append(torch.std(grad_flat).item())
                        self.gradient_stats[name]['max'].append(torch.max(grad_flat).item())
                        self.gradient_stats[name]['min'].append(torch.min(grad_flat).item())
                        
                        # Store histogram data (sample to avoid memory issues)
                        if len(grad_flat) > 1000:
                            sampled_grad = grad_flat[torch.randperm(len(grad_flat))[:1000]]
                        else:
                            sampled_grad = grad_flat
                        self.gradient_histograms[name].append(sampled_grad.detach().cpu().numpy())
            
            return hook_fn
        
        # Register hooks on specified layers or all layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self.hook_layers is None or name in self.hook_layers:
                    handle = module.register_backward_hook(create_hook(name))
                    self.hook_handles.append(handle)
                    print(f"Registered hook on layer: {name}")
    
    def clear_statistics(self):
        """Clear all stored gradient statistics"""
        self.gradient_norms.clear()
        self.gradient_stats.clear()
        self.gradient_histograms.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
    
    def get_gradient_norms(self, layer_name=None):
        """Get gradient norms for a specific layer or all layers"""
        if layer_name:
            return self.gradient_norms.get(layer_name, [])
        return dict(self.gradient_norms)
    
    def get_gradient_stats(self, layer_name=None):
        """Get gradient statistics for a specific layer or all layers"""
        if layer_name:
            return self.gradient_stats.get(layer_name, {})
        return dict(self.gradient_stats)
    
    def plot_gradient_norms(self, figsize=(12, 8), show_recent=None):
        """Plot gradient norms over training steps"""
        if not self.gradient_norms:
            print("No gradient norm data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        layer_names = list(self.gradient_norms.keys())
        
        for i, layer_name in enumerate(layer_names[:4]):  # Show first 4 layers
            if i >= len(axes):
                break
                
            norms = self.gradient_norms[layer_name]
            if show_recent:
                norms = norms[-show_recent:]
            
            axes[i].plot(norms, alpha=0.7)
            axes[i].set_title(f'Gradient Norms: {layer_name}')
            axes[i].set_xlabel('Training Step')
            axes[i].set_ylabel('Gradient Norm')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')  # Log scale for better visualization
        
        # Hide unused subplots
        for i in range(len(layer_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distributions(self, layer_name, figsize=(12, 4)):
        """Plot gradient distribution evolution for a specific layer"""
        if layer_name not in self.gradient_histograms:
            print(f"No histogram data for layer: {layer_name}")
            return
        
        histograms = self.gradient_histograms[layer_name]
        if len(histograms) < 2:
            print("Not enough data for distribution plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Show first, middle, and last distributions
        indices = [0, len(histograms)//2, -1]
        labels = ['Early Training', 'Mid Training', 'Late Training']
        
        for i, (idx, label) in enumerate(zip(indices, labels)):
            axes[i].hist(histograms[idx], bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'{label}\n{layer_name}')
            axes[i].set_xlabel('Gradient Value')
            axes[i].set_ylabel('Density')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def detect_gradient_issues(self, threshold_vanishing=1e-6, threshold_exploding=10.0):
        """Detect gradient vanishing or exploding issues"""
        issues = {}
        
        for layer_name, norms in self.gradient_norms.items():
            if not norms:
                continue
                
            recent_norms = norms[-10:] if len(norms) >= 10 else norms
            avg_norm = np.mean(recent_norms)
            
            if avg_norm < threshold_vanishing:
                issues[layer_name] = f"Vanishing gradients (avg norm: {avg_norm:.2e})"
            elif avg_norm > threshold_exploding:
                issues[layer_name] = f"Exploding gradients (avg norm: {avg_norm:.2e})"
        
        return issues
    
    def print_gradient_summary(self):
        """Print summary of gradient statistics"""
        print("Gradient Monitor Summary")
        print(f"Tracking {len(self.gradient_norms)} layers\n")
        
        for layer_name in self.gradient_norms.keys():
            norms = self.gradient_norms[layer_name]
            if norms:
                avg_norm = np.mean(norms)
                min_norm = np.min(norms)
                max_norm = np.max(norms)
                
                # Classify gradient status
                if avg_norm < 1e-6:
                    status = "VANISH"
                elif avg_norm > 10:
                    status = "EXPLODE"
                else:
                    status = "NORMAL"
                
                print(f"{layer_name:<15} | {status:<7} | Avg: {avg_norm:.4f} | Range: [{min_norm:.4f}, {max_norm:.4f}]")
        
        # Check for issues
        issues = self.detect_gradient_issues()
        if issues:
            print(f"\nFound {len(issues)} potential issues:")
            for layer, issue in issues.items():
                print(f"   {layer}: {issue}")
        else:
            print("\nAll gradients are normal")



'''            
model = VAE(input_dim=100, hidden_dims=[64, 32], latent_dim=16)

# Create gradient monitor
grad_monitor = GradientMonitor(
    model, 
    hook_layers=['encoder.0', 'encoder.3', 'decoder.0', 'decoder.3'],  # Specific layers
    track_norms=True,
    track_distributions=True
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for step in range(10):
        # Dummy data
        x = torch.randn(32, 100)
        
        # Forward pass
        recon, mu, log_var = model(x)
        
        # Calculate loss
        loss, recon_loss, kl_loss = VAE.vae_loss(recon, x, mu, log_var)
        
        # Backward pass (this triggers the hooks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Analyze gradients
grad_monitor.print_gradient_summary()
grad_monitor.plot_gradient_norms()

# Check for specific layer
if 'encoder.0' in grad_monitor.gradient_norms:
    grad_monitor.plot_gradient_distributions('encoder.0')

# Clean up when done
grad_monitor.remove_hooks()
'''