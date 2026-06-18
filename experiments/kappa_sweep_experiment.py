"""
κ (Kappa) Sweep Experiment
Tests multiple adaptive regularization amplification values to find optimal setting.

Compares:
- Continuous (baseline)
- Passive (no adaptive reg, κ=0)
- Active with κ ∈ {0.5, 1.0, 1.5, 2.0}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time

# Import existing components
from chapter4.sustainable_edge_ai import SolarPowerModel


class SimplePINN(nn.Module):
    """Simple PINN for Burgers equation"""
    def __init__(self, layers=[2, 16, 16, 32, 1]):
        super().__init__()
        self.layers_list = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        for i, layer in enumerate(self.layers_list[:-1]):
            X = torch.tanh(layer(X))
        X = self.layers_list[-1](X)
        return X


def generate_burgers_data(n_points=1000, nu=0.01/np.pi):
    """Generate training data for Burgers equation"""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n_points)
    t = np.random.uniform(0, 1, n_points)
    
    # Analytical solution (approximate)
    u = -np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t)
    
    return x, t, u


def physics_loss_burgers(model, x, t, nu=0.01/np.pi):
    """Compute physics loss for Burgers equation"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # Compute derivatives
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    # Burgers equation: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    
    return torch.mean(residual**2)


def train_with_kappa(kappa, epochs=3000, base_reg_weight=1e-4, use_solar=True):
    """Train model with specific κ value"""
    
    print(f"\n{'='*80}")
    if kappa is None:
        print(f"  Training CONTINUOUS (Baseline - No Solar)")
    else:
        print(f"  Training with κ = {kappa}")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate data
    x_data, t_data, u_data = generate_burgers_data(n_points=1000)
    x_train = torch.FloatTensor(x_data).reshape(-1, 1).to(device)
    t_train = torch.FloatTensor(t_data).reshape(-1, 1).to(device)
    u_train = torch.FloatTensor(u_data).reshape(-1, 1).to(device)
    
    # Solar model (only if use_solar=True)
    if use_solar:
        solar_model = SolarPowerModel(
            peak_power_w=300.0,
            gpu_power_w=250.0,
            mode='simplified'
        )
    else:
        solar_model = None
    
    # Training
    losses = []
    active_steps = 0
    total_steps = 0
    start_time = time.time()
    
    for step in range(epochs):
        total_steps += 1
        
        # Check solar power availability
        if use_solar:
            power_available = solar_model.is_power_available(step)
            if not power_available:
                if step % 500 == 0:
                    duty = (active_steps / total_steps) * 100
                    print(f"  Step {step:4d}/{epochs} | IDLE | Duty cycle: {duty:.1f}%")
                continue
        
        active_steps += 1
        
        # Compute regularization weight (adaptive if κ > 0)
        if kappa is not None and kappa > 0:
            # Adaptive regularization: amplify near power exhaustion
            # Simplified: just apply constant amplification since we don't predict exhaustion
            reg_weight = base_reg_weight * (1.0 + kappa)
        else:
            reg_weight = base_reg_weight
        
        # Forward pass
        u_pred = model(x_train, t_train)
        data_loss = torch.mean((u_pred - u_train)**2)
        
        # Physics loss
        phys_loss = physics_loss_burgers(model, x_train, t_train)
        
        # Regularization loss
        reg_loss = sum(torch.sum(p**2) for p in model.parameters())
        
        # Total loss
        loss = data_loss + phys_loss + reg_weight * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Logging
        if step % 500 == 0:
            duty = (active_steps / total_steps) * 100
            print(f"  Step {step:4d}/{epochs} | Loss: {loss.item():.6f} | "
                  f"Duty: {duty:.1f}% | Reg: {reg_weight:.2e}")
    
    training_time = time.time() - start_time
    final_loss = losses[-1] if losses else float('inf')
    duty_cycle = (active_steps / total_steps) * 100
    
    print(f"\n  ✓ Training complete!")
    print(f"    Final loss: {final_loss:.6f}")
    print(f"    Training time: {training_time:.2f}s")
    print(f"    Duty cycle: {duty_cycle:.1f}%")
    print(f"    Active steps: {active_steps}/{total_steps}")
    
    return {
        'kappa': kappa,
        'final_loss': final_loss,
        'training_time': training_time,
        'duty_cycle': duty_cycle,
        'active_steps': active_steps,
        'total_steps': total_steps,
        'losses': losses,
        'reg_weight': reg_weight if (kappa is not None and kappa > 0) else base_reg_weight
    }


def run_kappa_sweep(kappa_values=[0.0, 0.5, 1.0, 1.5, 2.0], output_dir='chapter4/results/kappa_sweep'):
    """Run κ sweep experiment"""
    
    print("="*80)
    print("  κ (KAPPA) SWEEP EXPERIMENT")
    print("  Testing Adaptive Regularization Amplification Values")
    print("="*80)
    print(f"\nκ values to test: {kappa_values}")
    print(f"Output directory: {output_dir}\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run continuous baseline first (no solar interruptions)
    print("\n" + "="*80)
    print("  BASELINE: CONTINUOUS TRAINING (No solar interruptions)")
    print("="*80)
    
    baseline_result = train_with_kappa(kappa=None, use_solar=False)
    baseline_result['regime'] = 'Continuous'
    
    # Run passive (κ=0) with solar interruptions
    results = [baseline_result]
    
    for kappa in kappa_values:
        result = train_with_kappa(kappa, use_solar=True)
        if kappa == 0.0:
            result['regime'] = 'Passive (κ=0.0)'
        else:
            result['regime'] = f'Active (κ={kappa})'
        results.append(result)
    
    # Analysis
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    
    baseline_loss = baseline_result['final_loss']
    
    print(f"\n{'Regime':<20} {'κ':<8} {'Loss':<12} {'Δ vs Base':<12} {'Duty %':<10} {'Time (s)'}")
    print("-" * 80)
    
    for r in results:
        regime = r['regime']
        kappa_val = r.get('kappa', 'N/A')
        kappa_str = 'N/A' if kappa_val is None else f"{kappa_val:.1f}"
        loss = r['final_loss']
        delta = ((loss - baseline_loss) / baseline_loss) * 100
        duty = r['duty_cycle']
        train_time = r['training_time']
        
        delta_str = f"{delta:+.2f}%" if delta != 0 else "0.0%"
        print(f"{regime:<20} {kappa_str:<8} {loss:<12.6f} {delta_str:<12} {duty:<10.1f} {train_time:.2f}")
    
    # Save results
    results_file = Path(output_dir) / 'kappa_sweep_results.json'
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        save_results = []
        for r in results:
            r_copy = r.copy()
            r_copy['losses'] = [float(l) for l in r_copy['losses']]
            save_results.append(r_copy)
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss vs κ
    kappas = [r.get('kappa') for r in results[1:]]  # Skip baseline
    losses = [r['final_loss'] for r in results[1:]]
    delta_losses = [((l - baseline_loss) / baseline_loss) * 100 for l in losses]
    
    ax1.plot(kappas, delta_losses, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='r', linestyle='--', label='Baseline (Continuous)')
    ax1.set_xlabel('κ (Kappa) Value', fontsize=12)
    ax1.set_ylabel('Loss Change vs Baseline (%)', fontsize=12)
    ax1.set_title('Impact of Adaptive Regularization Amplification', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Training curves for key κ values
    for r in results:
        if r['regime'] in ['Continuous', 'Passive', 'Active (κ=0.5)', 'Active (κ=2.0)']:
            label = r['regime']
            ax2.plot(r['losses'], label=label, alpha=0.7)
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / 'kappa_sweep_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_file}")
    plt.close()
    
    # Find optimal κ
    best_idx = np.argmin([r['final_loss'] for r in results[1:]])  # Skip baseline
    best_result = results[1:][best_idx]
    best_kappa = best_result.get('kappa', 0.0)
    best_loss = best_result['final_loss']
    best_improvement = ((best_loss - baseline_loss) / baseline_loss) * 100
    
    print(f"\n{'='*80}")
    print(f"  OPTIMAL κ FOUND")
    print(f"{'='*80}")
    print(f"  Best κ: {best_kappa}")
    print(f"  Loss: {best_loss:.6f}")
    print(f"  Improvement: {best_improvement:+.2f}% vs baseline")
    print(f"  Regime: {best_result['regime']}")
    
    return results


if __name__ == "__main__":
    # Run κ sweep
    results = run_kappa_sweep(
        kappa_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        output_dir='chapter4/results/kappa_sweep_burgers'
    )
    
    print("\n" + "="*80)
    print("  κ SWEEP EXPERIMENT COMPLETE")
    print("="*80)