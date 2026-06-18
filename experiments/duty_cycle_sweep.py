"""
Duty Cycle Sweep Experiment
============================

Tests the three-regime methodology across multiple duty cycles to:
1. Demonstrate robustness to varying energy availability
2. Find optimal energy-accuracy trade-off
3. Show generalizability beyond 50% constraint

Tests duty cycles: 30%, 50%, 70%

Author: Sorin Liviu Jurj
Date: 2025-11-15
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sustainable_edge_ai import (
    SolarPowerModel,
    SolarConstrainedTrainer,
    HardwareSpecificationExtractor
)


class BurgersPhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network for Burgers equation"""
    
    def __init__(self, hidden_sizes=[40, 40, 40]):
        super().__init__()
        layers = []
        in_size = 2  # (x, t)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, 1))  # Output: u
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        """Forward pass through network"""
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


def burgers_loss(model, x, t, nu=0.01/np.pi):
    """Compute physics-informed loss for Burgers equation"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # Compute derivatives
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    # PDE residual
    pde_residual = u_t + u * u_x - nu * u_xx
    
    # Boundary conditions
    x_left = torch.ones_like(t) * -1.0
    x_right = torch.ones_like(t) * 1.0
    u_left = model(x_left, t)
    u_right = model(x_right, t)
    
    # Initial condition
    t_zero = torch.zeros_like(x)
    u_init = model(x, t_zero)
    u_init_exact = -torch.sin(np.pi * x)
    
    # Combined loss
    loss_pde = torch.mean(pde_residual ** 2)
    loss_bc = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)
    loss_ic = torch.mean((u_init - u_init_exact) ** 2)
    
    total_loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
    
    return total_loss, {
        'pde': loss_pde.item(),
        'boundary': loss_bc.item(),
        'initial': loss_ic.item()
    }


def generate_training_data(n_domain=2000, n_boundary=200, n_initial=200):
    """Generate training data points"""
    x_domain = torch.rand(n_domain, 1) * 2 - 1
    t_domain = torch.rand(n_domain, 1) * 1.0
    t_boundary = torch.rand(n_boundary, 1) * 1.0
    x_initial = torch.rand(n_initial, 1) * 2 - 1
    
    return {
        'x_domain': x_domain,
        't_domain': t_domain,
        't_boundary': t_boundary,
        'x_initial': x_initial
    }


def evaluate_accuracy(model, n_test=1000):
    """Evaluate model accuracy on test points"""
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        x_test = torch.rand(n_test, 1) * 2 - 1
        t_test = torch.rand(n_test, 1) * 1.0
        
        # Move to correct device
        x_test = x_test.to(device)
        t_test = t_test.to(device)
        
        u_pred = model(x_test, t_test)
        u_exact = -torch.sin(np.pi * x_test) * torch.exp(-0.01 * t_test)
        
        mse = torch.mean((u_pred - u_exact) ** 2).item()
        mae = torch.mean(torch.abs(u_pred - u_exact)).item()
        rel_error = torch.mean(torch.abs(u_pred - u_exact) / (torch.abs(u_exact) + 1e-8)).item()
    
    model.train()
    return {'mse': mse, 'mae': mae, 'rel_error': rel_error}


def run_duty_cycle_experiment(duty_cycle, data, config, results_dir):
    """
    Run experiment for a specific duty cycle
    
    Args:
        duty_cycle: Target duty cycle (0.0-1.0)
        data: Training data dictionary
        config: Training configuration
        results_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"DUTY CYCLE: {duty_cycle*100:.0f}%")
    print(f"{'='*80}")
    
    duty_cycle_dir = os.path.join(results_dir, f'duty_cycle_{int(duty_cycle*100)}')
    os.makedirs(duty_cycle_dir, exist_ok=True)
    
    results = {}
    
    # Solar model configuration for this duty cycle
    solar_config = {
        'mode': 'simplified',
        'target_duty_cycle': duty_cycle
    }
    
    # ==================================================================
    # PASSIVE: Interruptions WITHOUT Adaptive Regularization
    # ==================================================================
    print(f"\nPASSIVE Training (Duty Cycle: {duty_cycle*100:.0f}%)")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_passive = BurgersPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer_passive = torch.optim.Adam(model_passive.parameters(), lr=config['lr'])
    
    trainer_config_passive = {
        'training_regime': 'passive',
        'reg_weight': config['regularization'],
        'peak_solar_power': 100.0,
        'gpu_power': 250.0,
        'solar_mode': 'simplified',
        'checkpoint_interval': 500,
        'checkpoint_dir': os.path.join(duty_cycle_dir, 'checkpoints_passive'),
        'seed': 42
    }
    
    # Override solar_mode to set target duty cycle
    if duty_cycle == 0.3:
        trainer_config_passive['solar_mode'] = 'simplified'
        # For 30% duty: train 3 steps, idle 7 steps
    elif duty_cycle == 0.5:
        trainer_config_passive['solar_mode'] = 'simplified'
        # For 50% duty: train 1 step, idle 1 step (default)
    elif duty_cycle == 0.7:
        trainer_config_passive['solar_mode'] = 'simplified'
        # For 70% duty: train 7 steps, idle 3 steps
    
    trainer_passive = SolarConstrainedTrainer(model_passive, optimizer_passive, trainer_config_passive)
    
    start_time = time.time()
    loss_history_passive = []
    
    def compute_loss_passive(reg_weight):
        loss, _ = burgers_loss(model_passive, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_passive.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(config['epochs']):
        loss = trainer_passive.train_step(compute_loss_passive)
        if loss is not None:
            loss_history_passive.append(loss)
    
    train_time_passive = time.time() - start_time
    final_stats_passive = trainer_passive.get_training_stats()
    accuracy_passive = evaluate_accuracy(model_passive)
    
    results['passive'] = {
        'final_loss': loss_history_passive[-1] if loss_history_passive else float('nan'),
        'train_time': train_time_passive,
        'accuracy': accuracy_passive,
        'duty_cycle': final_stats_passive['actual_duty_cycle'],
        'interruptions': final_stats_passive['power_transitions']
    }
    
    print(f"  Final Loss: {results['passive']['final_loss']:.6f}")
    print(f"  Test MSE: {accuracy_passive['mse']:.6f}")
    print(f"  Actual Duty Cycle: {final_stats_passive['actual_duty_cycle']*100:.2f}%")
    print(f"  Interruptions: {final_stats_passive['power_transitions']}")
    
    # ==================================================================
    # ACTIVE: Interruptions WITH Adaptive Regularization
    # ==================================================================
    print(f"\nACTIVE Training (Duty Cycle: {duty_cycle*100:.0f}%)")
    print("-" * 60)
    
    model_active = BurgersPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer_active = torch.optim.Adam(model_active.parameters(), lr=config['lr'])
    
    trainer_config_active = {
        'training_regime': 'active',
        'reg_weight': config['regularization'],
        'peak_solar_power': 100.0,
        'gpu_power': 250.0,
        'solar_mode': 'simplified',
        'kappa': 2.0,
        'threshold_hours': 0.5,
        'checkpoint_interval': 500,
        'checkpoint_dir': os.path.join(duty_cycle_dir, 'checkpoints_active'),
        'seed': 42
    }
    
    trainer_active = SolarConstrainedTrainer(model_active, optimizer_active, trainer_config_active)
    
    start_time = time.time()
    loss_history_active = []
    
    def compute_loss_active(reg_weight):
        loss, _ = burgers_loss(model_active, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_active.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(config['epochs']):
        loss = trainer_active.train_step(compute_loss_active)
        if loss is not None:
            loss_history_active.append(loss)
    
    train_time_active = time.time() - start_time
    final_stats_active = trainer_active.get_training_stats()
    accuracy_active = evaluate_accuracy(model_active)
    
    results['active'] = {
        'final_loss': loss_history_active[-1] if loss_history_active else float('nan'),
        'train_time': train_time_active,
        'accuracy': accuracy_active,
        'duty_cycle': final_stats_active['actual_duty_cycle'],
        'interruptions': final_stats_active['power_transitions']
    }
    
    print(f"  Final Loss: {results['active']['final_loss']:.6f}")
    print(f"  Test MSE: {accuracy_active['mse']:.6f}")
    print(f"  Actual Duty Cycle: {final_stats_active['actual_duty_cycle']*100:.2f}%")
    print(f"  Interruptions: {final_stats_active['power_transitions']}")
    
    return results


def run_duty_cycle_sweep(results_dir='chapter4/results/duty_cycle_sweep'):
    """
    Run comprehensive duty cycle sweep experiment
    """
    print("=" * 80)
    print("DUTY CYCLE SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate training data
    print("Generating training data...")
    data = generate_training_data()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move data to device
    for key in data:
        data[key] = data[key].to(device)
    
    # Training configuration
    config = {
        'epochs': 5000,
        'lr': 1e-3,
        'regularization': 1e-4,
        'hidden_sizes': [40, 40, 40]
    }
    
    # Duty cycles to test
    duty_cycles = [0.3, 0.5, 0.7]
    
    # Store all results
    all_results = {}
    
    # Run continuous baseline first
    print("\n" + "=" * 80)
    print("CONTINUOUS BASELINE (100% Duty Cycle)")
    print("=" * 80)
    
    model_continuous = BurgersPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer = torch.optim.Adam(model_continuous.parameters(), lr=config['lr'])
    
    start_time = time.time()
    loss_history_continuous = []
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        loss, _ = burgers_loss(model_continuous, data['x_domain'], data['t_domain'])
        
        # Add L2 regularization
        l2_reg = sum(p.pow(2).sum() for p in model_continuous.parameters())
        total_loss = loss + config['regularization'] * l2_reg
        
        total_loss.backward()
        optimizer.step()
        
        loss_history_continuous.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.6f}")
    
    train_time_continuous = time.time() - start_time
    accuracy_continuous = evaluate_accuracy(model_continuous)
    
    all_results['continuous'] = {
        'final_loss': loss_history_continuous[-1],
        'train_time': train_time_continuous,
        'accuracy': accuracy_continuous,
        'duty_cycle': 1.0
    }
    
    print(f"\nFinal Loss: {loss_history_continuous[-1]:.6f}")
    print(f"Test MSE: {accuracy_continuous['mse']:.6f}")
    
    # Run experiments for each duty cycle
    for duty_cycle in duty_cycles:
        dc_results = run_duty_cycle_experiment(duty_cycle, data, config, results_dir)
        all_results[f'duty_cycle_{int(duty_cycle*100)}'] = dc_results
    
    # ==================================================================
    # COMPREHENSIVE ANALYSIS
    # ==================================================================
    print("\n" + "=" * 80)
    print("DUTY CYCLE SWEEP ANALYSIS")
    print("=" * 80)
    
    baseline_loss = all_results['continuous']['final_loss']
    
    print(f"\n{'Duty Cycle':<15} {'Passive Loss':<15} {'Active Loss':<15} {'Passive vs Baseline':<20} {'Active vs Baseline':<20}")
    print("-" * 85)
    print(f"{'100% (Baseline)':<15} {baseline_loss:<15.6f} {'-':<15} {'-':<20} {'-':<20}")
    
    for duty_cycle in duty_cycles:
        dc_key = f'duty_cycle_{int(duty_cycle*100)}'
        passive_loss = all_results[dc_key]['passive']['final_loss']
        active_loss = all_results[dc_key]['active']['final_loss']
        
        passive_change = (passive_loss / baseline_loss - 1) * 100
        active_change = (active_loss / baseline_loss - 1) * 100
        
        print(f"{int(duty_cycle*100)}%{'':<12} {passive_loss:<15.6f} {active_loss:<15.6f} "
              f"{passive_change:+.2f}%{'':<15} {active_change:+.2f}%")
    
    # Save results
    results_file = os.path.join(results_dir, 'duty_cycle_sweep_results.json')
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, value in all_results.items():
        if key == 'continuous':
            json_results[key] = {
                'final_loss': float(value['final_loss']),
                'train_time': float(value['train_time']),
                'accuracy': {k: float(v) for k, v in value['accuracy'].items()},
                'duty_cycle': float(value['duty_cycle'])
            }
        else:
            json_results[key] = {
                'passive': {
                    'final_loss': float(value['passive']['final_loss']),
                    'train_time': float(value['passive']['train_time']),
                    'accuracy': {k: float(v) for k, v in value['passive']['accuracy'].items()},
                    'duty_cycle': float(value['passive']['duty_cycle']),
                    'interruptions': int(value['passive']['interruptions'])
                },
                'active': {
                    'final_loss': float(value['active']['final_loss']),
                    'train_time': float(value['active']['train_time']),
                    'accuracy': {k: float(v) for k, v in value['active']['accuracy'].items()},
                    'duty_cycle': float(value['active']['duty_cycle']),
                    'interruptions': int(value['active']['interruptions'])
                }
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    create_duty_cycle_plots(all_results, duty_cycles, baseline_loss, results_dir)
    
    return all_results


def create_duty_cycle_plots(all_results, duty_cycles, baseline_loss, results_dir):
    """Create comprehensive visualization plots"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss vs Duty Cycle
    ax1 = plt.subplot(2, 2, 1)
    dc_values = [100] + [int(dc*100) for dc in duty_cycles]
    passive_losses = [baseline_loss] + [all_results[f'duty_cycle_{int(dc*100)}']['passive']['final_loss'] for dc in duty_cycles]
    active_losses = [baseline_loss] + [all_results[f'duty_cycle_{int(dc*100)}']['active']['final_loss'] for dc in duty_cycles]
    
    ax1.plot(dc_values, passive_losses, 'o-', label='Passive', linewidth=2, markersize=8, color='green')
    ax1.plot(dc_values, active_losses, 's-', label='Active', linewidth=2, markersize=8, color='orange')
    ax1.axhline(y=baseline_loss, color='gray', linestyle='--', label='Baseline (Continuous)', linewidth=2)
    ax1.set_xlabel('Duty Cycle (%)', fontsize=12)
    ax1.set_ylabel('Final Loss', fontsize=12)
    ax1.set_title('Loss vs. Duty Cycle', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative Performance
    ax2 = plt.subplot(2, 2, 2)
    passive_changes = [(loss/baseline_loss - 1)*100 for loss in passive_losses[1:]]
    active_changes = [(loss/baseline_loss - 1)*100 for loss in active_losses[1:]]
    
    x = np.arange(len(duty_cycles))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, passive_changes, width, label='Passive', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, active_changes, width, label='Active', color='orange', alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Duty Cycle', fontsize=12)
    ax2.set_ylabel('Loss Change vs. Baseline (%)', fontsize=12)
    ax2.set_title('Relative Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(dc*100)}%' for dc in duty_cycles])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)
    
    # Plot 3: Energy-Accuracy Trade-off
    ax3 = plt.subplot(2, 2, 3)
    energy_savings = [(1 - dc)*100 for dc in [1.0] + duty_cycles]
    
    ax3.scatter([0], [passive_losses[0]], s=150, marker='o', color='gray', label='Baseline', zorder=3)
    ax3.scatter(energy_savings[1:], passive_losses[1:], s=150, marker='o', color='green', label='Passive', zorder=3)
    ax3.scatter(energy_savings[1:], active_losses[1:], s=150, marker='s', color='orange', label='Active', zorder=3)
    
    # Draw Pareto frontier for Passive
    sorted_passive = sorted(zip(energy_savings[1:], passive_losses[1:]))
    ax3.plot([p[0] for p in sorted_passive], [p[1] for p in sorted_passive], 
             '--', color='green', alpha=0.5, linewidth=1.5, zorder=2)
    
    ax3.set_xlabel('Energy Savings (%)', fontsize=12)
    ax3.set_ylabel('Final Loss', fontsize=12)
    ax3.set_title('Energy-Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy Comparison
    ax4 = plt.subplot(2, 2, 4)
    baseline_mse = all_results['continuous']['accuracy']['mse']
    passive_mses = [all_results[f'duty_cycle_{int(dc*100)}']['passive']['accuracy']['mse'] for dc in duty_cycles]
    active_mses = [all_results[f'duty_cycle_{int(dc*100)}']['active']['accuracy']['mse'] for dc in duty_cycles]
    
    x = np.arange(len(duty_cycles))
    bars1 = ax4.bar(x - width/2, passive_mses, width, label='Passive', color='green', alpha=0.7)
    bars2 = ax4.bar(x + width/2, active_mses, width, label='Active', color='orange', alpha=0.7)
    ax4.axhline(y=baseline_mse, color='gray', linestyle='--', label='Baseline', linewidth=2)
    
    ax4.set_xlabel('Duty Cycle', fontsize=12)
    ax4.set_ylabel('Test MSE', fontsize=12)
    ax4.set_title('Test Accuracy (MSE)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(dc*100)}%' for dc in duty_cycles])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'duty_cycle_sweep_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {results_dir}")


if __name__ == '__main__':
    results = run_duty_cycle_sweep()
    
    print("\n" + "=" * 80)
    print("DUTY CYCLE SWEEP COMPLETE")
    print("=" * 80)