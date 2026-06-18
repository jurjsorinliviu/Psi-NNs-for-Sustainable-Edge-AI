"""
Three-Regime Advection Equation Experiment
===========================================

Advection (transport) equation: ∂u/∂t + c∂u/∂x = 0

Tests the three-regime methodology on a first-order hyperbolic PDE
representing pure convection/transport phenomena.

Author: Sorin Liviu Jurj
Date: 2025-11-16
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

from sustainable_edge_ai import SolarConstrainedTrainer


class AdvectionPhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for 1D Advection equation:
    ∂u/∂t + c∂u/∂x = 0
    
    where c = 1.0 is advection speed
    """
    
    def __init__(self, hidden_sizes=[40, 40, 40]):
        super().__init__()
        layers = []
        in_size = 2  # (x, t)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


def advection_loss(model, x, t, c=1.0):
    """
    Physics-informed loss for advection equation
    
    Loss = MSE(PDE) + MSE(BC) + MSE(IC)
    """
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # Compute derivatives
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    # PDE residual: ∂u/∂t + c∂u/∂x = 0
    pde_residual = u_t + c * u_x
    
    # Periodic boundary condition: u(0,t) = u(1,t)
    x_left = torch.zeros_like(t)
    x_right = torch.ones_like(t)
    u_left = model(x_left, t)
    u_right = model(x_right, t)
    
    # Initial condition: u(x,0) = sin(2πx) (traveling wave)
    t_zero = torch.zeros_like(x)
    u_init = model(x, t_zero)
    u_init_exact = torch.sin(2 * np.pi * x)
    
    # Combined loss
    loss_pde = torch.mean(pde_residual ** 2)
    loss_bc = torch.mean((u_left - u_right) ** 2)  # Periodic BC
    loss_ic = torch.mean((u_init - u_init_exact) ** 2)
    
    total_loss = loss_pde + 10.0 * loss_bc + 10.0 * loss_ic
    
    return total_loss, {
        'pde': loss_pde.item(),
        'boundary': loss_bc.item(),
        'initial': loss_ic.item()
    }


def generate_training_data(n_domain=2000, n_boundary=200, n_initial=200, seed=42):
    """Generate training data"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x_domain = torch.rand(n_domain, 1)
    t_domain = torch.rand(n_domain, 1) * 1.0
    
    t_boundary = torch.rand(n_boundary, 1) * 1.0
    x_initial = torch.rand(n_initial, 1)
    
    return {
        'x_domain': x_domain,
        't_domain': t_domain,
        't_boundary': t_boundary,
        'x_initial': x_initial
    }


def evaluate_accuracy(model, n_test=1000):
    """Evaluate model accuracy"""
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        x_test = torch.rand(n_test, 1).to(device)
        t_test = torch.rand(n_test, 1).to(device) * 1.0
        
        u_pred = model(x_test, t_test)
        # Analytical solution: u(x,t) = sin(2π(x - ct)) for periodic BC
        u_exact = torch.sin(2 * np.pi * (x_test - t_test))
        
        mse = torch.mean((u_pred - u_exact) ** 2).item()
        mae = torch.mean(torch.abs(u_pred - u_exact)).item()
    
    model.train()
    return {'mse': mse, 'mae': mae}


def run_three_regime_advection(epochs=3000, save_dir='chapter4/results/three_regime_advection', seed=42):
    """Run three-regime training on Advection equation"""
    
    print("="*80)
    print("THREE-REGIME ADVECTION EQUATION EXPERIMENT")
    print("="*80)
    
    os.makedirs(save_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate training data
    print("Generating training data...")
    data = generate_training_data(seed=seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for key in data:
        data[key] = data[key].to(device)
    
    config = {
        'epochs': epochs,
        'lr': 5e-3,  # Slightly higher than default for faster convergence
        'regularization': 1e-4,
        'hidden_sizes': [40, 40, 40]
    }
    
    results = {}
    
    # REGIME 1: Continuous
    print("\n" + "="*80)
    print("REGIME 1: CONTINUOUS")
    print("="*80)
    
    model_continuous = AdvectionPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer = torch.optim.Adam(model_continuous.parameters(), lr=config['lr'])
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        loss, _ = advection_loss(model_continuous, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_continuous.parameters())
        total_loss = loss + config['regularization'] * l2_reg
        total_loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item():.6f}")
    
    train_time = time.time() - start_time
    accuracy = evaluate_accuracy(model_continuous)
    
    results['continuous'] = {
        'final_loss': loss_history[-1],
        'train_time': train_time,
        'accuracy': accuracy,
        'loss_history': loss_history
    }
    
    print(f"\nFinal Loss: {loss_history[-1]:.6f}")
    print(f"Test MSE: {accuracy['mse']:.6f}")
    
    # REGIME 2: Passive
    print("\n" + "="*80)
    print("REGIME 2: PASSIVE")
    print("="*80)
    
    model_passive = AdvectionPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer_passive = torch.optim.Adam(model_passive.parameters(), lr=config['lr'])
    
    trainer_config = {
        'training_regime': 'passive',
        'reg_weight': config['regularization'],
        'seed': seed
    }
    
    trainer = SolarConstrainedTrainer(model_passive, optimizer_passive, trainer_config)
    
    loss_history_passive = []
    start_time = time.time()
    
    def compute_loss(reg_weight):
        loss, _ = advection_loss(model_passive, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_passive.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(config['epochs']):
        loss = trainer.train_step(compute_loss)
        if loss is not None:
            loss_history_passive.append(loss)
            if (epoch + 1) % 500 == 0:
                stats = trainer.get_training_stats()
                print(f"Epoch {epoch+1} | Loss: {loss:.6f} | Duty: {stats['actual_duty_cycle']:.2%}")
    
    train_time_passive = time.time() - start_time
    stats_passive = trainer.get_training_stats()
    accuracy_passive = evaluate_accuracy(model_passive)
    
    results['passive'] = {
        'final_loss': loss_history_passive[-1] if loss_history_passive else float('nan'),
        'train_time': train_time_passive,
        'accuracy': accuracy_passive,
        'loss_history': loss_history_passive,
        'duty_cycle': stats_passive['actual_duty_cycle']
    }
    
    # REGIME 3: Active
    print("\n" + "="*80)
    print("REGIME 3: ACTIVE")
    print("="*80)
    
    model_active = AdvectionPhysicsInformedNN(config['hidden_sizes']).to(device)
    optimizer_active = torch.optim.Adam(model_active.parameters(), lr=config['lr'])
    
    trainer_config_active = {
        'training_regime': 'active',
        'reg_weight': config['regularization'],
        'kappa': 2.0,
        'seed': seed
    }
    
    trainer_active = SolarConstrainedTrainer(model_active, optimizer_active, trainer_config_active)
    
    loss_history_active = []
    start_time = time.time()
    
    def compute_loss_active(reg_weight):
        loss, _ = advection_loss(model_active, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_active.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(config['epochs']):
        loss = trainer_active.train_step(compute_loss_active)
        if loss is not None:
            loss_history_active.append(loss)
            if (epoch + 1) % 500 == 0:
                stats = trainer_active.get_training_stats()
                print(f"Epoch {epoch+1} | Loss: {loss:.6f} | Duty: {stats['actual_duty_cycle']:.2%}")
    
    train_time_active = time.time() - start_time
    stats_active = trainer_active.get_training_stats()
    accuracy_active = evaluate_accuracy(model_active)
    
    results['active'] = {
        'final_loss': loss_history_active[-1] if loss_history_active else float('nan'),
        'train_time': train_time_active,
        'accuracy': accuracy_active,
        'loss_history': loss_history_active,
        'duty_cycle': stats_active['actual_duty_cycle']
    }
    
    # Save and analyze
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    baseline = results['continuous']['final_loss']
    print(f"\nContinuous: {baseline:.6f}")
    print(f"Passive:    {results['passive']['final_loss']:.6f} ({((results['passive']['final_loss']/baseline-1)*100):+.2f}%)")
    print(f"Active:     {results['active']['final_loss']:.6f} ({((results['active']['final_loss']/baseline-1)*100):+.2f}%)")
    
    # Save results
    with open(f'{save_dir}/results.json', 'w') as f:
        json_results = {}
        for regime, data in results.items():
            json_results[regime] = {
                'final_loss': float(data['final_loss']),
                'train_time': float(data['train_time']),
                'accuracy': {k: float(v) for k, v in data['accuracy'].items()},
                'duty_cycle': float(data.get('duty_cycle', 1.0))
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}/results.json")
    
    return results


if __name__ == '__main__':
    results = run_three_regime_advection()
    print("\n" + "="*80)
    print("ADVECTION EQUATION EXPERIMENT COMPLETE")
    print("="*80)