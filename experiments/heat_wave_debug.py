"""
Heat/Wave Hyperparameter Debug
===============================

Systematically test different hyperparameters to fix catastrophic failures.

Strategy:
1. Test increased epochs (6000, 10000)
2. Test increased learning rates (5e-3, 1e-2)
3. Test deeper architectures ([60,60,60,60])
4. Find configurations that work with 50% duty cycle

Author: Sorin Liviu Jurj
Date: 2025-11-16
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from sustainable_edge_ai import SolarConstrainedTrainer
from three_regime_heat_experiment import HeatPhysicsInformedNN, heat_loss, generate_training_data as gen_heat_data, evaluate_accuracy as eval_heat
from three_regime_wave_experiment import WavePhysicsInformedNN, wave_loss, generate_training_data as gen_wave_data, evaluate_accuracy as eval_wave


def test_heat_config(epochs, lr, hidden_sizes, seed=42):
    """Test a specific Heat configuration"""
    print(f"\nTesting Heat: epochs={epochs}, lr={lr}, arch={hidden_sizes}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = gen_heat_data(seed=seed)
    for key in data:
        data[key] = data[key].to(device)
    
    # Continuous baseline
    model_continuous = HeatPhysicsInformedNN(hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model_continuous.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = heat_loss(model_continuous, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_continuous.parameters())
        total_loss = loss + 1e-4 * l2_reg
        total_loss.backward()
        optimizer.step()
    
    continuous_loss = loss.item()
    
    # Passive regime
    model_passive = HeatPhysicsInformedNN(hidden_sizes).to(device)
    optimizer_passive = torch.optim.Adam(model_passive.parameters(), lr=lr)
    
    trainer_config = {
        'training_regime': 'passive',
        'reg_weight': 1e-4,
        'seed': seed
    }
    
    trainer = SolarConstrainedTrainer(model_passive, optimizer_passive, trainer_config)
    
    loss_history_passive = []
    
    def compute_loss(reg_weight):
        loss, _ = heat_loss(model_passive, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_passive.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(epochs):
        loss = trainer.train_step(compute_loss)
        if loss is not None:
            loss_history_passive.append(loss)
    
    passive_loss = loss_history_passive[-1] if loss_history_passive else float('nan')
    degradation = ((passive_loss / continuous_loss) - 1) * 100 if continuous_loss > 0 else float('inf')
    
    print(f"  Continuous: {continuous_loss:.6f}")
    print(f"  Passive:    {passive_loss:.6f} ({degradation:+.2f}%)")
    
    return {
        'continuous': continuous_loss,
        'passive': passive_loss,
        'degradation': degradation,
        'success': abs(degradation) < 50  # Success if <50% degradation
    }


def test_wave_config(epochs, lr, hidden_sizes, seed=42):
    """Test a specific Wave configuration"""
    print(f"\nTesting Wave: epochs={epochs}, lr={lr}, arch={hidden_sizes}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = gen_wave_data(seed=seed)
    for key in data:
        data[key] = data[key].to(device)
    
    # Continuous baseline
    model_continuous = WavePhysicsInformedNN(hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model_continuous.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = wave_loss(model_continuous, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_continuous.parameters())
        total_loss = loss + 1e-4 * l2_reg
        total_loss.backward()
        optimizer.step()
    
    continuous_loss = loss.item()
    
    # Passive regime
    model_passive = WavePhysicsInformedNN(hidden_sizes).to(device)
    optimizer_passive = torch.optim.Adam(model_passive.parameters(), lr=lr)
    
    trainer_config = {
        'training_regime': 'passive',
        'reg_weight': 1e-4,
        'seed': seed
    }
    
    trainer = SolarConstrainedTrainer(model_passive, optimizer_passive, trainer_config)
    
    loss_history_passive = []
    
    def compute_loss(reg_weight):
        loss, _ = wave_loss(model_passive, data['x_domain'], data['t_domain'])
        l2_reg = sum(p.pow(2).sum() for p in model_passive.parameters())
        return loss + reg_weight * l2_reg
    
    for epoch in range(epochs):
        loss = trainer.train_step(compute_loss)
        if loss is not None:
            loss_history_passive.append(loss)
    
    passive_loss = loss_history_passive[-1] if loss_history_passive else float('nan')
    degradation = ((passive_loss / continuous_loss) - 1) * 100 if continuous_loss > 0 else float('inf')
    
    print(f"  Continuous: {continuous_loss:.6f}")
    print(f"  Passive:    {passive_loss:.6f} ({degradation:+.2f}%)")
    
    return {
        'continuous': continuous_loss,
        'passive': passive_loss,
        'degradation': degradation,
        'success': abs(degradation) < 50
    }


def run_debug_grid_search():
    """Run systematic grid search for both Heat and Wave"""
    
    print("="*80)
    print("HEAT/WAVE HYPERPARAMETER DEBUG")
    print("="*80)
    print("\nGoal: Find configurations where Passive degrades <50% vs Continuous")
    print()
    
    # Configuration grid
    configs = [
        # Original (known to fail)
        {'epochs': 3000, 'lr': 1e-3, 'hidden': [40, 40, 40], 'name': 'Original (baseline)'},
        
        # More epochs
        {'epochs': 6000, 'lr': 1e-3, 'hidden': [40, 40, 40], 'name': '2x epochs'},
        {'epochs': 10000, 'lr': 1e-3, 'hidden': [40, 40, 40], 'name': '3.3x epochs'},
        
        # Higher learning rate
        {'epochs': 3000, 'lr': 5e-3, 'hidden': [40, 40, 40], 'name': '5x learning rate'},
        {'epochs': 3000, 'lr': 1e-2, 'hidden': [40, 40, 40], 'name': '10x learning rate'},
        
        # Deeper architecture
        {'epochs': 3000, 'lr': 1e-3, 'hidden': [60, 60, 60, 60], 'name': 'Deeper network'},
        
        # Combined: more epochs + higher LR
        {'epochs': 6000, 'lr': 5e-3, 'hidden': [40, 40, 40], 'name': '2x epochs + 5x LR'},
        
        # Combined: more epochs + deeper
        {'epochs': 6000, 'lr': 1e-3, 'hidden': [60, 60, 60, 60], 'name': '2x epochs + deeper'},
    ]
    
    results = {
        'heat': [],
        'wave': []
    }
    
    # Test Heat equation
    print("\n" + "="*80)
    print("HEAT EQUATION DEBUG")
    print("="*80)
    
    for cfg in configs:
        print(f"\n--- Configuration: {cfg['name']} ---")
        result = test_heat_config(cfg['epochs'], cfg['lr'], cfg['hidden'])
        result['config'] = cfg
        results['heat'].append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Test Wave equation
    print("\n" + "="*80)
    print("WAVE EQUATION DEBUG")
    print("="*80)
    
    for cfg in configs:
        print(f"\n--- Configuration: {cfg['name']} ---")
        result = test_wave_config(cfg['epochs'], cfg['lr'], cfg['hidden'])
        result['config'] = cfg
        results['wave'].append(result)
        time.sleep(1)
    
    # Analyze results
    print("\n" + "="*80)
    print("SUMMARY: SUCCESSFUL CONFIGURATIONS")
    print("="*80)
    
    print("\nHeat Equation:")
    heat_successes = [r for r in results['heat'] if r['success']]
    if heat_successes:
        print(f"Found {len(heat_successes)} successful configurations:")
        for r in heat_successes:
            cfg = r['config']
            print(f"  ✓ {cfg['name']}: {r['degradation']:+.2f}% degradation")
    else:
        print("  ✗ No successful configurations found")
        print("  Best result:")
        best = min(results['heat'], key=lambda x: abs(x['degradation']))
        print(f"    {best['config']['name']}: {best['degradation']:+.2f}% degradation")
    
    print("\nWave Equation:")
    wave_successes = [r for r in results['wave'] if r['success']]
    if wave_successes:
        print(f"Found {len(wave_successes)} successful configurations:")
        for r in wave_successes:
            cfg = r['config']
            print(f"  ✓ {cfg['name']}: {r['degradation']:+.2f}% degradation")
    else:
        print("  ✗ No successful configurations found")
        print("  Best result:")
        best = min(results['wave'], key=lambda x: abs(x['degradation']))
        print(f"    {best['config']['name']}: {best['degradation']:+.2f}% degradation")
    
    # Save results
    save_dir = 'chapter4/results/tier2_debug'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f'{save_dir}/heat_wave_debug_results.json', 'w') as f:
        # Convert to serializable format
        save_results = {
            'heat': [{
                'config': r['config'],
                'continuous': float(r['continuous']),
                'passive': float(r['passive']),
                'degradation': float(r['degradation']),
                'success': r['success']
            } for r in results['heat']],
            'wave': [{
                'config': r['config'],
                'continuous': float(r['continuous']),
                'passive': float(r['passive']),
                'degradation': float(r['degradation']),
                'success': r['success']
            } for r in results['wave']]
        }
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}/heat_wave_debug_results.json")
    
    return results


if __name__ == '__main__':
    results = run_debug_grid_search()
    print("\n" + "="*80)
    print("DEBUG COMPLETE!")
    print("="*80)