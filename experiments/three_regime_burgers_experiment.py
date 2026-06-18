"""
Three-Regime Burgers Equation Experiment

Implements Section III-C methodology comparing three training regimes:
1. Continuous Training (Baseline) - Grid power, standard regularization
2. Intermittent Training (Passive) - Solar interruptions WITHOUT adaptive regularization  
3. Intermittent Training (Active) - Solar interruptions WITH adaptive regularization

This validates whether solar constraints improve structure compression.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple

# Add paths
BASE_DIR = Path(__file__).parent.parent.parent
PSIHDL_DIR = BASE_DIR / "PSI-HDL-implementation"
sys.path.insert(0, str(PSIHDL_DIR / "Code"))
sys.path.insert(0, str(PSIHDL_DIR / "Psi-NN-main" / "Module"))
sys.path.insert(0, str(BASE_DIR / "chapter4"))

# Import framework components
import PsiNN_burgers
from structure_extractor import StructureExtractor
from sustainable_edge_ai import (
    SolarConstrainedTrainer,
    HardwareSpecificationExtractor,
    EdgeAIPlatformRecommender,
    CarbonFootprintAnalyzer
)


def generate_burgers_data(n_points: int = 1000, 
                          nu: float = 0.01/np.pi) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for Burgers equation
    
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    Args:
        n_points: Number of training points
        nu: Viscosity parameter
        
    Returns:
        X: Input tensor [t, x]
        u: Solution values
    """
    # Domain: t ∈ [0, 1], x ∈ [-1, 1]
    t = np.random.uniform(0, 1, n_points)
    x = np.random.uniform(-1, 1, n_points)
    
    # Analytical solution: u(x,t) = -2νπ·sin(πx)·exp(-ν·π²·t) / (1 + cos(πx)·exp(-ν·π²·t))
    u = -2 * nu * np.pi * np.sin(np.pi * x) * np.exp(-nu * np.pi**2 * t) / \
        (1 + np.cos(np.pi * x) * np.exp(-nu * np.pi**2 * t))
    
    X = torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32, requires_grad=True)
    u_tensor = torch.tensor(u.reshape(-1, 1), dtype=torch.float32)
    
    return X, u_tensor


def physics_loss(model: nn.Module, X: torch.Tensor, nu: float = 0.01/np.pi) -> torch.Tensor:
    """
    Compute Burgers equation physics loss
    
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    Args:
        model: Ψ-NN model
        X: Input tensor [t, x]
        nu: Viscosity
        
    Returns:
        Physics residual loss
    """
    u_pred = model(X)
    
    # Compute gradients
    grad_u = torch.autograd.grad(u_pred, X, 
                                  grad_outputs=torch.ones_like(u_pred),
                                  create_graph=True)[0]
    
    u_t = grad_u[:, 0:1]  # ∂u/∂t
    u_x = grad_u[:, 1:2]  # ∂u/∂x
    
    # Second derivative ∂²u/∂x²
    u_xx = torch.autograd.grad(u_x, X,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 1:2]
    
    # Burgers equation residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²
    residual = u_t + u_pred * u_x - nu * u_xx
    
    return torch.mean(residual ** 2)


def boundary_loss(model: nn.Module, n_boundary: int = 100) -> torch.Tensor:
    """
    Enforce boundary and initial conditions
    
    Args:
        model: Ψ-NN model
        n_boundary: Number of boundary points
        
    Returns:
        Boundary condition loss
    """
    device = next(model.parameters()).device
    
    # Initial condition: u(x, 0) = -sin(πx)
    x_ic = torch.linspace(-1, 1, n_boundary).reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    X_ic = torch.cat([t_ic, x_ic], dim=1).to(device)
    X_ic.requires_grad = True
    
    u_ic_pred = model(X_ic)
    u_ic_true = -torch.sin(np.pi * X_ic[:, 1:2])
    
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)
    
    # Boundary conditions: u(-1, t) = u(1, t) = 0
    t_bc = torch.linspace(0, 1, n_boundary).reshape(-1, 1)
    x_bc_left = -torch.ones_like(t_bc)
    x_bc_right = torch.ones_like(t_bc)
    
    X_bc_left = torch.cat([t_bc, x_bc_left], dim=1).to(device)
    X_bc_right = torch.cat([t_bc, x_bc_right], dim=1).to(device)
    X_bc_left.requires_grad = True
    X_bc_right.requires_grad = True
    
    u_bc_left = model(X_bc_left)
    u_bc_right = model(X_bc_right)
    
    loss_bc = torch.mean(u_bc_left ** 2) + torch.mean(u_bc_right ** 2)
    
    return loss_ic + loss_bc


def train_single_regime(regime: str, config: Dict, save_dir: Path) -> Dict:
    """
    Train Ψ-NN model under specified regime
    
    Args:
        regime: 'continuous', 'passive', or 'active'
        config: Training configuration
        save_dir: Directory to save results
        
    Returns:
        Results dictionary with metrics and statistics
    """
    print(f"\n{'='*80}")
    print(f"Training Regime: {regime.upper()}")
    print(f"{'='*80}\n")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PsiNN_burgers.Net(node_num=config['node_num']).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create solar-constrained trainer
    trainer_config = {
        'training_regime': regime,
        'reg_weight': config['reg_weight'],
        'peak_solar_power': config.get('peak_solar_power', 100.0),
        'gpu_power': config.get('gpu_power', 250.0),
        'kappa': config.get('kappa', 2.0),
        'threshold_hours': config.get('threshold_hours', 0.5),
        'checkpoint_interval': config.get('checkpoint_interval', 500),
        'checkpoint_dir': str(save_dir / f'{regime}_checkpoints'),
        'seed': config.get('seed', 42)
    }
    
    trainer = SolarConstrainedTrainer(model, optimizer, trainer_config)
    
    # Generate training data
    X_train, u_train = generate_burgers_data(n_points=config['n_train_points'])
    X_train = X_train.to(device)
    u_train = u_train.to(device)
    
    # Training loop
    nu = 0.01 / np.pi
    epochs = config['epochs']
    loss_history = []
    
    def compute_loss(reg_weight):
        """Composite loss function"""
        # Data fitting loss
        u_pred = model(X_train)
        data_loss = torch.mean((u_pred - u_train) ** 2)
        
        # Physics loss
        phys_loss = physics_loss(model, X_train, nu=nu)
        
        # Boundary loss
        bound_loss = boundary_loss(model, n_boundary=50)
        
        # Regularization loss
        reg_loss = sum(torch.sum(p ** 2) for p in model.parameters())
        
        total_loss = (
            data_loss + 
            config['physics_weight'] * phys_loss + 
            config['boundary_weight'] * bound_loss +
            reg_weight * reg_loss
        )
        
        return total_loss
    
    print(f"Training for {epochs} epochs ({trainer.get_regime_description()})")
    print(f"Device: {device}")
    print(f"Model architecture: {config['node_num']} nodes per layer\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        loss = trainer.train_step(compute_loss)
        
        if loss is not None:
            loss_history.append(loss)
            
            if (epoch + 1) % 500 == 0:
                stats = trainer.get_training_stats()
                print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f} | "
                      f"Active steps: {stats['active_steps']:5d} | "
                      f"Duty cycle: {stats['actual_duty_cycle']:.2%} | "
                      f"Reg weight: {stats['current_reg_weight']:.2e}")
        else:
            # Idle step - no loss computed
            if (epoch + 1) % 500 == 0:
                stats = trainer.get_training_stats()
                print(f"Epoch {epoch+1:5d} | IDLE | "
                      f"Duty cycle: {stats['actual_duty_cycle']:.2%}")
    
    training_time = time.time() - start_time
    
    # Get final statistics
    final_stats = trainer.get_training_stats()
    final_loss = loss_history[-1] if loss_history else float('nan')
    
    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Active steps: {final_stats['active_steps']}/{final_stats['total_steps']}")
    print(f"Actual duty cycle: {final_stats['actual_duty_cycle']:.2%}")
    
    # Extract structure
    print("\nExtracting discovered structure...")
    structure_extractor = StructureExtractor(model, model_type="PsiNN_burgers")
    structure = structure_extractor.extract()
    
    print(f"Discovered structure: {len(structure.layers)} layers")
    total_clusters = sum(1 for layer in structure.layers if layer['type'] in ['psi_plus_minus', 'psi_symmetric', 'psi_combination', 'linear'])
    print(f"Total layers: {total_clusters}")
    
    # Compute hardware specifications
    print("\nComputing hardware specifications...")
    hw_extractor = HardwareSpecificationExtractor(model, structure_extractor)
    
    ops = hw_extractor.compute_operations()
    tops_req = hw_extractor.compute_tops_requirement(target_fps=30.0)
    memory = hw_extractor.compute_memory_requirements()
    power = hw_extractor.estimate_power_consumption()
    
    print(f"Operations per inference: {ops['total_operations']:,}")
    print(f"Required TOPS: {tops_req:.6f}")
    print(f"Memory requirement: {memory['total_memory_kb']:.2f} KB")
    print(f"Estimated power: {power['total_power_mw']:.2f} mW")
    
    # Platform recommendation
    print("\nRecommending hardware platforms...")
    recommender = EdgeAIPlatformRecommender()
    
    requirements = {
        'tops': tops_req,
        'memory_kb': memory['total_memory_kb'],
        'power_mw': power['total_power_mw']
    }
    
    platforms = recommender.recommend_platform(requirements)
    
    if platforms:
        print(f"\nTop 3 recommended platforms:")
        for i, platform in enumerate(platforms[:3], 1):
            print(f"{i}. {platform['name']} ({platform['tier']})")
            print(f"   Cost: ${platform['cost_usd']}, "
                  f"TOPS: {platform['tops']}, "
                  f"Score: {platform['score']:.3f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {regime.capitalize()} Regime')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / f'{regime}_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compile results
    results = {
        'regime': regime,
        'regime_description': trainer.get_regime_description(),
        'config': config,
        'training': {
            'epochs': epochs,
            'training_time_s': training_time,
            'final_loss': final_loss,
            'loss_history': loss_history,
            'statistics': final_stats
        },
        'structure': {
            'total_layers': len(structure.layers),
            'total_clusters': total_clusters,
            'compression_ratio': None  # Will be computed from weight analysis
        },
        'hardware': {
            'operations': ops,
            'tops_required': tops_req,
            'memory': memory,
            'power': power
        },
        'platforms': platforms[:5] if platforms else []
    }
    
    # Save results
    with open(save_dir / f'{regime}_results.json', 'w') as f:
        # Convert to JSON-serializable format
        results_json = {
            k: v if not isinstance(v, (torch.Tensor, np.ndarray)) else v.tolist() if hasattr(v, 'tolist') else str(v)
            for k, v in results.items()
        }
        json.dump(results_json, f, indent=2, default=str)
    
    return results


def compare_regimes(results: Dict[str, Dict], save_dir: Path):
    """
    Generate comprehensive comparison between three training regimes
    
    Args:
        results: Dictionary mapping regime name to results
        save_dir: Directory to save comparison
    """
    print(f"\n{'='*80}")
    print("THREE-REGIME COMPARISON ANALYSIS")
    print(f"{'='*80}\n")
    
    # Extract key metrics
    comparison = {}
    
    for regime, data in results.items():
        comparison[regime] = {
            'final_loss': data['training']['final_loss'],
            'training_time_s': data['training']['training_time_s'],
            'active_steps': data['training']['statistics']['active_steps'],
            'total_steps': data['training']['statistics']['total_steps'],
            'duty_cycle': data['training']['statistics']['actual_duty_cycle'],
            'total_clusters': data['structure']['total_clusters'],
            'operations': data['hardware']['operations']['total_operations'],
            'tops_required': data['hardware']['tops_required'],
            'memory_kb': data['hardware']['memory']['total_memory_kb'],
            'power_mw': data['hardware']['power']['total_power_mw'],
            'top_platform': data['platforms'][0]['name'] if data['platforms'] else 'None',
            'platform_cost': data['platforms'][0]['cost_usd'] if data['platforms'] else 0
        }
    
    # Print comparison table
    print("Performance Comparison:")
    print("-" * 100)
    print(f"{'Metric':<30} | {'Continuous':>20} | {'Passive':>20} | {'Active':>20}")
    print("-" * 100)
    
    metrics = [
        ('Final Loss', 'final_loss', '.6f'),
        ('Training Time (s)', 'training_time_s', '.2f'),
        ('Active Steps', 'active_steps', 'd'),
        ('Duty Cycle (%)', 'duty_cycle', '.1%'),
        ('Total Clusters', 'total_clusters', 'd'),
        ('Operations/Inference', 'operations', ',d'),
        ('TOPS Required', 'tops_required', '.6f'),
        ('Memory (KB)', 'memory_kb', '.2f'),
        ('Power (mW)', 'power_mw', '.2f'),
        ('Top Platform', 'top_platform', 's'),
        ('Platform Cost ($)', 'platform_cost', '.2f')
    ]
    
    for label, key, fmt in metrics:
        continuous_val = comparison['continuous'][key]
        passive_val = comparison['passive'][key]
        active_val = comparison['active'][key]
        
        if isinstance(continuous_val, str):
            print(f"{label:<30} | {continuous_val:>20} | {passive_val:>20} | {active_val:>20}")
        else:
            print(f"{label:<30} | {continuous_val:>20{fmt}} | {passive_val:>20{fmt}} | {active_val:>20{fmt}}")
    
    print("-" * 100)
    
    # Compute improvements
    print("\nImprovement Analysis (vs Continuous Baseline):")
    print("-" * 80)
    
    for regime in ['passive', 'active']:
        print(f"\n{regime.capitalize()} Regime:")
        
        ops_reduction = (comparison['continuous']['operations'] - comparison[regime]['operations']) / \
                       comparison['continuous']['operations'] * 100
        
        clusters_reduction = (comparison['continuous']['total_clusters'] - comparison[regime]['total_clusters']) / \
                            comparison['continuous']['total_clusters'] * 100
        
        cost_savings = comparison['continuous']['platform_cost'] - comparison[regime]['platform_cost']
        cost_savings_pct = cost_savings / comparison['continuous']['platform_cost'] * 100
        
        accuracy_change = (comparison[regime]['final_loss'] - comparison['continuous']['final_loss']) / \
                         comparison['continuous']['final_loss'] * 100
        
        print(f"  - Operations reduction: {ops_reduction:+.1f}%")
        print(f"  - Cluster reduction: {clusters_reduction:+.1f}%")
        print(f"  - Platform cost savings: ${cost_savings:.2f} ({cost_savings_pct:+.1f}%)")
        print(f"  - Loss change: {accuracy_change:+.1f}%")
    
    # Export comparison table (CSV)
    with open(save_dir / 'regime_comparison.csv', 'w') as f:
        f.write("Metric,Continuous,Passive,Active\n")
        for label, key, _ in metrics:
            continuous_val = comparison['continuous'][key]
            passive_val = comparison['passive'][key]
            active_val = comparison['active'][key]
            f.write(f"{label},{continuous_val},{passive_val},{active_val}\n")
    
    # Export LaTeX table
    with open(save_dir / 'regime_comparison.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Three-Regime Training Comparison for Burgers Equation}\n")
        f.write("\\label{tab:three_regime_comparison}\n")
        f.write("\\begin{tabular}{l|ccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Continuous} & \\textbf{Passive} & \\textbf{Active} \\\\\n")
        f.write("\\hline\n")
        
        for label, key, fmt in metrics[:9]:  # Exclude platform name
            continuous_val = comparison['continuous'][key]
            passive_val = comparison['passive'][key]
            active_val = comparison['active'][key]
            
            if 'f' in fmt:
                f.write(f"{label} & {continuous_val:{fmt}} & {passive_val:{fmt}} & {active_val:{fmt}} \\\\\n")
            elif 'd' in fmt or ',' in fmt:
                f.write(f"{label} & {continuous_val:,} & {passive_val:,} & {active_val:,} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nComparison tables saved to:")
    print(f"  - {save_dir / 'regime_comparison.csv'}")
    print(f"  - {save_dir / 'regime_comparison.tex'}")


def main():
    """Main experiment runner"""
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results' / 'three_regime_burgers'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("THREE-REGIME BURGERS EQUATION EXPERIMENT")
    print("Methodology: Section III-C (Lines 2306-2342)")
    print("="*80)
    
    # Experiment configuration
    config = {
        'node_num': 16,
        'learning_rate': 1e-3,
        'epochs': 3000,
        'reg_weight': 1e-4,
        'n_train_points': 1000,
        'physics_weight': 1.0,
        'boundary_weight': 1.0,
        'threshold': 0.1,
        'peak_solar_power': 300.0,       # Peak solar power (W)
        'gpu_power': 250.0,              # GPU training power (W)
        'solar_mode': 'simplified',      # 'simplified' = exactly 50% duty cycle
        'kappa': 2.0,
        'threshold_hours': 0.5,
        'checkpoint_interval': 500,
        'seed': 42
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run all three regimes
    regimes = ['continuous', 'passive', 'active']
    all_results = {}
    
    for regime in regimes:
        results = train_single_regime(regime, config, results_dir)
        all_results[regime] = results
    
    # Generate comparison
    compare_regimes(all_results, results_dir)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    print("\nKey findings:")
    print("1. Check regime_comparison.csv for detailed metrics")
    print("2. Review *_training_curve.png for training dynamics")
    print("3. Examine *_results.json for complete data")
    print("="*80)


def run_three_regime_burgers(epochs=3000, hidden_sizes=None, save_dir=None, seed=42):
    """
    Wrapper function for statistical validation and architecture sensitivity
    
    Args:
        epochs: Number of training epochs
        hidden_sizes: Network architecture (not used in PsiNN, kept for interface)
        save_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary with results for continuous, passive, and active regimes
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create results directory
    if save_dir is None:
        results_dir = Path(__file__).parent.parent / 'results' / 'three_regime_burgers'
    else:
        results_dir = Path(save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment configuration
    config = {
        'node_num': 16,
        'learning_rate': 1e-3,
        'epochs': epochs,
        'reg_weight': 1e-4,
        'n_train_points': 1000,
        'physics_weight': 1.0,
        'boundary_weight': 1.0,
        'threshold': 0.1,
        'peak_solar_power': 300.0,
        'gpu_power': 250.0,
        'solar_mode': 'simplified',
        'kappa': 2.0,
        'threshold_hours': 0.5,
        'checkpoint_interval': 500,
        'seed': seed
    }
    
    # Run all three regimes
    regimes = ['continuous', 'passive', 'active']
    all_results = {}
    
    for regime in regimes:
        results = train_single_regime(regime, config, results_dir)
        all_results[regime] = results
    
    # Generate comparison
    compare_regimes(all_results, results_dir)
    
    return all_results


if __name__ == '__main__':
    main()