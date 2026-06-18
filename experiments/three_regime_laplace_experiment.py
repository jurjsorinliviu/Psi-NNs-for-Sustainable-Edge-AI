"""
Three-Regime Laplace Equation Experiment

Compares three training regimes for Laplace equation:
1. Continuous (100% grid power, standard regularization)
2. Passive (50% solar duty, NO adaptive regularization)
3. Active (50% solar duty, WITH adaptive regularization)

Implements methodology from manuscript Section III-C (lines 2306-2342)
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add PSI-HDL implementation to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "PSI-HDL-implementation" / "Code"))
sys.path.insert(0, str(BASE_DIR / "chapter4"))

# Import PsiNN Laplace architecture
from PsiNN_laplace import Net as PsiNN_Laplace

# Import our Chapter 4 extensions
from sustainable_edge_ai import (
    SolarPowerModel,
    SolarConstrainedTrainer,
    HardwareSpecificationExtractor,
    EdgeAIPlatformRecommender
)


def generate_laplace_data(n_points=2000):
    """
    Generate training data for 2D Laplace equation: ∇²u = 0
    
    Boundary conditions:
    - u(x, 0) = 0 (bottom)
    - u(x, 1) = sin(πx) (top)
    - u(0, y) = 0 (left)
    - u(1, y) = 0 (right)
    
    Analytical solution: u(x,y) = sin(πx) * sinh(πy) / sinh(π)
    """
    print("[DATA] Generating Laplace equation training data...")
    
    # Domain sampling
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    
    # Analytical solution
    u = np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
    
    # Add boundary points
    n_boundary = 200
    x_boundary = []
    y_boundary = []
    u_boundary = []
    
    # Bottom boundary (y=0)
    x_b = np.linspace(0, 1, n_boundary)
    x_boundary.extend(x_b)
    y_boundary.extend([0.0] * n_boundary)
    u_boundary.extend([0.0] * n_boundary)
    
    # Top boundary (y=1)
    x_boundary.extend(x_b)
    y_boundary.extend([1.0] * n_boundary)
    u_boundary.extend(np.sin(np.pi * x_b))
    
    # Left boundary (x=0)
    y_b = np.linspace(0, 1, n_boundary)
    x_boundary.extend([0.0] * n_boundary)
    y_boundary.extend(y_b)
    u_boundary.extend([0.0] * n_boundary)
    
    # Right boundary (x=1)
    x_boundary.extend([1.0] * n_boundary)
    y_boundary.extend(y_b)
    u_boundary.extend([0.0] * n_boundary)
    
    # Combine interior and boundary
    x_all = np.concatenate([x, np.array(x_boundary)])
    y_all = np.concatenate([y, np.array(y_boundary)])
    u_all = np.concatenate([u, np.array(u_boundary)])
    
    print(f"  Generated {len(x_all)} points ({n_points} interior + {len(x_boundary)} boundary)")
    print(f"  Domain: [0,1] × [0,1]")
    print(f"  u range: [{u_all.min():.4f}, {u_all.max():.4f}]")
    
    return x_all.reshape(-1, 1), y_all.reshape(-1, 1), u_all.reshape(-1, 1)


def physics_loss(model, xy, device='cpu'):
    """
    Compute Laplace equation physics loss: ∇²u = ∂²u/∂x² + ∂²u/∂y² = 0
    """
    xy_tensor = torch.FloatTensor(xy).to(device)
    xy_tensor.requires_grad = True
    
    # Forward pass
    u = model(xy_tensor)
    
    # First derivatives
    du = torch.autograd.grad(
        u, xy_tensor,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]
    
    # Second derivatives
    d2u_dx2 = torch.autograd.grad(
        du_dx, xy_tensor,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0][:, 0:1]
    
    d2u_dy2 = torch.autograd.grad(
        du_dy, xy_tensor,
        grad_outputs=torch.ones_like(du_dy),
        create_graph=True
    )[0][:, 1:2]
    
    # Laplacian
    laplacian = d2u_dx2 + d2u_dy2
    
    return torch.mean(laplacian**2)


def boundary_loss(model, x, y, u_true, device='cpu'):
    """Compute loss on boundary conditions"""
    xy = np.hstack([x, y])
    xy_tensor = torch.FloatTensor(xy).to(device)
    u_true_tensor = torch.FloatTensor(u_true).to(device)
    
    u_pred = model(xy_tensor)
    
    return torch.mean((u_pred - u_true_tensor)**2)


def train_single_regime(regime_name, x_data, y_data, u_data, 
                       node_num=16, epochs=3000, lr=1e-3,
                       use_solar=False, adaptive_reg=False,
                       results_dir=None):
    """
    Train Laplace model under specified regime
    
    Args:
        regime_name: 'continuous', 'passive', or 'active'
        x_data, y_data, u_data: Training data
        node_num: Hidden layer size
        epochs: Training iterations
        lr: Learning rate
        use_solar: Enable solar power constraints
        adaptive_reg: Enable adaptive regularization
        results_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"TRAINING REGIME: {regime_name.upper()}")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Solar power: {'ENABLED' if use_solar else 'DISABLED'}")
    print(f"  Adaptive regularization: {'ENABLED' if adaptive_reg else 'DISABLED'}")
    print(f"  Architecture: [2, {node_num}, {node_num}, {2*node_num}, 1]")
    print(f"  Training steps: {epochs}")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PsiNN_Laplace(node_num=node_num, output_num=1).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup solar model if needed
    if use_solar:
        solar_model = SolarPowerModel(mode='simplified')
    else:
        solar_model = None
    
    # Prepare data
    xy_data = np.hstack([x_data, y_data])
    
    # Split into interior and boundary
    n_interior = 2000
    xy_interior = xy_data[:n_interior]
    xy_boundary = xy_data[n_interior:]
    u_boundary = u_data[n_interior:]
    
    # Training loop
    losses = []
    duty_cycle_log = []
    start_time = time.time()
    
    for step in range(epochs):
        if use_solar:
            # Check power availability
            power_available = solar_model.is_power_available(step)
            duty_cycle_log.append(1 if power_available else 0)
            
            if not power_available:
                continue
            
            # Get current regularization (adaptive if enabled)
            if adaptive_reg:
                # Simple adaptive: increase reg when approaching end
                reg_weight = 1e-4 * (1.0 + 0.5 * (step / epochs))
            else:
                reg_weight = 1e-4
        else:
            power_available = True
            reg_weight = 1e-4
        
        optimizer.zero_grad()
        
        # Physics loss (Laplace equation)
        loss_physics = physics_loss(model, xy_interior, device)
        
        # Boundary loss
        loss_boundary = boundary_loss(
            model, 
            xy_boundary[:, 0:1], 
            xy_boundary[:, 1:2],
            u_boundary,
            device
        )
        
        # Regularization loss
        loss_reg = 0
        for param in model.parameters():
            loss_reg += torch.sum(param**2)
        loss_reg = reg_weight * loss_reg
        
        # Combined loss
        loss = loss_physics + 10.0 * loss_boundary + loss_reg
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 500 == 0:
            duty = 100 * (sum(duty_cycle_log) / len(duty_cycle_log)) if duty_cycle_log else 100.0
            print(f"  Step {step+1:4d}/{epochs}: Loss={loss.item():.6f} "
                  f"(Physics={loss_physics.item():.6f}, Boundary={loss_boundary.item():.6f}) "
                  f"Duty={duty:.1f}%")
    
    training_time = time.time() - start_time
    final_loss = losses[-1]
    duty_cycle = 100 * (sum(duty_cycle_log) / len(duty_cycle_log)) if duty_cycle_log else 100.0
    
    print(f"\n  ✓ Training complete!")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Duty cycle: {duty_cycle:.2f}%")
    
    # Save results
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'regime': regime_name,
            'final_loss': float(final_loss),
            'training_time': float(training_time),
            'duty_cycle': float(duty_cycle),
            'epochs': epochs,
            'node_num': node_num,
            'use_solar': use_solar,
            'adaptive_reg': adaptive_reg,
            'losses': [float(l) for l in losses[::10]],  # Subsample
            'duty_log': duty_cycle_log[::10] if duty_cycle_log else []
        }
        
        results_file = results_dir / f"{regime_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to: {results_file}")
        
        # Save model
        model_file = results_dir / f"{regime_name}_model.pth"
        torch.save(model.state_dict(), model_file)
        print(f"  Saved model to: {model_file}")
    
    return {
        'model': model,
        'final_loss': final_loss,
        'training_time': training_time,
        'duty_cycle': duty_cycle,
        'losses': losses,
        'regime': regime_name
    }


def compare_regimes(results_continuous, results_passive, results_active, output_dir):
    """Generate comparison tables and visualizations"""
    print(f"\n{'='*80}")
    print("REGIME COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Comparison table
    data = {
        'Regime': ['Continuous', 'Passive', 'Active'],
        'Duty Cycle (%)': [
            results_continuous['duty_cycle'],
            results_passive['duty_cycle'],
            results_active['duty_cycle']
        ],
        'Final Loss': [
            results_continuous['final_loss'],
            results_passive['final_loss'],
            results_active['final_loss']
        ],
        'Training Time (s)': [
            results_continuous['training_time'],
            results_passive['training_time'],
            results_active['training_time']
        ]
    }
    
    # Calculate relative metrics
    baseline_loss = data['Final Loss'][0]
    data['Loss vs Baseline (%)'] = [
        100 * (l - baseline_loss) / baseline_loss for l in data['Final Loss']
    ]
    
    # Print table
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Regime':<15} {'Duty%':<10} {'Final Loss':<15} {'Δ vs Base':<15} {'Time(s)':<10}")
    print("-" * 80)
    for i in range(3):
        print(f"{data['Regime'][i]:<15} "
              f"{data['Duty Cycle (%)'][i]:>8.2f}% "
              f"{data['Final Loss'][i]:>13.6f} "
              f"{data['Loss vs Baseline (%)'][i]:>+13.2f}% "
              f"{data['Training Time (s)'][i]:>8.1f}")
    print("-" * 80)
    
    # Save CSV
    import csv
    csv_file = output_dir / "regime_comparison.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        for i in range(3):
            row = {k: data[k][i] for k in data.keys()}
            writer.writerow(row)
    print(f"\n✓ Saved CSV to: {csv_file}")
    
    # Save LaTeX table
    latex_file = output_dir / "regime_comparison.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Laplace Equation: Three-Regime Training Comparison}\n")
        f.write("\\label{tab:laplace_regimes}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("Regime & Duty Cycle & Final Loss & $\\Delta$ vs Base & Time(s) \\\\\n")
        f.write("\\hline\n")
        for i in range(3):
            f.write(f"{data['Regime'][i]} & "
                   f"{data['Duty Cycle (%)'][i]:.1f}\\% & "
                   f"{data['Final Loss'][i]:.6f} & "
                   f"{data['Loss vs Baseline (%)'][i]:+.2f}\\% & "
                   f"{data['Training Time (s)'][i]:.1f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"✓ Saved LaTeX to: {latex_file}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss comparison
    axes[0].bar(data['Regime'], data['Final Loss'], color=['blue', 'green', 'red'])
    axes[0].set_ylabel('Final Loss')
    axes[0].set_title('Training Loss by Regime')
    axes[0].grid(True, alpha=0.3)
    
    # Duty cycle
    axes[1].bar(data['Regime'], data['Duty Cycle (%)'], color=['blue', 'green', 'red'])
    axes[1].set_ylabel('Duty Cycle (%)')
    axes[1].set_title('Power Availability')
    axes[1].set_ylim([0, 105])
    axes[1].grid(True, alpha=0.3)
    
    # Training time
    axes[2].bar(data['Regime'], data['Training Time (s)'], color=['blue', 'green', 'red'])
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Training Duration')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "regime_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {plot_file}")
    plt.close()
    
    return data


def main():
    """Run complete three-regime Laplace experiment"""
    print("\n" + "="*80)
    print("CHAPTER 4: LAPLACE EQUATION - THREE REGIME EXPERIMENT")
    print("="*80)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "results" / "three_regime_laplace"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    x_data, y_data, u_data = generate_laplace_data(n_points=2000)
    
    # Run three regimes
    print("\n" + "="*80)
    print("PHASE 1: CONTINUOUS TRAINING (Baseline)")
    print("="*80)
    results_continuous = train_single_regime(
        'continuous',
        x_data, y_data, u_data,
        node_num=16,
        epochs=3000,
        use_solar=False,
        adaptive_reg=False,
        results_dir=output_dir
    )
    
    print("\n" + "="*80)
    print("PHASE 2: PASSIVE SOLAR TRAINING")
    print("="*80)
    results_passive = train_single_regime(
        'passive',
        x_data, y_data, u_data,
        node_num=16,
        epochs=3000,
        use_solar=True,
        adaptive_reg=False,
        results_dir=output_dir
    )
    
    print("\n" + "="*80)
    print("PHASE 3: ACTIVE SOLAR TRAINING")
    print("="*80)
    results_active = train_single_regime(
        'active',
        x_data, y_data, u_data,
        node_num=16,
        epochs=3000,
        use_solar=True,
        adaptive_reg=True,
        results_dir=output_dir
    )
    
    # Compare results
    comparison_data = compare_regimes(
        results_continuous,
        results_passive,
        results_active,
        output_dir
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey Finding:")
    passive_delta = comparison_data['Loss vs Baseline (%)'][1]
    active_delta = comparison_data['Loss vs Baseline (%)'][2]
    print(f"  Passive regime: {passive_delta:+.2f}% loss change at 50% duty cycle")
    print(f"  Active regime: {active_delta:+.2f}% loss change at 50% duty cycle")
    
    if abs(active_delta) < abs(passive_delta):
        print(f"  ✓ Active regularization IMPROVED performance!")
    else:
        print(f"  ✗ Active regularization did NOT improve over passive")
    
    return {
        'continuous': results_continuous,
        'passive': results_passive,
        'active': results_active,
        'comparison': comparison_data
    }


def run_three_regime_laplace(epochs=3000, hidden_sizes=None, save_dir=None, seed=42):
    """
    Wrapper function for statistical validation
    
    Args:
        epochs: Number of training epochs
        hidden_sizes: Network architecture (not used, kept for interface)
        save_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Dictionary with results for continuous, passive, and active regimes
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if save_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "three_regime_laplace"
    else:
        output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    x_data, y_data, u_data = generate_laplace_data(n_points=2000)
    
    results_continuous = train_single_regime(
        'continuous', x_data, y_data, u_data,
        node_num=16, epochs=epochs,
        use_solar=False, adaptive_reg=False,
        results_dir=output_dir
    )
    
    results_passive = train_single_regime(
        'passive', x_data, y_data, u_data,
        node_num=16, epochs=epochs,
        use_solar=True, adaptive_reg=False,
        results_dir=output_dir
    )
    
    results_active = train_single_regime(
        'active', x_data, y_data, u_data,
        node_num=16, epochs=epochs,
        use_solar=True, adaptive_reg=True,
        results_dir=output_dir
    )
    
    compare_regimes(results_continuous, results_passive, results_active, output_dir)
    
    return {
        'continuous': results_continuous,
        'passive': results_passive,
        'active': results_active
    }


if __name__ == "__main__":
    results = main()