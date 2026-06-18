"""
GPU Power Measurement During PINN Training
===========================================

This script measures actual GPU power consumption during PINN training.

Usage:
    python measure_gpu_power.py

Output:
    - Real-time power readings during training
    - Summary statistics (mean, max, min power)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import subprocess
import threading
import time
import json
import csv
import os
from datetime import datetime
from typing import List, Dict

# =============================================================================
# GPU POWER MONITORING
# =============================================================================

class GPUPowerMonitor:
    """Monitor GPU power consumption in a background thread."""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.power_readings: List[float] = []
        self.timestamps: List[float] = []
        self.running = False
        self.thread = None
        self.start_time = None
        
    def _sample_power(self):
        """Query nvidia-smi for current power draw."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            power = float(result.stdout.strip())
            return power
        except Exception as e:
            print(f"Error reading GPU power: {e}")
            return None
    
    def _monitor_loop(self):
        """Background thread that samples power."""
        self.start_time = time.time()
        while self.running:
            power = self._sample_power()
            if power is not None:
                self.power_readings.append(power)
                self.timestamps.append(time.time() - self.start_time)
                print(f"\r  GPU Power: {power:6.1f} W | Samples: {len(self.power_readings)}", end="", flush=True)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.power_readings = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("GPU power monitoring started...")
        
    def stop(self) -> Dict:
        """Stop monitoring and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("\n")
        
        if not self.power_readings:
            return {'error': 'No power readings collected'}
        
        readings = np.array(self.power_readings)
        return {
            'samples': len(readings),
            'duration_sec': self.timestamps[-1] if self.timestamps else 0,
            'mean_power_w': float(np.mean(readings)),
            'max_power_w': float(np.max(readings)),
            'min_power_w': float(np.min(readings)),
            'std_power_w': float(np.std(readings)),
            'readings': readings.tolist()
        }


# =============================================================================
# SIMPLE PINN (Same as manuscript experiments)
# =============================================================================

class SimplePINN(nn.Module):
    """Simplified PINN for Burgers equation (same as manuscript)."""
    
    def __init__(self, hidden_sizes: List[int] = [40, 40, 40]):
        super().__init__()
        
        layers = []
        in_size = 2  # (x, t)
        
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Tanh())
            in_size = h
        
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


def burgers_loss(model: nn.Module, x: torch.Tensor, t: torch.Tensor, 
                 nu: float = 0.01/np.pi):
    """Physics-informed loss for Burgers equation."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    inputs = torch.cat([x, t], dim=1)
    u = model(inputs)
    
    # Compute derivatives
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    # PDE residual
    pde_residual = u_t + u * u_x - nu * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Boundary conditions
    t_bc = torch.rand(100, 1, device=x.device)
    x_left = torch.full_like(t_bc, -1.0)
    x_right = torch.full_like(t_bc, 1.0)
    
    u_left = model(torch.cat([x_left, t_bc], dim=1))
    u_right = model(torch.cat([x_right, t_bc], dim=1))
    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)
    
    # Initial condition
    x_ic = torch.rand(100, 1, device=x.device) * 2 - 1
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_exact = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic - u_ic_exact)**2)
    
    total_loss = loss_pde + 10*loss_bc + 10*loss_ic
    return total_loss


# =============================================================================
# MAIN TEST
# =============================================================================

def run_power_measurement_test(epochs: int = 500, n_points: int = 2000):
    """
    Run PINN training while measuring GPU power consumption.
    
    Args:
        epochs: Number of training epochs (default 500 for ~1-2 min test)
        n_points: Collocation points per batch (same as manuscript: 2000)
    """
    print("="*70)
    print("GPU POWER MEASUREMENT DURING PINN TRAINING")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot measure GPU power.")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Training epochs: {epochs}")
    print(f"Collocation points: {n_points}")
    print(f"Network: 2 -> 40 -> 40 -> 40 -> 1 (same as manuscript)")
    
    # Initialize model
    model = SimplePINN([40, 40, 40]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate training data
    x = (torch.rand(n_points, 1, device=device) * 2 - 1)
    t = torch.rand(n_points, 1, device=device)
    
    # Warm up GPU
    print("\nWarming up GPU...")
    for _ in range(50):
        optimizer.zero_grad()
        loss = burgers_loss(model, x, t)
        loss.backward()
        optimizer.step()
    
    # Start power monitoring
    monitor = GPUPowerMonitor(sample_interval=0.5)
    monitor.start()
    
    print(f"\nTraining for {epochs} epochs...")
    print("-"*70)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = burgers_loss(model, x, t)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    
    # Stop monitoring and get results
    stats = monitor.stop()
    
    # Print results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    print(f"Epochs per second: {epochs/elapsed:.1f}")
    
    if 'error' not in stats:
        print(f"\nGPU Power Statistics:")
        print(f"  Mean Power:  {stats['mean_power_w']:.1f} W")
        print(f"  Max Power:   {stats['max_power_w']:.1f} W")
        print(f"  Min Power:   {stats['min_power_w']:.1f} W")
        print(f"  Std Dev:     {stats['std_power_w']:.1f} W")
        print(f"  Samples:     {stats['samples']}")
        
        print("\n" + "="*70)
        print("MANUSCRIPT VERIFICATION")
        print("="*70)
        
        mean_power = stats['mean_power_w']
        if 200 <= mean_power <= 300:
            print(f"\n✅ VERIFIED: Mean power ({mean_power:.0f}W) is close to manuscript's 250W claim")
        elif mean_power < 200:
            print(f"\n⚠️  Mean power ({mean_power:.0f}W) is LOWER than manuscript's 250W")
            print("   Consider updating manuscript or noting power-efficient training")
        else:
            print(f"\n⚠️  Mean power ({mean_power:.0f}W) is HIGHER than manuscript's 250W")
            print("   Consider updating manuscript to reflect actual power draw")
        
        print(f"\nRecommendation for manuscript:")
        print(f"  \"GPU power consumption during training: {mean_power:.0f}W (RTX 4090)\"")
    
    # Save results to CSV
    save_results(stats, "standard_test", epochs, n_points, [40, 40, 40])
    
    return stats


def save_results(stats: Dict, test_name: str, epochs: int, n_points: int, architecture: List[int]):
    """Save results to CSV and JSON files."""
    results_dir = "results/gpu_power_measurement"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary to JSON
    summary = {
        'timestamp': timestamp,
        'test_name': test_name,
        'epochs': epochs,
        'collocation_points': n_points,
        'architecture': architecture,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'mean_power_w': stats.get('mean_power_w', 0),
        'max_power_w': stats.get('max_power_w', 0),
        'min_power_w': stats.get('min_power_w', 0),
        'std_power_w': stats.get('std_power_w', 0),
        'samples': stats.get('samples', 0),
        'duration_sec': stats.get('duration_sec', 0),
        'manuscript_claim_w': 250,
        'discrepancy_percent': round((250 - stats.get('mean_power_w', 0)) / 250 * 100, 1)
    }
    
    json_path = os.path.join(results_dir, f"power_measurement_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save power readings to CSV
    csv_path = os.path.join(results_dir, f"power_readings_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'power_watts'])
        for i, power in enumerate(stats.get('readings', [])):
            writer.writerow([i, power])
    
    # Append to summary CSV
    summary_csv = os.path.join(results_dir, "all_measurements.csv")
    file_exists = os.path.exists(summary_csv)
    
    with open(summary_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'test_name', 'epochs', 'collocation_points',
                           'architecture', 'mean_power_w', 'max_power_w', 'min_power_w',
                           'std_power_w', 'samples', 'manuscript_claim_w', 'discrepancy_percent'])
        writer.writerow([timestamp, test_name, epochs, n_points,
                        '->'.join(map(str, architecture)),
                        round(stats.get('mean_power_w', 0), 1),
                        round(stats.get('max_power_w', 0), 1),
                        round(stats.get('min_power_w', 0), 1),
                        round(stats.get('std_power_w', 0), 1),
                        stats.get('samples', 0),
                        250, summary['discrepancy_percent']])
    
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {summary_csv}")


def run_manuscript_config_test():
    """
    Run with exact manuscript configuration:
    - 4 hidden layers × 50 neurons (as specified in Section IV.B.3)
    - 1000 collocation points per batch
    - 6000-10000 epochs
    """
    print("="*70)
    print("GPU POWER MEASUREMENT - MANUSCRIPT CONFIGURATION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Network: 2 -> 50 -> 50 -> 50 -> 50 -> 1 (manuscript config)")
    print(f"Collocation points: 1000 (manuscript config)")
    print(f"Epochs: 6000 (manuscript config)")
    
    # Build manuscript network: 4 hidden layers × 50 neurons
    model = SimplePINN([50, 50, 50, 50]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Manuscript uses 1000 collocation points
    x = (torch.rand(1000, 1, device=device) * 2 - 1)
    t = torch.rand(1000, 1, device=device)
    
    # Warm up
    print("\nWarming up GPU...")
    for _ in range(100):
        optimizer.zero_grad()
        loss = burgers_loss(model, x, t)
        loss.backward()
        optimizer.step()
    
    # Start monitoring
    monitor = GPUPowerMonitor(sample_interval=0.5)
    monitor.start()
    
    print(f"\nTraining for 6000 epochs (manuscript config)...")
    print("-"*70)
    
    start_time = time.time()
    
    for epoch in range(6000):
        optimizer.zero_grad()
        loss = burgers_loss(model, x, t)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    stats = monitor.stop()
    
    print("="*70)
    print("RESULTS - MANUSCRIPT CONFIGURATION")
    print("="*70)
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    print(f"Epochs per second: {6000/elapsed:.1f}")
    
    if 'error' not in stats:
        print(f"\nGPU Power Statistics:")
        print(f"  Mean Power:  {stats['mean_power_w']:.1f} W")
        print(f"  Max Power:   {stats['max_power_w']:.1f} W")
        print(f"  Min Power:   {stats['min_power_w']:.1f} W")
        print(f"  Std Dev:     {stats['std_power_w']:.1f} W")
        print(f"  Samples:     {stats['samples']}")
        
        mean_power = stats['mean_power_w']
        print("\n" + "="*70)
        print("MANUSCRIPT IMPACT")
        print("="*70)
        print(f"\nMeasured: {mean_power:.0f}W vs Manuscript claim: 250W")
        print(f"Difference: {abs(250 - mean_power):.0f}W ({abs(250-mean_power)/250*100:.0f}% lower)")
        
        if mean_power < 100:
            print("\n⚠️  SIGNIFICANT DISCREPANCY!")
            print("   The 250W claim overestimates actual power by >60%")
            print("\n   Options for manuscript:")
            print(f"   1. Update to measured value: ~{mean_power:.0f}W")
            print("   2. Note that 250W is TDP, actual training draws less")
            print("   3. This actually STRENGTHENS the solar feasibility argument")
            print("      (lower power = easier to achieve with smaller panels)")
    
    # Save results to CSV
    save_results(stats, "manuscript_config", 6000, 1000, [50, 50, 50, 50])
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--manuscript':
        # Run with exact manuscript configuration
        run_manuscript_config_test()
    else:
        # Parse command line args
        epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
        n_points = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        
        print(f"\nRunning with epochs={epochs}, n_points={n_points}")
        results = run_power_measurement_test(epochs=epochs, n_points=n_points)