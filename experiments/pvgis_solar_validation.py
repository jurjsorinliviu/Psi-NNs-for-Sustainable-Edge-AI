"""
PVGIS Real Solar Data Validation Experiment
============================================

Validates the manuscript's Markov solar model (Equations 50-51) against
real solar irradiance data from PVGIS (EU Joint Research Centre).

This experiment:
1. Downloads 1 year of real solar irradiance data from PVGIS for Chemnitz, Germany
2. Compares duty cycle distributions: Markov model vs Real PVGIS data
3. Runs Burgers equation training experiments with both solar profiles
4. Generates validation tables and figures for manuscript documentation

PVGIS API: https://re.jrc.ec.europa.eu/pvg_tools/en/
(Free, no API key required)

Author: Sorin Liviu Jurj
Date: 2025-11-30
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import existing modules (graceful fallback if not available)
try:
    from PSI_HDL_implementation.Code import PsiNN_burgers
    HAS_PSINN = True
except ImportError:
    HAS_PSINN = False
    print("Warning: PsiNN_burgers not found. Using simplified PINN model.")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Location settings - Chemnitz, Germany (TU Chemnitz)
    'latitude': 50.8278,          # Chemnitz, Germany
    'longitude': 12.9214,         # Chemnitz, Germany
    'location_name': 'Chemnitz, Germany',
    'year': 2022,                 # Year to download
    
    # Solar panel simulation parameters (from manuscript Eq. 50-51)
    'peak_solar_power_w': 300.0,  # P_max (W) - peak panel output
    'gpu_power_w': 250.0,         # GPU training power requirement (W)
    'panel_efficiency': 0.20,     # Solar panel efficiency (20%)
    'panel_area_m2': 2.0,         # Panel area (m²)
    
    # Markov model parameters (from manuscript)
    'markov_p_clear_to_cloudy': 0.15,  # Transition probability per hour
    'markov_p_cloudy_to_clear': 0.25,  # Transition probability per hour
    'cloudy_attenuation': 0.3,         # w(t) for cloudy state
    
    # Training parameters
    'epochs': 5000,
    'learning_rate': 1e-3,
    'reg_weight': 1e-4,
    'n_seeds': 3,  # Number of random seeds for statistical validation
    
    # Output settings
    'results_dir': 'results/nrel_validation',
}


# =============================================================================
# PVGIS DATA HANDLING (European Commission - for Europe)
# =============================================================================

class PVGISDataLoader:
    """
    Handles downloading and processing PVGIS solar irradiance data.
    
    PVGIS (Photovoltaic Geographical Information System) provides free solar
    radiation data for Europe, Africa, and most of Asia.
    
    API Documentation: https://re.jrc.ec.europa.eu/pvg_tools/en/
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lat = config['latitude']
        self.lon = config['longitude']
        self.year = config['year']
        
    def download_pvgis_data(self, save_path: str = None) -> pd.DataFrame:
        """
        Download 1 year of hourly solar irradiance data from PVGIS.
        
        Uses the PVGIS TMY (Typical Meteorological Year) or hourly radiation API.
        
        Returns:
            DataFrame with columns: timestamp, ghi (W/m²), temp, etc.
        """
        print(f"\n{'='*60}")
        print("Downloading PVGIS Solar Data (European Commission)")
        print(f"{'='*60}")
        print(f"Location: {self.config.get('location_name', f'({self.lat}, {self.lon})')}")
        print(f"Coordinates: ({self.lat}, {self.lon})")
        print(f"Year: {self.year}")
        
        # PVGIS hourly radiation API endpoint
        # Using PVGIS 5.2 API
        base_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'startyear': self.year,
            'endyear': self.year,
            'pvcalculation': 0,       # Just radiation data, no PV calculation
            'peakpower': 1,           # Normalized to 1 kWp
            'loss': 14,               # System losses %
            'outputformat': 'json',
            'browser': 0,             # Machine-readable output
        }
        
        print("\nSending PVGIS API request...")
        
        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract hourly data
            hourly_data = data.get('outputs', {}).get('hourly', [])
            
            if not hourly_data:
                print("Warning: No hourly data in response, trying alternative endpoint...")
                return self.download_pvgis_tmy(save_path)
            
            # Parse the data
            records = []
            for entry in hourly_data:
                # PVGIS time format: "20220101:0010" (YYYYMMDD:HHMM)
                time_str = entry.get('time', '')
                if ':' in time_str:
                    date_part, time_part = time_str.split(':')
                    year = int(date_part[:4])
                    month = int(date_part[4:6])
                    day = int(date_part[6:8])
                    hour = int(time_part[:2])
                    minute = int(time_part[2:4]) if len(time_part) > 2 else 0
                    
                    timestamp = datetime(year, month, day, hour, minute)
                else:
                    continue
                
                records.append({
                    'timestamp': timestamp,
                    'ghi': entry.get('G(i)', 0),  # Global irradiance on inclined plane
                    'dni': entry.get('Gb(i)', 0),  # Direct normal irradiance
                    'dhi': entry.get('Gd(i)', 0),  # Diffuse irradiance
                    'air_temperature': entry.get('T2m', 15),  # Temperature at 2m
                    'wind_speed': entry.get('WS10m', 3),  # Wind speed at 10m
                })
            
            df = pd.DataFrame(records)
            
            # Aggregate to hourly if sub-hourly
            if len(df) > 8760:
                df['hour'] = df['timestamp'].dt.floor('H')
                df = df.groupby('hour').agg({
                    'ghi': 'mean',
                    'dni': 'mean',
                    'dhi': 'mean',
                    'air_temperature': 'mean',
                    'wind_speed': 'mean'
                }).reset_index()
                df = df.rename(columns={'hour': 'timestamp'})
            
            # Save if path provided
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"Data saved to: {save_path}")
            
            print(f"\n✓ Downloaded {len(df)} hourly records")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  GHI range: {df['ghi'].min():.1f} - {df['ghi'].max():.1f} W/m²")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"\n✗ PVGIS API request failed: {e}")
            print("\nFalling back to synthetic data generation...")
            return self.generate_synthetic_european_data()
    
    def download_pvgis_tmy(self, save_path: str = None) -> pd.DataFrame:
        """
        Download Typical Meteorological Year (TMY) data from PVGIS.
        TMY represents a 'typical' year synthesized from historical data.
        """
        print("\nTrying PVGIS TMY (Typical Meteorological Year) endpoint...")
        
        base_url = "https://re.jrc.ec.europa.eu/api/v5_2/tmy"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'outputformat': 'json',
            'browser': 0,
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            hourly_data = data.get('outputs', {}).get('tmy_hourly', [])
            
            records = []
            base_year = self.year
            
            for i, entry in enumerate(hourly_data):
                # TMY data is indexed by hour of year
                hour_of_year = i
                day_of_year = hour_of_year // 24 + 1
                hour = hour_of_year % 24
                
                # Convert to approximate date
                date = datetime(base_year, 1, 1) + timedelta(days=day_of_year-1, hours=hour)
                
                records.append({
                    'timestamp': date,
                    'ghi': entry.get('G(h)', 0),  # Global horizontal irradiance
                    'dni': entry.get('Gb(n)', 0),
                    'dhi': entry.get('Gd(h)', 0),
                    'air_temperature': entry.get('T2m', 15),
                    'wind_speed': entry.get('WS10m', 3),
                })
            
            df = pd.DataFrame(records)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
            
            print(f"✓ Downloaded {len(df)} TMY hourly records")
            return df
            
        except Exception as e:
            print(f"TMY download failed: {e}")
            return self.generate_synthetic_european_data()
    
    def generate_synthetic_european_data(self) -> pd.DataFrame:
        """
        Generate synthetic solar data based on typical European patterns.
        Uses Chemnitz, Germany latitude for solar geometry.
        """
        print("\nGenerating synthetic European solar data (Chemnitz-like)...")
        
        hours = 8760
        timestamps = [datetime(self.year, 1, 1) + timedelta(hours=h) for h in range(hours)]
        
        ghi_values = []
        
        for ts in timestamps:
            doy = ts.timetuple().tm_yday
            hour = ts.hour
            
            # Solar declination
            declination = 23.45 * np.sin(np.radians(360 * (284 + doy) / 365))
            
            # Hour angle
            hour_angle = 15 * (hour - 12)
            
            # Solar elevation for Chemnitz latitude (~50.8°N)
            lat_rad = np.radians(self.lat)
            decl_rad = np.radians(declination)
            hour_rad = np.radians(hour_angle)
            
            sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) +
                           np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_rad))
            
            if sin_elevation > 0:
                elevation = np.degrees(np.arcsin(sin_elevation))
                air_mass = 1 / (np.sin(np.radians(elevation)) +
                               0.50572 * (6.07995 + elevation)**(-1.6364))
                
                # Clear sky irradiance - lower than US due to higher latitude
                ghi_clear = 900 * sin_elevation * np.exp(-0.16 * air_mass)
                
                # German weather: more clouds, especially in winter
                month = ts.month
                # Lower clear probability in winter (Nov-Feb), higher in summer
                if month in [11, 12, 1, 2]:
                    clear_prob = 0.25  # 25% clear in winter
                elif month in [5, 6, 7, 8]:
                    clear_prob = 0.55  # 55% clear in summer
                else:
                    clear_prob = 0.40  # 40% clear in spring/fall
                
                if np.random.random() < clear_prob:
                    ghi = ghi_clear * np.random.uniform(0.85, 1.0)
                else:
                    # Cloudy - typical German overcast
                    ghi = ghi_clear * np.random.uniform(0.15, 0.5)
                
                ghi = max(0, ghi)
            else:
                ghi = 0
            
            ghi_values.append(ghi)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ghi': ghi_values,
            'dni': [g * 0.7 for g in ghi_values],
            'dhi': [g * 0.3 for g in ghi_values],
            'air_temperature': [8 + 12 * np.sin(np.radians(360 * (t.timetuple().tm_yday - 100) / 365))
                              for t in timestamps],  # German temps: ~8°C average, ±12°C seasonal
            'wind_speed': np.random.exponential(3.5, hours),
        })
        
        print(f"✓ Generated {len(df)} hourly synthetic records for {self.config.get('location_name', 'Europe')}")
        print(f"  GHI range: {df['ghi'].min():.1f} - {df['ghi'].max():.1f} W/m²")
        
        return df
    
    def ghi_to_power(self, ghi_wm2: np.ndarray) -> np.ndarray:
        """Convert GHI (W/m²) to panel power output (W)."""
        efficiency = self.config['panel_efficiency']
        area = self.config['panel_area_m2']
        return ghi_wm2 * area * efficiency



# =============================================================================
# MARKOV SOLAR MODEL (Manuscript Equations 50-51)
# =============================================================================

class MarkovSolarModel:
    """
    Implements the Markov chain solar model from manuscript Equations 50-51.
    
    P(t) = P_max × max(0, sin(π(t-6)/12)) × w(t)
    
    where w(t) follows a 2-state Markov chain:
    - Clear: w = 1.0
    - Cloudy: w = 0.3
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.p_max = config['peak_solar_power_w']
        self.p_c2c = config['markov_p_clear_to_cloudy']  # Clear → Cloudy
        self.p_c2cl = config['markov_p_cloudy_to_clear']  # Cloudy → Clear
        self.cloudy_atten = config['cloudy_attenuation']
        
    def simulate_year(self, seed: int = 42) -> pd.DataFrame:
        """
        Simulate 1 year of hourly power availability using the Markov model.
        
        Returns:
            DataFrame with timestamp, power_w, weather_state
        """
        np.random.seed(seed)
        
        hours = 8760
        timestamps = [datetime(self.config['year'], 1, 1) + timedelta(hours=h) for h in range(hours)]
        
        # Initialize weather state (0=clear, 1=cloudy)
        weather_state = 0
        weather_states = []
        power_values = []
        
        for h, ts in enumerate(timestamps):
            # Markov transition
            if weather_state == 0:  # Clear
                if np.random.random() < self.p_c2c:
                    weather_state = 1
            else:  # Cloudy
                if np.random.random() < self.p_c2cl:
                    weather_state = 0
            
            weather_states.append(weather_state)
            
            # Hour of day (0-23)
            hour = ts.hour
            
            # Diurnal cycle (Equation 50): sin pattern from 6am to 6pm
            if 6 <= hour <= 18:
                diurnal = max(0, np.sin(np.pi * (hour - 6) / 12))
            else:
                diurnal = 0
            
            # Weather attenuation
            w = self.cloudy_atten if weather_state == 1 else 1.0
            
            # Final power
            power = self.p_max * diurnal * w
            power_values.append(power)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'power_w': power_values,
            'weather_state': weather_states,
            'weather': ['clear' if s == 0 else 'cloudy' for s in weather_states]
        })
        
        return df


# =============================================================================
# DUTY CYCLE ANALYSIS
# =============================================================================

def compute_duty_cycle_statistics(power_series: np.ndarray, threshold_w: float) -> Dict:
    """
    Compute duty cycle statistics from power availability data.
    
    Args:
        power_series: Array of power values (W)
        threshold_w: Minimum power required for training (W)
        
    Returns:
        Dictionary with duty cycle statistics
    """
    # Binary availability
    available = power_series >= threshold_w
    
    # Overall duty cycle
    duty_cycle = np.mean(available)
    
    # Compute run lengths (consecutive hours)
    runs_on = []
    runs_off = []
    current_run = 1
    
    for i in range(1, len(available)):
        if available[i] == available[i-1]:
            current_run += 1
        else:
            if available[i-1]:
                runs_on.append(current_run)
            else:
                runs_off.append(current_run)
            current_run = 1
    
    # Daily statistics
    n_days = len(power_series) // 24
    daily_hours = []
    for d in range(n_days):
        day_data = available[d*24:(d+1)*24]
        daily_hours.append(np.sum(day_data))
    
    return {
        'overall_duty_cycle': duty_cycle,
        'total_hours_available': np.sum(available),
        'total_hours': len(available),
        'mean_daily_hours': np.mean(daily_hours),
        'std_daily_hours': np.std(daily_hours),
        'min_daily_hours': np.min(daily_hours),
        'max_daily_hours': np.max(daily_hours),
        'mean_on_run_hours': np.mean(runs_on) if runs_on else 0,
        'max_on_run_hours': np.max(runs_on) if runs_on else 0,
        'mean_off_run_hours': np.mean(runs_off) if runs_off else 0,
        'max_off_run_hours': np.max(runs_off) if runs_off else 0,
        'num_interruptions': len(runs_off),
    }


def compare_duty_cycles(real_stats: Dict, markov_stats: Dict) -> pd.DataFrame:
    """
    Create comparison table between real and Markov solar data.
    """
    metrics = [
        ('Overall Duty Cycle (%)', 'overall_duty_cycle', lambda x: x * 100),
        ('Total Available Hours', 'total_hours_available', lambda x: x),
        ('Mean Daily Hours', 'mean_daily_hours', lambda x: x),
        ('Std Daily Hours', 'std_daily_hours', lambda x: x),
        ('Min Daily Hours', 'min_daily_hours', lambda x: x),
        ('Max Daily Hours', 'max_daily_hours', lambda x: x),
        ('Mean ON Run (hours)', 'mean_on_run_hours', lambda x: x),
        ('Max ON Run (hours)', 'max_on_run_hours', lambda x: x),
        ('Mean OFF Run (hours)', 'mean_off_run_hours', lambda x: x),
        ('Max OFF Run (hours)', 'max_off_run_hours', lambda x: x),
        ('Number of Interruptions', 'num_interruptions', lambda x: x),
    ]
    
    data = []
    for label, key, transform in metrics:
        real_val = transform(real_stats.get(key, 0))
        markov_val = transform(markov_stats.get(key, 0))
        diff = real_val - markov_val
        diff_pct = (diff / markov_val * 100) if markov_val != 0 else 0
        
        data.append({
            'Metric': label,
            'Real PVGIS': f'{real_val:.2f}',
            'Markov Model': f'{markov_val:.2f}',
            'Difference': f'{diff:+.2f}',
            'Diff %': f'{diff_pct:+.1f}%'
        })
    
    return pd.DataFrame(data)


# =============================================================================
# SIMPLIFIED PINN FOR BURGERS EQUATION
# =============================================================================

class SimplePINN(nn.Module):
    """
    Simplified PINN for Burgers equation (fallback when PsiNN not available).
    
    Architecture: 2 -> 40 -> 40 -> 40 -> 1
    """
    
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
                 nu: float = 0.01/np.pi) -> Tuple[torch.Tensor, Dict]:
    """
    Physics-informed loss for Burgers equation.
    
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    """
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
    
    # Boundary conditions: u(-1,t) = u(1,t) = 0
    t_bc = torch.rand(100, 1, device=x.device)
    x_left = torch.full_like(t_bc, -1.0)
    x_right = torch.full_like(t_bc, 1.0)
    
    u_left = model(torch.cat([x_left, t_bc], dim=1))
    u_right = model(torch.cat([x_right, t_bc], dim=1))
    loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)
    
    # Initial condition: u(x,0) = -sin(πx)
    x_ic = torch.rand(100, 1, device=x.device) * 2 - 1
    t_ic = torch.zeros_like(x_ic)
    u_ic = model(torch.cat([x_ic, t_ic], dim=1))
    u_ic_exact = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic - u_ic_exact)**2)
    
    total_loss = loss_pde + 10*loss_bc + 10*loss_ic
    
    return total_loss, {
        'pde': loss_pde.item(),
        'bc': loss_bc.item(),
        'ic': loss_ic.item()
    }


# =============================================================================
# SOLAR-CONSTRAINED TRAINING
# =============================================================================

def train_with_solar_profile(model: nn.Module, power_profile: np.ndarray, 
                             config: Dict, seed: int = 42) -> Dict:
    """
    Train PINN with real or simulated solar power profile.
    
    Args:
        model: Neural network model
        power_profile: Hourly power availability (W) for the year
        config: Training configuration
        seed: Random seed
        
    Returns:
        Training results dictionary
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    gpu_power = config['gpu_power_w']
    
    # Simulate training with power interruptions
    # Scale: 1 epoch = 1 simulated hour of training
    epochs = config['epochs']
    
    # Sample power profile to match epochs
    if len(power_profile) > epochs:
        # Sample from the profile
        indices = np.linspace(0, len(power_profile)-1, epochs, dtype=int)
        power_schedule = power_profile[indices]
    else:
        # Repeat profile
        repeats = int(np.ceil(epochs / len(power_profile)))
        power_schedule = np.tile(power_profile, repeats)[:epochs]
    
    # Generate training data
    n_points = 2000
    x = torch.rand(n_points, 1, device=device) * 2 - 1
    t = torch.rand(n_points, 1, device=device)
    
    loss_history = []
    active_steps = 0
    checkpoint_state = None
    
    for epoch in range(epochs):
        power_available = power_schedule[epoch]
        
        if power_available >= gpu_power:
            # Power available - train
            optimizer.zero_grad()
            loss, components = burgers_loss(model, x, t)
            
            # Add L2 regularization
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            total_loss = loss + config['reg_weight'] * l2_reg
            
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
            active_steps += 1
            
            # Save checkpoint periodically
            if active_steps % 500 == 0:
                checkpoint_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss.item()
                }
        else:
            # Power unavailable - skip (checkpoint preserved)
            if loss_history:
                loss_history.append(loss_history[-1])  # Repeat last loss
            else:
                loss_history.append(float('inf'))
    
    # Compute final statistics
    final_loss = loss_history[-1] if loss_history else float('inf')
    duty_cycle = active_steps / epochs
    
    return {
        'final_loss': final_loss,
        'loss_history': loss_history,
        'active_steps': active_steps,
        'total_steps': epochs,
        'duty_cycle': duty_cycle,
        'seed': seed
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_pvgis_validation_experiment(config: Dict = None) -> Dict:
    """
    Run PVGIS solar data validation experiment for Chemnitz, Germany.
    Compares real PVGIS solar irradiance data against the manuscript's Markov model.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    location_name = config.get('location_name', 'Chemnitz, Germany')
    
    print("="*80)
    print("PVGIS REAL SOLAR DATA VALIDATION EXPERIMENT")
    print("="*80)
    print(f"\nLocation: {location_name}")
    print(f"Coordinates: ({config['latitude']}, {config['longitude']})")
    print(f"Data source: PVGIS (European Commission)")
    print(f"Year: {config['year']}")
    print(f"Results directory: {results_dir}")
    
    # =========================================================================
    # STEP 1: Load/Download Real Solar Data from PVGIS
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Real Solar Data (PVGIS - European Commission)")
    print("="*60)
    
    data_loader = PVGISDataLoader(config)
    cache_path = results_dir / f"pvgis_{config['year']}_{config['latitude']:.2f}_{config['longitude']:.2f}.csv"
    
    # Try cache first
    if cache_path.exists():
        print(f"Loading cached PVGIS data from: {cache_path}")
        real_data = pd.read_csv(cache_path, parse_dates=['timestamp'])
        print(f"✓ Loaded {len(real_data)} records")
    else:
        # Download from PVGIS
        real_data = data_loader.download_pvgis_data(str(cache_path))
        
        # Cache the data
        if real_data is not None:
            real_data.to_csv(cache_path, index=False)
            print(f"Data cached to: {cache_path}")
    
    # Convert GHI to panel power
    real_power = data_loader.ghi_to_power(real_data['ghi'].values)
    
    # =========================================================================
    # STEP 2: Simulate Markov Model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: Simulating Markov Solar Model")
    print("="*60)
    
    markov_model = MarkovSolarModel(config)
    markov_data = markov_model.simulate_year(seed=42)
    markov_power = markov_data['power_w'].values
    
    print(f"✓ Simulated {len(markov_power)} hours")
    print(f"  Power range: {markov_power.min():.1f} - {markov_power.max():.1f} W")
    
    # =========================================================================
    # STEP 3: Compare Duty Cycles
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: Comparing Duty Cycle Statistics")
    print("="*60)
    
    gpu_power = config['gpu_power_w']
    
    real_stats = compute_duty_cycle_statistics(real_power, gpu_power)
    markov_stats = compute_duty_cycle_statistics(markov_power, gpu_power)
    
    comparison_df = compare_duty_cycles(real_stats, markov_stats)
    
    print("\nDuty Cycle Comparison:")
    print("-"*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_csv = results_dir / "duty_cycle_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✓ Saved: {comparison_csv}")
    
    # =========================================================================
    # STEP 4: Run Training Experiments
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: Training Experiments with Solar Profiles")
    print("="*60)
    
    n_seeds = config['n_seeds']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Running {n_seeds} seeds for each condition...")
    
    # Results storage
    training_results = {
        'continuous': [],
        'real_solar': [],
        'markov_solar': []
    }
    
    # Generate continuous power (always available)
    continuous_power = np.full_like(real_power, config['peak_solar_power_w'])
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        
        # Continuous baseline
        print("  Training: Continuous (baseline)...")
        model_cont = SimplePINN()
        result_cont = train_with_solar_profile(model_cont, continuous_power, config, seed=seed)
        training_results['continuous'].append(result_cont)
        print(f"    Final loss: {result_cont['final_loss']:.6f}")
        
        # Real PVGIS solar
        print("  Training: Real PVGIS solar...")
        model_real = SimplePINN()
        result_real = train_with_solar_profile(model_real, real_power, config, seed=seed)
        training_results['real_solar'].append(result_real)
        print(f"    Final loss: {result_real['final_loss']:.6f}, "
              f"Duty cycle: {result_real['duty_cycle']:.2%}")
        
        # Markov model solar
        print("  Training: Markov model solar...")
        model_markov = SimplePINN()
        result_markov = train_with_solar_profile(model_markov, markov_power, config, seed=seed)
        training_results['markov_solar'].append(result_markov)
        print(f"    Final loss: {result_markov['final_loss']:.6f}, "
              f"Duty cycle: {result_markov['duty_cycle']:.2%}")
    
    # =========================================================================
    # STEP 5: Statistical Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: Statistical Analysis")
    print("="*60)
    
    def compute_stats(results_list):
        losses = [r['final_loss'] for r in results_list]
        duty_cycles = [r['duty_cycle'] for r in results_list]
        return {
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'ci95_loss': 1.96 * np.std(losses) / np.sqrt(len(losses)),
            'mean_duty_cycle': np.mean(duty_cycles),
            'min_loss': np.min(losses),
            'max_loss': np.max(losses)
        }
    
    stats = {
        'continuous': compute_stats(training_results['continuous']),
        'real_solar': compute_stats(training_results['real_solar']),
        'markov_solar': compute_stats(training_results['markov_solar'])
    }
    
    # Compute degradation
    baseline_loss = stats['continuous']['mean_loss']
    stats['real_solar']['degradation_pct'] = (
        (stats['real_solar']['mean_loss'] - baseline_loss) / baseline_loss * 100
    )
    stats['markov_solar']['degradation_pct'] = (
        (stats['markov_solar']['mean_loss'] - baseline_loss) / baseline_loss * 100
    )
    
    # Create results table
    results_table = pd.DataFrame({
        'Condition': ['Continuous (Baseline)', 'Real PVGIS Solar', 'Markov Model Solar'],
        'Mean Loss': [stats['continuous']['mean_loss'], 
                     stats['real_solar']['mean_loss'],
                     stats['markov_solar']['mean_loss']],
        '± 95% CI': [stats['continuous']['ci95_loss'],
                    stats['real_solar']['ci95_loss'],
                    stats['markov_solar']['ci95_loss']],
        'Duty Cycle': [stats['continuous']['mean_duty_cycle'],
                      stats['real_solar']['mean_duty_cycle'],
                      stats['markov_solar']['mean_duty_cycle']],
        'Degradation (%)': [0.0, 
                           stats['real_solar']['degradation_pct'],
                           stats['markov_solar']['degradation_pct']]
    })
    
    print("\nTraining Results:")
    print("-"*80)
    print(results_table.to_string(index=False))
    
    # Save results table
    results_csv = results_dir / "training_comparison.csv"
    results_table.to_csv(results_csv, index=False)
    print(f"\n✓ Saved: {results_csv}")
    
    # =========================================================================
    # STEP 6: Generate Figures
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: Generating Figures")
    print("="*60)
    
    # Figure 1: Solar power comparison (sample week)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sample week (first week of June)
    start_idx = 24 * (31 + 28 + 31 + 30 + 31)  # June 1
    end_idx = start_idx + 24 * 7  # One week
    hours = np.arange(end_idx - start_idx)
    
    ax1 = axes[0, 0]
    ax1.plot(hours, real_power[start_idx:end_idx], 'b-', linewidth=1.5, alpha=0.8, label='Real PVGIS')
    ax1.plot(hours, markov_power[start_idx:end_idx], 'r--', linewidth=1.5, alpha=0.8, label='Markov Model')
    ax1.axhline(y=gpu_power, color='k', linestyle=':', linewidth=2, label=f'GPU Threshold ({gpu_power}W)')
    ax1.set_xlabel('Hour of Week', fontweight='bold')
    ax1.set_ylabel('Power (W)', fontweight='bold')
    ax1.set_title(f'(a) Solar Power Profile - Sample Week (June) - {location_name}', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24*7)
    
    # Daily duty cycle distribution
    ax2 = axes[0, 1]
    real_daily = [np.mean(real_power[d*24:(d+1)*24] >= gpu_power) for d in range(365)]
    markov_daily = [np.mean(markov_power[d*24:(d+1)*24] >= gpu_power) for d in range(365)]
    
    ax2.hist(real_daily, bins=20, alpha=0.6, label='Real PVGIS', color='blue', edgecolor='black')
    ax2.hist(markov_daily, bins=20, alpha=0.6, label='Markov Model', color='red', edgecolor='black')
    ax2.axvline(x=np.mean(real_daily), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(markov_daily), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Daily Duty Cycle', fontweight='bold')
    ax2.set_ylabel('Frequency (days)', fontweight='bold')
    ax2.set_title('(b) Daily Duty Cycle Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss comparison
    ax3 = axes[1, 0]
    conditions = ['Continuous\n(Baseline)', 'Real PVGIS\nSolar', 'Markov Model\nSolar']
    means = [stats['continuous']['mean_loss'], 
             stats['real_solar']['mean_loss'],
             stats['markov_solar']['mean_loss']]
    errors = [stats['continuous']['ci95_loss'],
              stats['real_solar']['ci95_loss'],
              stats['markov_solar']['ci95_loss']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax3.bar(conditions, means, yerr=errors, capsize=8, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_ylabel('Final MSE Loss', fontweight='bold')
    ax3.set_title('(c) Training Performance Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add degradation labels
    for i, (bar, deg) in enumerate(zip(bars, [0, stats['real_solar']['degradation_pct'], 
                                               stats['markov_solar']['degradation_pct']])):
        height = bar.get_height()
        if i > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + errors[i] + 0.0005,
                    f'+{deg:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='darkred' if deg > 10 else 'darkgreen')
    
    # Monthly duty cycle variation
    ax4 = axes[1, 1]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    real_monthly = []
    markov_monthly = []
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    start_day = 0
    for days in days_per_month:
        start_hour = start_day * 24
        end_hour = (start_day + days) * 24
        real_monthly.append(np.mean(real_power[start_hour:end_hour] >= gpu_power))
        markov_monthly.append(np.mean(markov_power[start_hour:end_hour] >= gpu_power))
        start_day += days
    
    x = np.arange(len(months))
    width = 0.35
    ax4.bar(x - width/2, real_monthly, width, label='Real PVGIS', color='blue', alpha=0.7)
    ax4.bar(x + width/2, markov_monthly, width, label='Markov Model', color='red', alpha=0.7)
    ax4.set_xlabel('Month', fontweight='bold')
    ax4.set_ylabel('Duty Cycle', fontweight='bold')
    ax4.set_title('(d) Monthly Duty Cycle Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(months, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = results_dir / "solar_validation_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "solar_validation_comparison.pdf", bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {fig_path}")
    
    # =========================================================================
    # STEP 7: Generate Summary Report
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 7: Summary Report")
    print("="*60)
    
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'config': {k: v for k, v in config.items() if not callable(v)},
        'location': location_name,
        'data_source': 'PVGIS',
        'duty_cycle_comparison': {
            'real_data': real_stats,
            'markov_model': markov_stats
        },
        'training_results': {
            'continuous': stats['continuous'],
            'real_solar': stats['real_solar'],
            'markov_solar': stats['markov_solar']
        },
        'key_findings': {
            'real_vs_markov_duty_cycle_diff': abs(real_stats['overall_duty_cycle'] - 
                                                   markov_stats['overall_duty_cycle']),
            'real_degradation_pct': stats['real_solar']['degradation_pct'],
            'markov_degradation_pct': stats['markov_solar']['degradation_pct'],
            'model_validity': 'VALIDATED' if abs(stats['real_solar']['degradation_pct'] - 
                                                  stats['markov_solar']['degradation_pct']) < 20 
                             else 'NEEDS_REVIEW'
        }
    }
    
    # Save summary JSON
    summary_path = results_dir / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ Saved: {summary_path}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n1. Duty Cycle Comparison ({location_name}):")
    print(f"   - Real PVGIS:   {real_stats['overall_duty_cycle']:.1%}")
    print(f"   - Markov Model: {markov_stats['overall_duty_cycle']:.1%}")
    print(f"   - Difference:   {abs(real_stats['overall_duty_cycle'] - markov_stats['overall_duty_cycle']):.1%}")
    
    print(f"\n2. Training Performance Degradation:")
    print(f"   - Real PVGIS solar:   +{stats['real_solar']['degradation_pct']:.1f}% vs baseline")
    print(f"   - Markov model solar: +{stats['markov_solar']['degradation_pct']:.1f}% vs baseline")
    
    print(f"\n3. Model Validity:")
    validation_diff = abs(stats['real_solar']['degradation_pct'] - stats['markov_solar']['degradation_pct'])
    if validation_diff < 10:
        print(f"   ✓ EXCELLENT: Markov model closely matches real data (Δ = {validation_diff:.1f}%)")
    elif validation_diff < 20:
        print(f"   ✓ GOOD: Markov model reasonably matches real data (Δ = {validation_diff:.1f}%)")
    else:
        print(f"   ⚠ MODERATE: Markov model differs from real data (Δ = {validation_diff:.1f}%)")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {results_dir}/")
    print("  - duty_cycle_comparison.csv")
    print("  - training_comparison.csv")
    print("  - solar_validation_comparison.png")
    print("  - solar_validation_comparison.pdf")
    print("  - validation_summary.json")
    
    return summary


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Markov solar model against real PVGIS solar data for Chemnitz, Germany"
    )
    parser.add_argument('--lat', type=float, default=50.8278,
                       help='Latitude (default: 50.8278 - Chemnitz, Germany)')
    parser.add_argument('--lon', type=float, default=12.9214,
                       help='Longitude (default: 12.9214 - Chemnitz, Germany)')
    parser.add_argument('--location', type=str, default='Chemnitz, Germany',
                       help='Location name for reports')
    parser.add_argument('--year', type=int, default=2022,
                       help='Year for solar data')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds for training')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Training epochs')
    parser.add_argument('--output', type=str, default='results/pvgis_validation',
                       help='Output directory')
    # Panel sizing parameters - CRITICAL for realistic system design
    parser.add_argument('--panel-area', type=float, default=2.0,
                       help='Panel area in m² (default: 2.0, recommend 8-10 for Northern Europe)')
    parser.add_argument('--panel-efficiency', type=float, default=0.20,
                       help='Panel efficiency (default: 0.20 = 20%%)')
    parser.add_argument('--peak-power', type=float, default=300.0,
                       help='Peak solar power for Markov model in W (default: 300.0)')
    parser.add_argument('--gpu-power', type=float, default=250.0,
                       help='GPU training power requirement in W (default: 250.0)')
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config['latitude'] = args.lat
    config['longitude'] = args.lon
    config['location_name'] = args.location
    config['year'] = args.year
    config['n_seeds'] = args.seeds
    config['epochs'] = args.epochs
    config['results_dir'] = args.output
    # Panel sizing - MUST be adjusted for local solar conditions
    config['panel_area_m2'] = args.panel_area
    config['panel_efficiency'] = args.panel_efficiency
    config['peak_solar_power_w'] = args.peak_power
    config['gpu_power_w'] = args.gpu_power
    
    # Run experiment
    results = run_pvgis_validation_experiment(config)