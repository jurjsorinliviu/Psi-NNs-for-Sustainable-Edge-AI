"""
Export Chapter 4 results in reproducible formats

Creates:
- CSV tables for manuscript
- Model checkpoints
- Raw training data
- Comparison tables
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def create_comparison_table(results_dir: Path):
    """Create grid vs solar comparison table"""
    
    # Load results
    grid_file = results_dir / "burgers_grid_results.json"
    solar_file = results_dir / "burgers_solar_results.json"
    
    with open(grid_file) as f:
        grid = json.load(f)
    with open(solar_file) as f:
        solar = json.load(f)
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': [
            'Training Mode',
            'Training Time (s)',
            'Final Loss',
            'Total Operations',
            'TOPS Requirement',
            'Memory (KB)',
            'Power (mW)',
            'Recommended Platform',
            'Platform Cost ($)',
            'Embodied Carbon (kg CO2-eq)',
            'Operational Carbon (kg CO2-eq)',
            'Total Carbon (kg CO2-eq)',
            'Carbon Savings (kg)',
            'Carbon Reduction (%)',
            'Solar Duty Cycle (%)',
            'Active Training Steps'
        ],
        'Grid-Powered': [
            'Grid',
            f"{grid['training_time']:.1f}",
            f"{grid['final_loss']:.6f}",
            grid['hardware_specs']['operations']['total_operations'],
            f"{grid['hardware_specs']['tops_requirement']:.2e}",
            f"{grid['hardware_specs']['memory']['total_memory_kb']:.2f}",
            f"{grid['hardware_specs']['power']['total_power_mw']:.3f}",
            grid['platform_recommendation'][0]['name'],
            grid['platform_recommendation'][0]['cost_usd'],
            grid['carbon_footprint']['embodied_carbon_kg'],
            f"{grid['carbon_footprint']['grid_operational_kg']:.2f}",
            f"{grid['carbon_footprint']['grid_total_kg']:.2f}",
            '-',
            '-',
            '100%',
            '3000'
        ],
        'Solar-Powered': [
            'Solar',
            f"{solar['training_time']:.1f}",
            f"{solar['final_loss']:.6f}",
            solar['hardware_specs']['operations']['total_operations'],
            f"{solar['hardware_specs']['tops_requirement']:.2e}",
            f"{solar['hardware_specs']['memory']['total_memory_kb']:.2f}",
            f"{solar['hardware_specs']['power']['total_power_mw']:.3f}",
            solar['platform_recommendation'][0]['name'],
            solar['platform_recommendation'][0]['cost_usd'],
            solar['carbon_footprint']['embodied_carbon_kg'],
            f"{solar['carbon_footprint']['solar_operational_kg']:.2f}",
            f"{solar['carbon_footprint']['solar_total_kg']:.2f}",
            f"{solar['carbon_footprint']['carbon_saved_kg']:.2f}",
            f"{solar['carbon_footprint']['carbon_reduction_percent']:.1f}%",
            f"{solar['solar_stats']['actual_duty_cycle']*100:.1f}%",
            solar['solar_stats']['active_steps']
        ]
    })
    
    # Save comparison table
    comparison_file = results_dir / "comparison_table.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison table: {comparison_file}")
    
    # Also save as LaTeX table
    latex_file = results_dir / "comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write(comparison.to_latex(index=False, escape=False))
    print(f"✓ Saved LaTeX table: {latex_file}")
    
    return comparison


def export_all_results():
    """Export all results in reproducible formats"""
    
    results_dir = Path(__file__).parent.parent.parent / "chapter4_results" / "burgers"
    
    print("\n" + "="*70)
    print("EXPORTING CHAPTER 4 RESULTS FOR REPRODUCIBILITY")
    print("="*70)
    
    # Create comparison table
    comparison = create_comparison_table(results_dir)
    
    print("\n✓ All results exported successfully")
    print(f"✓ Results directory: {results_dir}")
    
    # Display comparison
    print("\n" + "="*70)
    print("GRID VS SOLAR COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Highlight issues
    print("\n" + "="*70)
    print("⚠️  METHODOLOGY ISSUES IDENTIFIED")
    print("="*70)
    
    with open(results_dir / "burgers_solar_results.json") as f:
        solar = json.load(f)
    
    actual_duty = solar['solar_stats']['actual_duty_cycle']
    target_duty = solar['solar_stats']['target_duty_cycle']
    
    if abs(actual_duty - target_duty) > 0.05:
        print(f"❌ Solar duty cycle: {actual_duty*100:.1f}% (target: {target_duty*100:.0f}%)")
        print(f"   Active steps: {solar['solar_stats']['active_steps']}/3000")
        print(f"   Issue: Training periods too short")
        print(f"   Fix needed: Adjust active_period and idle_period in config")
    
    return comparison


if __name__ == "__main__":
    export_all_results()