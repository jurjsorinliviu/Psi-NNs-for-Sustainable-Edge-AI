"""
Statistical Validation Framework
==================================

Runs experiments multiple times with different random seeds to provide:
- Mean and standard deviation of results
- Confidence intervals
- Statistical significance testing
- Publication-quality error bars

Author: Sorin Liviu Jurj
Date: 2025-11-15
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Callable, Any
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))


def run_experiment_multiple_times(
    experiment_fn: Callable,
    num_runs: int = 10,
    seeds: List[int] = None,
    experiment_name: str = "experiment",
    results_dir: str = "chapter4/results/statistical_validation"
) -> Dict[str, Any]:
    """
    Run an experiment multiple times with different random seeds
    
    Args:
        experiment_fn: Function that runs single experiment, returns dict of results
        num_runs: Number of times to run experiment
        seeds: List of random seeds (auto-generated if None)
        experiment_name: Name for results files
        results_dir: Directory to save results
        
    Returns:
        Dictionary with aggregated statistics
    """
    
    if seeds is None:
        # Generate reproducible seeds
        np.random.seed(42)
        seeds = np.random.randint(0, 10000, num_runs).tolist()
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL VALIDATION: {experiment_name}")
    print(f"{'='*80}")
    print(f"Running {num_runs} independent trials with seeds: {seeds}")
    print()
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Store all runs
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"RUN {i+1}/{num_runs} (seed={seed})")
        print(f"{'='*80}")
        
        # Run experiment with this seed
        results = experiment_fn(seed=seed)
        results['seed'] = seed
        results['run_number'] = i + 1
        all_results.append(results)
        
        print(f"\nRun {i+1} Complete: Loss={results.get('final_loss', 'N/A')}")
    
    # Compute statistics
    statistics = compute_statistics(all_results, experiment_name)
    
    # Save results
    save_path = Path(results_dir) / f"{experiment_name}_statistical_validation.json"
    with open(save_path, 'w') as f:
        json.dump({
            'num_runs': num_runs,
            'seeds': seeds,
            'individual_results': all_results,
            'statistics': statistics
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {save_path}")
    
    return statistics


def compute_statistics(results: List[Dict], experiment_name: str) -> Dict[str, Any]:
    """
    Compute statistical measures from multiple runs
    
    Args:
        results: List of result dictionaries from individual runs
        experiment_name: Name of experiment
        
    Returns:
        Dictionary with statistical measures
    """
    
    # Extract metrics from each run
    metrics = {}
    for key in results[0].keys():
        if key in ['seed', 'run_number']:
            continue
        
        values = [r[key] for r in results if key in r]
        
        # Skip if not numeric
        if not all(isinstance(v, (int, float)) for v in values):
            continue
        
        values = np.array(values)
        
        # Compute statistics
        metrics[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'sem': float(stats.sem(values)),  # Standard error of mean
            'ci_95': {
                'lower': float(np.mean(values) - 1.96 * stats.sem(values)),
                'upper': float(np.mean(values) + 1.96 * stats.sem(values))
            }
        }
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL SUMMARY: {experiment_name}")
    print(f"{'='*80}")
    
    for metric_name, stats_dict in metrics.items():
        print(f"\n{metric_name}:")
        print(f"  Mean ± Std:  {stats_dict['mean']:.6f} ± {stats_dict['std']:.6f}")
        print(f"  95% CI:      [{stats_dict['ci_95']['lower']:.6f}, {stats_dict['ci_95']['upper']:.6f}]")
        print(f"  Range:       [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]")
        print(f"  Median:      {stats_dict['median']:.6f}")
    
    return metrics


def compare_regimes_statistical(
    continuous_results: Dict,
    passive_results: Dict,
    active_results: Dict,
    metric: str = 'final_loss',
    results_dir: str = "chapter4/results/statistical_validation"
) -> Dict[str, Any]:
    """
    Compare three training regimes with statistical significance testing
    
    Args:
        continuous_results: Statistics from continuous regime
        passive_results: Statistics from passive regime
        active_results: Statistics from active regime
        metric: Which metric to compare
        results_dir: Where to save results
        
    Returns:
        Dictionary with comparison results and p-values
    """
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON OF TRAINING REGIMES")
    print(f"{'='*80}")
    print(f"Metric: {metric}")
    print()
    
    # Get values from each regime
    continuous_vals = [r[metric] for r in continuous_results['individual_results'] if metric in r]
    passive_vals = [r[metric] for r in passive_results['individual_results'] if metric in r]
    active_vals = [r[metric] for r in active_results['individual_results'] if metric in r]
    
    continuous_vals = np.array(continuous_vals)
    passive_vals = np.array(passive_vals)
    active_vals = np.array(active_vals)
    
    # Perform statistical tests
    # 1. Paired t-test (if same seeds used)
    t_stat_passive, p_value_passive = stats.ttest_rel(continuous_vals, passive_vals)
    t_stat_active, p_value_active = stats.ttest_rel(continuous_vals, active_vals)
    t_stat_passive_active, p_value_passive_active = stats.ttest_rel(passive_vals, active_vals)
    
    # 2. Effect size (Cohen's d)
    def cohens_d(group1, group2):
        pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    effect_passive = cohens_d(continuous_vals, passive_vals)
    effect_active = cohens_d(continuous_vals, active_vals)
    effect_passive_active = cohens_d(passive_vals, active_vals)
    
    comparison = {
        'continuous_vs_passive': {
            't_statistic': float(t_stat_passive),
            'p_value': float(p_value_passive),
            'cohens_d': float(effect_passive),
            'significant': p_value_passive < 0.05,
            'mean_difference': float(np.mean(passive_vals) - np.mean(continuous_vals)),
            'percent_change': float((np.mean(passive_vals) / np.mean(continuous_vals) - 1) * 100)
        },
        'continuous_vs_active': {
            't_statistic': float(t_stat_active),
            'p_value': float(p_value_active),
            'cohens_d': float(effect_active),
            'significant': p_value_active < 0.05,
            'mean_difference': float(np.mean(active_vals) - np.mean(continuous_vals)),
            'percent_change': float((np.mean(active_vals) / np.mean(continuous_vals) - 1) * 100)
        },
        'passive_vs_active': {
            't_statistic': float(t_stat_passive_active),
            'p_value': float(p_value_passive_active),
            'cohens_d': float(effect_passive_active),
            'significant': p_value_passive_active < 0.05,
            'mean_difference': float(np.mean(active_vals) - np.mean(passive_vals)),
            'percent_change': float((np.mean(active_vals) / np.mean(passive_vals) - 1) * 100)
        }
    }
    
    # Print results
    print("\nContinuous vs Passive:")
    print(f"  p-value: {comparison['continuous_vs_passive']['p_value']:.4f} "
          f"({'significant' if comparison['continuous_vs_passive']['significant'] else 'not significant'})")
    print(f"  Cohen's d: {comparison['continuous_vs_passive']['cohens_d']:.3f}")
    print(f"  Mean difference: {comparison['continuous_vs_passive']['percent_change']:+.2f}%")
    
    print("\nContinuous vs Active:")
    print(f"  p-value: {comparison['continuous_vs_active']['p_value']:.4f} "
          f"({'significant' if comparison['continuous_vs_active']['significant'] else 'not significant'})")
    print(f"  Cohen's d: {comparison['continuous_vs_active']['cohens_d']:.3f}")
    print(f"  Mean difference: {comparison['continuous_vs_active']['percent_change']:+.2f}%")
    
    print("\nPassive vs Active:")
    print(f"  p-value: {comparison['passive_vs_active']['p_value']:.4f} "
          f"({'significant' if comparison['passive_vs_active']['significant'] else 'not significant'})")
    print(f"  Cohen's d: {comparison['passive_vs_active']['cohens_d']:.3f}")
    print(f"  Mean difference: {comparison['passive_vs_active']['percent_change']:+.2f}%")
    
    # Create comparison plot with error bars
    create_comparison_plot_with_errors(
        continuous_results, passive_results, active_results,
        metric, results_dir
    )
    
    # Save comparison
    save_path = Path(results_dir) / f"statistical_comparison_{metric}.json"
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def create_comparison_plot_with_errors(
    continuous_results: Dict,
    passive_results: Dict,
    active_results: Dict,
    metric: str,
    results_dir: str
):
    """Create bar plot with error bars showing statistical comparison"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data
    regimes = ['Continuous', 'Passive', 'Active']
    results_list = [continuous_results, passive_results, active_results]
    
    means = [r['statistics'][metric]['mean'] for r in results_list]
    stds = [r['statistics'][metric]['std'] for r in results_list]
    sems = [r['statistics'][metric]['sem'] for r in results_list]
    
    # Create bars
    x = np.arange(len(regimes))
    bars = ax.bar(x, means, yerr=sems, capsize=5, 
                   color=['gray', 'green', 'orange'], alpha=0.7,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Customize
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison with 95% CI')
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.6f}\n±{std:.6f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f'statistical_comparison_{metric}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {results_dir}/statistical_comparison_{metric}.png")


if __name__ == '__main__':
    print("Statistical Validation Framework")
    print("=" * 80)
    print("This module provides tools for running experiments multiple times")
    print("and computing statistical measures with confidence intervals.")
    print()
    print("Usage:")
    print("  from statistical_validation import run_experiment_multiple_times")
    print("  results = run_experiment_multiple_times(my_experiment_fn, num_runs=10)")