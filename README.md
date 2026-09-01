# Physics Structure-Informed Neural Networks in the TinyML Pipeline: Antisymmetry Preservation, Deployment Accounting, and Training-Budget Sensitivity

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI)

> **Author**: Sorin Liviu Jurj 
> **Status**: Research paper published in the [Machine Learning and Knowledge Extraction](https://www.mdpi.com/2504-4990/8/9/267) journal. 

<img width="3440" height="6652" alt="fig1_pipeline" src="https://github.com/user-attachments/assets/21493fac-72d1-4b1e-9b28-ec8309863d30" />


> **Scope:** The deployment-pipeline preservation result is experimentally established for Burgers odd-in-$x$ antisymmetry. The study does not experimentally test conservation, positivity, monotonicity, or other symmetry classes.

## 🧪 Deployment and Robustness Experiment Suite (`revision/`)

Beyond the original seven-problem intermittent-training study, the repository includes an extended
experiment suite under [`revision/`](revision/): strong compression baselines (`exp1`),
ε sensitivity (`exp2`), stochastic-interruption and lossy-checkpoint study (`exp3`),
clustering with centroid retraining (`exp4`, `distill_cluster.py`), **linked-binary
and emulated Cortex-M4/M7 deployment evaluation** (`exp5_mcu/`: C exporter, firmware,
linker scripts, QEMU harness), scaling frontier (`exp6`), simulator-size sweep (`exp7`),
robust estimator (`exp8`), illustrative solar-availability analysis (`exp9`), and **four
additional physics benchmarks** taking
the suite from 7 to 11 with a permutation test of the predictor question (`exp10`,
`pdes_extra.py`).

Key facts these experiments establish:

- **Memory must count the relation matrix.** A count of cluster centroids is not a
  measurement of memory: the relation matrix **R** costs 1 index byte per parameter,
   bounding the weight-memory compression of a clustered FP32 network near **4×**.
   The 13-cluster figure is a count of *distinct weight values*.
- **Clustering needs its centroids retrained.** Cluster-and-stop degrades the model at
  every ε (memristor at ε=0.1: test MSE 2.6e-2 vs 1.3e-6 unclustered); re-optimizing
   the K centroids with **R** frozen recovers, and can exceed, the unclustered accuracy.
- **Budget sensitivity spans 0.97×–23.7×.** On a scale-invariant paired log-ratio over
  the 11-problem suite, a halved training budget multiplies the solution error by
   between 0.97× and 23.7×. None of the four tested equation descriptors is statistically
   significant in this 11-problem dataset.
- **The deployment evidence has three levels.** Flash and statically allocated RAM are
  read from the linked firmware binary, instructions and numerical outputs are obtained
   under deterministic Cortex-M target-ISA emulation, and latency and energy are derived
   from instruction counts and stated hardware assumptions.
- **The structured Verilog-A export carries no simulator-side penalty** at any circuit
  size tested (1–256 devices); single-device wall-clock margins decay with circuit size
   and vanish by ~64 devices.

**Exploratory finding:** the three *elliptic* problems, identical under the four recorded
descriptors (elliptic, no time dependence, linear, 2nd order), span **1.33× (Laplace),
3.82× (Helmholtz), and 23.70× (Poisson)** in budget sensitivity. This within-class spread
shows that the four coarse descriptors are insufficient for this suite. Budget sensitivity
should therefore be measured when it affects a deployment decision.

The harness that produces the paper's main tables (`control_arm.py`, `full_sweep.py`,
`experiments/reproducibility.py`) is included, so the main tables are reproducible from
a clean clone.

## 📋 Overview

This repository follows physics structure through the TinyML pipeline. The central
question is not simply whether a neural surrogate is compact, but which benefits of
architecture-encoded structure survive compression and deployment, and which hardware,
training, and lifecycle effects must be measured separately.

The study connects three technical threads. First, it tests whether Burgers odd-in-$x$
antisymmetry survives clustering, quantization, compilation, and Cortex-M execution.
Second, it converts the learned representation into deployable memory, instruction,
throughput, and power quantities rather than treating parameter or centroid count as
hardware footprint. Third, it separates interruption timing from effective training
budget and checkpoint fidelity. Platform screening and lifecycle carbon are then
evaluated from these quantities under explicit system assumptions.

### Key Results

| Metric                                          | Value                                                        |
| ----------------------------------------------- | ------------------------------------------------------------ |
| **Burgers antisymmetry through deployment**     | The structured model attains 6.0% relative L2 error and an antisymmetry residual of 0.006. None of the seven evaluated baselines matches both quantities under the common protocol. |
| **Structural value at the evaluated footprint** | The structured model improves Burgers accuracy and antisymmetry, but matched structured and dense models have comparable flash and instruction requirements. Both fit the evaluated nRF52840 target. |
| **Deployable memory accounting**                | Centroid storage alone is insufficient. With one signed index byte per parameter, clustered FP32 weight-memory reduction is bounded near 4× under the stated assumptions. |
| **Deployment evidence**                         | Flash and statically allocated RAM come from the linked binary. Instructions and numerical outputs come from target-ISA emulation. Latency and energy are derived. |
| **Training-budget sensitivity**                 | A halved budget multiplies solution error by 0.97×–23.7× across eleven benchmarks, with 9 of 11 resolved. None of the four tested descriptors is statistically significant in this 11-problem dataset. |
| **Hardware-substitution carbon scenario**       | The estimated five-year lifecycle comparison is up to approximately 45× lower for the microcontroller scenario, with approximately 99.9% of the difference attributed to hardware substitution rather than model structure. |

### Graphical Result Summaries

The manuscript includes three graphical summaries that complement the numerical tables:

- **Figure 4:** Burgers relative \(L_2\) error versus antisymmetry residual across the
  structured model, seven evaluated compression baselines, and the dense reference.
- **Figure 5:** multiplicative budget-sensitivity factors across all eleven benchmark
  problems, with unresolved cases identified by confidence intervals crossing \(1\times\).
- **Figure 6:** normalized Cortex-M deployment quantities relative to Dense FP32.
  Flash and static RAM are linked-binary quantities; instruction count and firmware
  numerical outputs are obtained under target-ISA emulation. Derived latency and energy
  are intentionally not included in this summary plot.

## 🔁 Reproduce Everything (One Command)

```bash
python reproduce_paper.py
```

This single entry point regenerates **every table, figure, and headline number**
in the paper into one self-contained tree:

```
paper_artifacts/
├── tables/ Table I, III, IV, V, VI, the κ-sweep, the binding-compute example,
│ and the carbon breakdown (each as .csv AND .tex)
├── figures/ methodology pipeline, five-cell decomposition, κ-sweep, compression, budget, and deployment summaries
└── data/ all_derived_numbers.json + MANIFEST.md (artifact → paper element)
```

- **Default (archived data, ~seconds, deterministic):** every statistic is
  *recomputed* from the blessed per-seed arrays in `results/` using the documented
   paired bootstrap (mean of per-seed ratios, 10,000 resamples, 95% CI, seed=42);
   nothing is copied from the manuscript. The scenario-based carbon comparison
   (238 kg to approximately 5.35 kg, or approximately 45×) is computed from the
   documented table inputs and equations, so it is auditable rather than asserted.
- **`--retrain`:** re-runs the canonical Burgers three-regime + κ-sweep end-to-end
  (CPU backend, to match the blessed runs) and rebuilds from the fresh JSONs. The
   full seven-problem control-arm sweep that produced the archived JSONs is heavy
   (~25–80 h CPU); the remaining problems are reproduced from their archived per-seed
   data (see `paper_artifacts/data/MANIFEST.md`).

```bash
python reproduce_paper.py --retrain # regenerate Burgers from scratch
python reproduce_paper.py --outdir my_dir # choose output directory
```

> The per-problem scripts under `experiments/` remain available for inspection, but
> `reproduce_paper.py` is the recommended, clean-clone-safe route to the paper's
> artifacts.

## 🎯 Key Contributions

### 1. Burgers Antisymmetry Through the TinyML Pipeline

The structured Psi-NN encodes Burgers odd-in-$x$ antisymmetry in its architecture. The
relation remains preserved through the evaluated clustering, quantization, compilation,
and Cortex-M pipeline. Under the common Burgers protocol, none of the seven evaluated
baselines matches the structured model's joint solution accuracy and antisymmetry. The
demonstrated benefit is improved accuracy and physical consistency at the evaluated
footprint rather than automatic footprint reduction.

### 2. Centroid Count Is Not Memory Footprint

For clustered models, deployable weight memory includes both the centroids and the
relation or index representation. Under FP32 centroid storage, one signed index byte
per parameter, and the stated cluster-count limit, the weight-memory reduction
approaches but does not exceed approximately 4×. This is a weight-storage result, not
an automatic bound on total flash or RAM.

### 3. Deployment Accounting and Platform Screening

The framework estimates computational throughput, memory, and power requirements from
the model architecture and stored representation:

- **TOPS requirements**: minimum computational throughput
- **Memory footprint**: RAM and ROM requirements, including index storage
- **Power budget**: analytical average and peak estimates
- **Platform screening**: TinyML, mid-range, or high-performance tiers

> _Applicability note: for the evaluated small Burgers models, all candidate platforms
> satisfy the screening constraints. Both the dense and structured models fit the
> nRF52840. The larger scaling workload is an illustrative stress test, and at matched
> parameter count structure does not automatically reduce flash or instruction count._

### 4. Training-Budget and Checkpoint Decomposition

A paired five-cell decomposition isolates regularization (D→C), budget (C→B), and
interruption timing (B→E). Under the deterministic 50% duty cycle and complete
lossless Adam-state checkpointing, E is equivalent to continuous training at the
halved budget B. Therefore, B→E is zero by construction in this null model, not as a
general empirical claim about renewable schedules. The central empirical contrasts are
regularization and budget. `revision/exp3` separately evaluates stochastic timing,
work rollback, optimizer-state loss, and degraded checkpoints.

### 5. Exploratory Budget-Sensitivity Analysis

Measured with a scale-invariant paired log-ratio, a halved budget multiplies solution
error by between 0.97× (Burgers) and 23.7× (Poisson) across the eleven-problem suite
(9 of 11 resolved). None of the four tested descriptors, namely PDE class, temporal
coupling, nonlinearity, and derivative order, is statistically significant in this
exploratory suite. Practitioners should measure budget sensitivity when it affects a
deployment decision, while richer predictors remain an open research question.

### 6. Adaptive Regularization on Burgers

The power-responsive κ mechanism multiplies the regularization weight by
`(1 + κ·exp(-t/τ))` when an active period begins after an interruption. On Burgers,
the κ sweep (κ ∈ {0, 0.5, 1.0, 1.5, 2.0}, n=10 seeds) gives point estimates from
−8.5% [−10.3, −6.6] at κ=0 to −11.2% [−13.5, −9.0] at κ=2. This is a
Burgers-specific result and is not presented as a general benefit of intermittent
training.

### 7. Illustrative Solar Availability Check

The Markov availability model is compared with location-calibrated synthetic irradiance
for Chemnitz, Germany (50.8°N, n=3 seeds), generated using solar geometry and PVGIS-informed
weather statistics. Downstream training loss is used as a model-fidelity metric:

| Solar Panel Area (m²) | PVGIS-informed Duty Cycle | Markov Duty Cycle | PVGIS-informed Degradation | Markov Degradation | Difference  |
| --------------------- | ------------------------- | ----------------- | -------------------------- | ------------------ | ----------- |
| 2 (undersized)        | 0.3%                      | 12.9%             | +2035%                     | +109%              | Diverges    |
| 10                    | 21.7%                     | 36.3%             | +89%                       | +60%               | 29 pp       |
| 15 (target)           | 27.4%                     | 39.5%             | +68%                       | +56%               | **11.3 pp** |

At 15 m², the downstream degradation estimates differ by 11.3 percentage points, while
the duty cycles remain different. This is an indicative model-fidelity comparison,
not field validation of intermittent renewable training.

```bash
python experiments/pvgis_solar_validation.py --epochs 3000 --seeds 3
python experiments/pvgis_solar_validation.py --epochs 3000 --seeds 3 \
 --panel-area 15.0 --peak-power 1500.0 \
 --output results/pvgis_validation_15m2
```

### 8. GPU Power Measurement

The manuscript uses 250 W as a conservative RTX 4090 training-power assumption.
A separate empirical run for the stated lightweight PINN configuration measured
57 W mean power, 92 W maximum power, and 50 W minimum power. This measurement is
reported as a sensitivity check on the assumed training-power input.

```bash
python experiments/measure_gpu_power.py
python experiments/measure_gpu_power.py --manuscript
```

### 9. Statistical Evaluation

- More than 60 experiments across the original seven problems
- Ten independent random seeds per configuration
- Paired bootstrap estimator with 10,000 resamples and 95% confidence intervals
- D-normalized contrasts for the additive decomposition and C-normalized contrasts
  for the original budget-sensitivity table
- Canonical metric: test MSE against the analytical solution

## 🏗️ Repository Structure

```
├── requirements.txt # Python dependencies
├── reproduce_paper.py # ⭐ One-command reproduction (see above)
├── sustainable_edge_ai.py # Main implementation
├── generate_figure2_decomposition.py # Figure 2, descriptive-condition update
├── generate_figure3_kappa_sweep.py # Figure 3, κ/context update
├── generate_figure4_compression_summary.py # Figure 4, compression-baseline summary
├── generate_figure5_budget_sensitivity.py # Figure 5, eleven-problem budget summary
├── generate_figure6_mcu_summary.py # Figure 6, Cortex-M deployment summary
│
├── experiments/ # Individual problem experiments
│ ├── three_regime_burgers_experiment.py
│ ├── three_regime_laplace_experiment.py
│ ├── three_regime_heat_experiment.py
│ ├── three_regime_wave_experiment.py
│ ├── three_regime_advection_experiment.py
│ ├── three_regime_allen_cahn_experiment.py
│ ├── three_regime_memristor_experiment.py
│ ├── kappa_sweep_experiment.py # Figure 3 (κ-sweep on Burgers)
│ ├── duty_cycle_sweep.py
│ ├── pvgis_solar_validation.py # PVGIS-calibrated feasibility comparison
│ ├── measure_gpu_power.py # GPU power measurement
│ ├── realistic_solar_validation.py
│ ├── statistical_validation.py
│ ├── heat_wave_debug.py # Hyperparameter debug utility
│ ├── export_results.py
│ ├── methodology_pipeline.html # Figure 1 source
│ └── fig1_pipeline.svg # Figure 1 vector export
│
├── PSI-HDL-implementation/ # Base Ψ-HDL framework
│ ├── Code/
│ │ ├── structure_extractor.py # Hierarchical clustering
│ │ ├── verilog_generator.py # HDL code generation
│ │ └── vteam_baseline.py # Memristor baseline
│ └── Psi-NN-main/ # Original Ψ-NN framework
│ ├── Module/
│ │ ├── PsiNN_burgers.py
│ │ ├── PsiNN_laplace.py
│ │ ├── PsiNN_poisson.py
│ │ └── Training.py
│ └── Config/ # Experiment configurations
│
└── results/ # Experimental outputs (CPU backend, blessed)
 ├── consolidated_sweep/ # 10-seed Pass/Cont/Active test-MSE runs (Table V)
 ├── control_arm/ # D, C, B, A cells for Table VI decomposition
 ├── burgers_kappa_sweep/ # κ ∈ {0, 0.5, 1.0, 1.5, 2.0} for Figure 3
 ├── pvgis_validation/ # Markov vs. PVGIS validation (2 m² panel)
 ├── pvgis_validation_50pct_duty/ # Markov vs. PVGIS (15 m² panel, target duty cycle)
 ├── pvgis_validation_10m2_panel/ # Markov vs. PVGIS (10 m² panel)
 ├── gpu_power_measurement/ # Measured RTX 4090 power during PINN training
 ├── architecture_sensitivity/ # Architecture-width sensitivity (Burgers deep/wide)
 ├── long_term_convergence/ # 10k-epoch convergence runs (Burgers, Laplace)
 └── statistical_validation/ # Per-PDE statistical validation outputs

paper_artifacts/ # Generated by reproduce_paper.py (git-ignorable)
├── tables/ # Table I/III/IV/V/VI, κ-sweep, carbon (.csv + .tex)
├── figures/ # Figures 1–6: pipeline, decomposition, κ-sweep, compression, budget, deployment
└── data/ # all_derived_numbers.json + MANIFEST.md
```

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended: Zero Setup)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI)

What's included:

- Python 3.11 with all dependencies pre-installed
- Jupyter Notebook support
- VS Code extensions for Python development
- Ready-to-run experiments

```bash
# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Run your first experiment (single PDE, three regimes)
python experiments/three_regime_burgers_experiment.py
```

### Option 2: Local Installation

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU optional (CPU backend is the canonical one: see Reproducibility)
nvidia-smi
```

```bash
git clone https://github.com/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI.git
cd Psi-NNs-for-Sustainable-Edge-AI
pip install -r requirements.txt
pip install -r PSI-HDL-implementation/requirements.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Running Experiments

> **To reproduce the paper's tables and figures, use [`reproduce_paper.py`](#-reproduce-everything-one-command).**
> The per-problem scripts below are provided for inspection and for training models
> from scratch. They assume the author's working-tree layout; if an import path fails
> on a fresh clone, prefer `reproduce_paper.py --retrain`, which is clean-clone-safe.

#### 1. Individual Problem Classes

```bash
# Three-regime comparison (Burgers PDE): primary example
python experiments/three_regime_burgers_experiment.py

# Same protocol for the other six problems:
python experiments/three_regime_laplace_experiment.py
python experiments/three_regime_heat_experiment.py
python experiments/three_regime_wave_experiment.py
python experiments/three_regime_advection_experiment.py
python experiments/three_regime_allen_cahn_experiment.py
python experiments/three_regime_memristor_experiment.py

# Statistical validation with 10 seeds
python experiments/statistical_validation.py

# κ-sweep analysis (κ = 0.0 to 2.0)
python experiments/kappa_sweep_experiment.py

# Realistic weather-dependent solar patterns (PVGIS-calibrated)
python experiments/pvgis_solar_validation.py --panel-area 15.0 --peak-power 1500.0
```

#### 2. Generate Paper Figures

```bash
# Figure 2: five-cell decomposition with descriptive conditions
python generate_figure2_decomposition.py

# Figure 3: Burgers κ-sweep with clarified κ and performance direction
python generate_figure3_kappa_sweep.py

# Figure 4: compression-baseline accuracy and antisymmetry summary
python generate_figure4_compression_summary.py

# Figure 5: eleven-problem budget-sensitivity summary
python generate_figure5_budget_sensitivity.py

# Figure 6: normalized Cortex-M deployment summary
python generate_figure6_mcu_summary.py
```

Figure 1 is built from `experiments/methodology_pipeline.html` and exported as
`experiments/fig1_pipeline.svg`. It presents structured-model extraction, deployment
accounting and platform screening, and controlled training-budget/checkpoint analysis as
separate analytical tracks. The figure also states that the small Burgers screening
constraints are non-binding and that the solar availability trace is illustrative.

## 📊 Core Modules

### 1. Controlled Intermittent-Power Training

```python
from sustainable_edge_ai import SolarConstrainedTrainer

trainer = SolarConstrainedTrainer(model, config={
 'duty_cycle': 0.5, # 50% solar availability
 'active_period': 10, # 10 steps on
 'idle_period': 10, # 10 steps off
 'checkpoint_frequency': 100, # Save every 100 steps
 'kappa': 2.0, # κ-mechanism amplification (0.0 = passive)
})

for epoch in range(num_epochs):
 loss = trainer.train_step(loss_fn=compute_loss, optimizer=optimizer)
 if trainer.should_checkpoint():
 trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
```

### 2. Hardware Specification Extraction

```python
from sustainable_edge_ai import HardwareSpecificationExtractor
from structure_extractor import StructureExtractor

struct_extractor = StructureExtractor(model, model_type="PsiNN_burgers")
hw_extractor = HardwareSpecificationExtractor(model, struct_extractor)

specs = {
 'operations': hw_extractor.compute_operations(),
 'tops': hw_extractor.compute_tops_requirement(target_fps=30.0),
 'memory_kb': hw_extractor.compute_memory_requirements() / 1024,
 'power_mw': hw_extractor.estimate_power_consumption() * 1000,
}
print(f"TOPS Required: {specs['tops']:.6f}")
print(f"Memory: {specs['memory_kb']:.2f} KB")
print(f"Power: {specs['power_mw']:.2f} mW")
```

### 3. Platform Screening

```python
from sustainable_edge_ai import EdgeAIPlatformRecommender

recommender = EdgeAIPlatformRecommender()
platforms = recommender.recommend_platform(
 requirements=specs,
 constraints={'max_cost_usd': 100, 'max_power_mw': 10000},
)
for i, platform in enumerate(platforms[:3], 1):
 print(f"{i}. {platform['name']}: "
 f"${platform['cost']:.2f}, "
 f"{platform['utilization']*100:.1f}% utilization, "
 f"Fit: {platform['fit_category']}")
```

### 4. Scenario-Based Carbon Accounting

```python
from sustainable_edge_ai import CarbonFootprintAnalyzer

analyzer = CarbonFootprintAnalyzer()
carbon = analyzer.compute_lifecycle_carbon(
 platform=platforms[0],
 deployment_years=5.0,
 training_regime='solar', # vs 'grid'
 duty_cycle=0.5,
)
print(f"Training Carbon: {carbon['training_kg_co2']:.3f} kg CO₂")
print(f"Deployment Carbon: {carbon['deployment_kg_co2']:.1f} kg CO₂")
print(f"Total Lifecycle: {carbon['total_kg_co2']:.1f} kg CO₂")
```

## 🔬 Experimental Results

### Table V: Passive vs. Continuous (Own-Denominator, test MSE)

Test MSE change at the 50% duty cycle (κ=0, passive) relative to continuous
training at full budget. **Paired bootstrap, n=10 seeds, 95% CI, 10,000 resamples.**

| Problem     | PDE Type                     | Pass-Cont test MSE      | Status     |
| ----------- | ---------------------------- | ----------------------- | ---------- |
| **Burgers** | Parabolic (nonlinear)        | **−8.5% [−10.3, −6.6]** | improves   |
| Laplace     | Elliptic (steady-state)      | +7.5% [−7, +20]         | unresolved |
| Allen-Cahn  | Nonlinear reaction-diffusion | +121.2% [+72, +169]     | resolved   |
| Heat        | Parabolic                    | +155.9% [+98, +215]     | resolved   |
| Wave        | Hyperbolic (2nd-order)       | +691.3% [+193, +1462]   | wide CI    |
| Memristor   | ODE (device physics)         | +768.8% [+419, +1149]   | resolved   |
| Advection   | Hyperbolic (1st-order)       | +861.5% [+293, +1615]   | resolved   |

### Table IV: Budget Sensitivity C→B (C-normalized)

The cost of halving the training budget at matched regularization. **5 of 7 resolved at 95% CI.**

| Problem    | C→B Point Estimate | 95% CI        | Status     |
| ---------- | ------------------ | ------------- | ---------- |
| Burgers    | **−3.2%**          | [−4, −2]      | resolved   |
| Laplace    | +41.5%             | [+12, +79]    | resolved   |
| Allen-Cahn | +97.6%             | [+67, +129]   | resolved   |
| Heat       | +176.8%            | [+85, +305]   | resolved   |
| Memristor  | +306.4%            | [−69, +714]   | unresolved |
| Advection  | **+620.0%**        | [+239, +1118] | resolved   |
| Wave       | +3474%             | [+30, +9677]  | wide CI    |

**Exploratory finding**: on the scale-invariant paired log-ratio over the full
eleven-problem suite, a halved budget multiplies solution error by **0.97×–23.7×**
(the table above reports the C-normalized percentage contrasts for the original seven
problems). None of the four tested descriptors is statistically significant in this
suite (permutation test: all p > 0.18). The three elliptic problems alone span
1.33×–23.7×, showing that the recorded coarse descriptors are insufficient here.
The Memristor's sign reversal between C-normalization (+306.4%) and D-normalization
(−767%) indicates that the estimator is at its lower sample-size bound for that
problem. C→B should be measured when it affects a deployment decision.

### Figure 3: Burgers PDE κ-Sweep (Weak-Monotone Improvement)

The κ-mechanism produces a weak-monotone improvement curve relative to the
continuous baseline. **All five points individually resolved at 95% CI.**

| κ Value | test MSE Change vs. Continuous | 95% CI        |
| ------- | ------------------------------ | ------------- |
| 0.0     | −8.5%                          | [−10.3, −6.6] |
| 0.5     | −9.1%                          | [−10.9, −7.3] |
| 1.0     | −9.7%                          | [−11.9, −7.5] |
| 1.5     | −10.3%                         | [−12.3, −8.4] |
| 2.0     | **−11.2%**                     | [−13.5, −9.0] |

Endpoint span (κ=0 to κ=2): **−2.7% [−3.3, −2.2]**: also resolved.
Cross-validation against independent passive-to-active comparison: agrees at 0.00 pp.

### Platform Screening Example (Burgers PDE)

| Platform            | TOPS  | Cost  | Power  | Utilization | Fit                                        | Score  |
| ------------------- | ----- | ----- | ------ | ----------- | ------------------------------------------ | ------ |
| STM32H7             | 0.082 | $8    | 400 mW | 0.027%      | Over-specified                             | 214    |
| **Nordic nRF52840** | 0.026 | $5.00 | 15 mW  | 0.085%      | **Over-specified (lowest-power feasible)** | 252    |
| TI AM62A            | 2.0   | $35   | 2 W    | <0.001%     | Over-specified                             | 17,500 |

### Scenario-Based Carbon Comparison (5-year lifecycle, per device)

| Scenario                                         | Training | Deployment | Total        | Reduction      |
| ------------------------------------------------ | -------- | ---------- | ------------ | -------------- |
| **Microcontroller scenario (solar electricity)** | 0.036 kg | 0.31 kg    | **~5.35 kg** | **~45× lower** |
| GPU-class baseline (grid electricity)            | 0.356 kg | 238 kg     | **238 kg**   | baseline       |

**Estimated per-device difference: approximately 233 kg CO₂. Approximately 99.9%
of this difference comes from hardware substitution (Jetson to nRF52840); the assumed
electricity-source change contributes less than 1% (approximately 0.32 kg).**

Under the same scenario assumptions, the estimated difference is approximately
2,330 metric tons CO₂ for 10,000 devices and 233,000 metric tons CO₂ for one million
devices.

## 📈 Reproducing Results

> [`reproduce_paper.py`](#-reproduce-everything-one-command) implements exactly the
> protocol documented in this section and regenerates Tables III–V, the κ-sweep, and
> the carbon breakdown from the archived per-seed data in seconds. The estimator,
> normalization choices, and runtimes below describe that same protocol; the runtime
> table applies to **training from scratch** (`--retrain` / per-problem scripts),
> not to the default archived-data rebuild.

### Statistical Validation Protocol

All performance contrasts use the same paired bootstrap estimator:

```python
import numpy as np

def paired_bootstrap_ci(num_arr, denom_arr, n_boot=10000, seed=42):
 """Mean of per-seed ratios, 95% percentile CI (10k resamples)."""
 num = np.array(num_arr); den = np.array(denom_arr)
 ratios = (num - den) / den
 point_est = ratios.mean()
 rng = np.random.default_rng(seed)
 idx = rng.integers(0, len(ratios), size=(n_boot, len(ratios)))
 boot = ratios[idx].mean(axis=1)
 lo, hi = np.percentile(boot, [2.5, 97.5])
 return point_est * 100, lo * 100, hi * 100 # return as percentages
```

### Normalization choices (per table)

- **Table V**: own-denominator (each contrast normalized to its own reference)
- **Table VI**: D-normalized (additive closure D→C + C→B + B→E = D→E, residual ≤ 4×10⁻¹⁴)
- **Table IV**: C-normalized (budget sensitivity relative to full-budget at high reg)
- **κ-sweep (Figure 3)**: D-normalized (improvement relative to continuous baseline)

### Expected Runtime

| Experiment               | Seeds            | Epochs | Wall Clock (CPU) |
| ------------------------ | ---------------- | ------ | ---------------- |
| Single problem           | 1                | 3000   | ~20 minutes      |
| Statistical validation   | 10               | 3000   | ~3.5 hours       |
| Full 7-problem sweep     | 10 × 7           | 3000   | ~25 hours        |
| Decomposition (Table VI) | 10 × 4 cells × 7 | 3000   | ~80 hours        |
| κ-sweep (Figure 3)       | 10 × 5           | 3000   | ~14 hours        |

**Note**: Solar-constrained training extends wall-clock time by ~2× due to the
50% duty cycle. All blessed results in the paper are CPU-backend runs to ensure
bit-exact reproducibility across machines.

## 🎓 Citation

```bibtex
@article{jurj2026psinn_tinyml,
  author  = {Sorin Liviu Jurj},
  title   = {Physics Structure-Informed Neural Networks in the TinyML Pipeline: Antisymmetry Preservation, Deployment Accounting, and Training-Budget Sensitivity},
  journal = {Machine Learning and Knowledge Extraction},
  year    = {2026},
  volume  = {8},
  article-number = {267},
  doi     = {10.3390/make8090267},
  url     = {https://doi.org/10.3390/make8090267}
}
```

## 📚 Related Publications

1. **Ψ-HDL Framework**: [PSI-HDL GitHub](https://github.com/jurjsorinliviu/PSI-HDL)
2. **Original Ψ-NN**: [Psi-NN GitHub](https://github.com/ZitiLiu/Psi-NN)
3. **Ψ-xLSTM**: [Psi-xLSTM GitHub](https://github.com/jurjsorinliviu/Psi-xLSTM)

## 🔍 Key Findings Summary

### ✅ What is established

1. **Architecture-encoded Burgers antisymmetry survives the evaluated TinyML pipeline.**
   Under the common Burgers protocol, the structured model reaches 6.0% relative L2
   error with an antisymmetry residual of 0.006, and none of the seven evaluated
   baselines matches both quantities.

2. **Centroid count is not deployable memory.**
   The relation or index representation must be stored together with the centroids.
   Under one signed index byte per parameter, clustered FP32 weight-memory reduction
   approaches but does not exceed approximately 4× under the stated assumptions.

3. **Structure improves the learned solution without automatically shrinking the binary.**
   At matched parameter count, flash and instruction counts are similar for the dense
   and structured families. Both small Burgers models fit the nRF52840. The demonstrated
   structural advantage is better Burgers accuracy and antisymmetry at the evaluated
   footprint.

4. **Complete lossless checkpointing removes interruption timing as an independent effect.**
   In the null model, B→E is zero because the complete optimization trajectory is
   restored. Reduced training budget, rollback, optimizer-state loss, and degraded
   checkpoints introduce the effects that matter experimentally.

5. **The deployment evidence is separated by measurement level.**
   Flash and statically allocated RAM are obtained from the linked binary. Instructions
   and numerical outputs are obtained under deterministic target-ISA emulation. Latency
   and energy are derived rather than measured on physical hardware.

6. **The lifecycle scenario is hardware-substitution driven.**
   Approximately 99.9% of the estimated 233 kg per-device difference comes from the
   Jetson-to-nRF52840 hardware change. It is not attributed to model structure alone.

### Scope and interpretation

7. **Budget sensitivity spans 0.97×–23.7× in the eleven-problem dataset.**
   Nine of eleven cases are resolved under the scale-invariant paired log-ratio. None of
   the four tested descriptors reaches statistical significance in this dataset. Richer
   predictors remain an open research question.

8. **The Burgers κ result is problem-specific.**
   The tested κ mechanism gives point estimates from −8.5% to −11.2% relative to the
   continuous baseline. This behavior is not generalized to other equations.

9. **Wave and Memristor remain unresolved under the robust estimator.**
   Wave has a wide confidence interval, and the Memristor interval crosses zero. Both
   are excluded from the resolved-nine-of-eleven summary.

### 🔬 Methodological checks

10. The additive decomposition D→C + C→B + B→E = D→E closes to a numerical residual
    of at most 4×10⁻¹⁴ where evaluated.

11. The κ=2 endpoint agrees with the independent passive-to-active comparison to
    0.00 percentage points in the Burgers experiment.

## 🛠️ Hardware Platforms Database

| Tier          | Platform         | TOPS  | Memory  | Power  | Cost  | Technology       |
| ------------- | ---------------- | ----- | ------- | ------ | ----- | ---------------- |
| **TinyML**    | Nordic nRF52840  | 0.026 | 256 KB  | 15 mW  | $5.00 | Cortex-M4        |
| **TinyML**    | STM32H7          | 0.082 | 1024 KB | 400 mW | $8    | Cortex-M7        |
| **Mid-Range** | TI AM62A         | 2.0   | 2 MB    | 2 W    | $35   | Cortex-A53+CNN   |
| **Mid-Range** | TI TDA4VM        | 8.0   | 8 MB    | 4.5 W  | $80   | Cortex-A72+DSP   |
| **High-Perf** | Hailo-8          | 26.0  | 4 MB    | 5 W    | $150  | Neural Processor |
| **High-Perf** | Jetson Orin Nano | 40.0  | 8 MB    | 10 W   | $249  | Ampere GPU       |

## 🌍 Scenario-Based Environmental Impact

### Single Device (5-year lifecycle)

- **GPU-class baseline scenario**: grid electricity plus Jetson Orin Nano, approximately **238 kg CO₂**
- **Microcontroller substitution scenario**: solar electricity plus Nordic nRF52840, approximately **5.35 kg CO₂**
- **Estimated difference**: approximately 233 kg CO₂ per device, or approximately 45× lower
- **Attribution**: approximately 99.9% from hardware substitution and less than 1% from the assumed electricity-source change

### At Scale

| Deployment     | GPU-class baseline | Microcontroller scenario | Estimated difference | Illustrative equivalent                              |
| -------------- | ------------------ | ------------------------ | -------------------- | ---------------------------------------------------- |
| 1,000 devices  | 238 tons           | 5.35 tons                | 233 tons             | approximately 51 cars for one year                   |
| 10,000 devices | 2,380 tons         | 53.5 tons                | 2,330 tons           | approximately 507 cars or 2,774 acres of forest      |
| 1M devices     | 238,000 tons       | 5,350 tons               | 233,000 tons         | approximately 50,700 cars or 277,000 acres of forest |

*The equivalencies use the stated EPA averages of 4.6 t CO₂ per car per year and
0.84 t CO₂ per acre per year of forest sequestration. They are illustrative
conversions of the scenario-based estimates.*

## 🤝 Contributing

Areas of interest:

- [ ] Additional PDE families (biharmonic, higher-order dispersive, systems of PDEs) to extend the budget-sensitivity map beyond the eleven benchmarks
- [ ] Extended platform database (Qualcomm, Google Coral, Intel Movidius, RISC-V, neuromorphic)
- [ ] Multi-physics coupled problems (thermoelasticity, MHD) to test whether physical coupling amplifies budget sensitivity
- [ ] Real hardware deployment validation with field solar measurements
- [ ] Wind + solar hybrid renewable strategies
- [ ] Reducing budget sensitivity for transport-dominated problems (Advection)

## 📞 Contact

**Sorin Liviu Jurj** 
Email: jurjsorinliviu@yahoo.de 
GitHub: [@jurjsorinliviu](https://github.com/jurjsorinliviu)

## 📄 License

Apache License 2.0: see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Ψ-HDL Framework: [Psi-HDL GitHub](https://github.com/jurjsorinliviu/Psi-HDL)
- Original Ψ-NN: [Psi-NN GitHub](https://github.com/ZitiLiu/Psi-NN)
- Ψ-xLSTM: [Psi-xLSTM GitHub](https://github.com/jurjsorinliviu/Psi-xLSTM)

---

**Last Updated**: August 2026 
**Paper Status**: Under review 
**Code Version**: v1.0
