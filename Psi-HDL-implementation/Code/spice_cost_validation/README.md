# Simulator-Cost Validation (dense vs. structured Verilog-A)

Bounded experiment for the manuscript subsection *"Simulator-Cost Comparison:
Dense vs. Structured Verilog-A Export"* (Table VIII). It tests **simulator-side**
cost of structure-aware compact models — **not** embedded hardware deployment.

## What it does
One small neural memristor read-characteristic model `I(V)` is trained and
exported to Verilog-A two ways, both implementing the same learned behavior:

| Export | Architecture | Weight terms | Error vs. analytic ref |
| --- | --- | --- | --- |
| Dense (over-parameterized) | `[1,16,16,1]` | 288 | 0.11% |
| Structured (discovered-scale) | `[1,4,4,1]` | 24 | 0.39% |

They agree with each other to ≤0.9% RMS. Each is compiled with **OpenVAF** to an
OSDI module and run in **ngspice** under DC-read, pulse-transient, and sinusoidal
testbenches.

## Result (min of 5 runs)
Both exports compile, both converge, identical accepted output points; the
structured export is consistently at least as fast (≈2% DC, ≈4% pulse, ≈10%
sinusoid), i.e. structure carries no simulator-side penalty and a small benefit
on top of its parameter/memory reduction.

## Reproduce
Requires `ngspice` on PATH and OpenVAF (`openvaf.exe`). From a working dir
containing `openvaf/openvaf.exe`:

```bash
python generate_dense_vs_structured.py        # writes mem_dense.va, mem_struct.va
./openvaf/openvaf.exe mem_dense.va             # -> mem_dense.osdi
./openvaf/openvaf.exe mem_struct.va            # -> mem_struct.osdi
python run_simulator_cost.py                   # runs the 3 testbenches, prints the table
```

`run_simulator_cost.py` uses `C:/ngspice_work` as its working directory (where the
`.osdi` modules live); adjust the `WD` constant for other locations.

## Related export-backend fixes
- **Device export** (`../output/memristor/`): `memristor_pinn.va` plus
  `memristor_dc_read_osdi.cir` / `memristor_pulse_osdi.cir`. The fix renamed
  working variables that shadowed the Verilog-A `V()`/`I()` access functions.
- **Neural PDE-surrogate export** (`../verilog_generator.py`): made
  OpenVAF-compatible by replacing multidimensional `parameter real W[..][..]`
  arrays and array-valued working variables with scalar parameters and explicit
  forward-pass expressions. The regenerated Burgers/Laplace exports
  (`../output/{burgers,laplace}/psi_nn_PsiNN_*.va`) compile with OpenVAF and load
  in ngspice (`*_osdi_sanity.cir`).
