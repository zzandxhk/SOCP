# GA + ADMM Planning Implementation

`planning.py` is an **independent implementation** based on the structure described in the revised manuscript. A genetic algorithm (GA) searches the capacity choices for 32 line upgrades and three soft open points (SOPs). Consensus ADMM with Laplace noise coordinates hourly schedules for three microgrids. For each candidate plan, an IEEE 33-bus branch-flow SOCP checks voltages and line capacities and calculates annualized investment and upstream-grid purchase costs.

This is **not the original program** that produced the manuscript's Case I–III tables. The original 8760-hour inputs, line lengths and thermal limits, complete equipment costs, and run-by-run records were not provided. Missing inputs in `demo_inputs.json` are explicitly marked synthetic demonstration values. Costs computed from that file **must not be reported as the manuscript's original results**.

## Run

Install `numpy`, `cvxpy`, and `clarabel`, then run from this directory:

```powershell
python planning.py --demo --quick --output smoke_result.json
```

`--quick` uses one representative scenario, the first two hours, four individuals, and two generations to check the workflow. Its output is marked `full_annual_horizon=false` and is not an annual-cost result. After supplying real parameters in a JSON file with the same structure as `demo_inputs.json`, run:

```powershell
python planning.py --input real_inputs.json --output planning_result.json
```

For a paired comparison, use `--noise-scale-mw 0` for the noise-free run and `--noise-scale-mw 0.155` for the noisy run, with the same `--seed` in both. The default energy-storage model uses a continuous relaxation. With an installed and licensed Gurobi solver, `--binary-ess` enables binary variables that enforce mutually exclusive charging and discharging.

## Output and limitations

The JSON output separately reports line and SOP investment, annualized investment, upstream-grid electricity purchases, local operating costs for the three microgrids, network-infeasibility counts, minimum voltage, maximum line loading, absolute SOCP gap, ADMM residuals, and whether the stopping threshold was reached. The manuscript's stated objective is `annualized investment + upstream-grid electricity purchases`. Local operating costs are reported separately and are not part of that stated objective.

With continued noise injection at the fixed scale `b=0.155 MW`, ADMM may reach its iteration limit without meeting the `1e-4` residual threshold; the program records this status. The output does not claim a proven total privacy budget. GA returns the **best evaluated candidate**, not a certified global optimum.

The default demonstration uses 12 individuals and eight generations and may still take time. Applying SOCP to every candidate in the manuscript's 50-individual, 100-generation, four-scenario, 24-hour setting would require many solves. A full reproduction first requires the real inputs and run records, followed by consideration of parameterized SOCPs, caching, parallel execution, or hierarchical screening. This smaller demonstration does not replace the manuscript's data.
