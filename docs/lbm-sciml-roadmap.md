# Differentiable LBM Solver - SciML / Broad-ML Roadmap
**CERG-FLUX Lab (Fluids, Learning and Uncertainty in compleX systems)**

> Scope: the machine-learning track of the PyTorch differentiable LBM solver. Broad ML, **not** just PINNs. Structured as a GitHub Projects board: milestones group epics, epics contain issues. Issue keys `E<milestone>.<n>` make dependencies explicit and mirror to GitHub "blocked by" relations.

---

## 1. Board setup

**Custom fields**
- **Effort**: S (<= 3 days) · M (1-2 weeks) · L (3-6 weeks) · XL (research-open, milestone-scale)
- **Priority**: P0 (blocker) · P1 (core) · P2 (nice-to-have)
- **Output**: which paper / deliverable the item feeds

**Labels**
- `type`: epic, feature, research, infra, eval, docs
- `area`: data, mlops, operators, rom, closures, generative, graph, sequence, uq, inverse, rl, discovery, foundation, integration

**Board columns (status)**: Backlog -> Todo -> In progress -> In review -> Done

**Issue line format**: `**key** title · area · effort · priority · deps: ...`

---

## 2. Milestones (capability tiers)

| # | Milestone | Purpose |
|---|-----------|---------|
| M0 | ML Foundations & MLOps | Prerequisite substrate for everything below |
| M1 | Surrogates & Neural Operators | Workhorse forward-prediction models |
| M2 | Hybrid & Physics-Constrained ML | PINN family + learned closures in the loop |
| M3 | Uncertainty, Inverse & Data Assimilation | Lab-core (the "Uncertainty" in the lab name) |
| M4 | Generative, Graph & Spatiotemporal | Distributions, unstructured/particle, long rollouts |
| M5 | Discovery, Control & Foundation Models | Interpretability, RL, transfer, pretraining |

Milestones are capability tiers, not a strict calendar. M0 gates all; M1-M5 then advance largely in parallel on the shared substrate.

---

## M0 - ML Foundations & MLOps

#### E0.A Differentiable solver ML interface
- [ ] **E0.1** nn.Module wrapper around the time loop (differentiable / eval modes) · integration · M · P0
- [ ] **E0.2** Gradient checkpointing across time steps (memory-bounded BPTT) · infra · M · P0 · deps: E0.1
- [ ] **E0.3** Gradient-accuracy benchmark: autodiff vs finite-difference; memory vs steps · eval · S · P0 · deps: E0.2
- [ ] **E0.4** Mixed-precision path (fp32 compute, fp64 accumulation) validated for training stability · infra · M · P1 · deps: E0.1

*Acceptance:* gradients match finite-difference to tolerance on a lid-driven-cavity inverse test; BPTT over N steps fits in 16 GB VRAM via checkpointing.

#### E0.B Data pipeline
- [ ] **E0.5** Solver-to-dataset generator (parametrised runs -> fields) · data · M · P0
- [ ] **E0.6** Storage schema in Zarr/HDF5 with metadata (Re, geometry, BCs, dt) · data · M · P0 · deps: E0.5
- [ ] **E0.7** Palabos result ingester (cross-source dataset from hydrocyclone runs) · data · M · P1 · deps: E0.6
- [ ] **E0.8** Streaming DataLoaders with on-the-fly normalisation and augmentation · data · M · P0 · deps: E0.6
- [ ] **E0.9** Deterministic train/val/test splitter with regime stratification · data · S · P0 · deps: E0.6

*Acceptance:* a single documented command regenerates a benchmark dataset from seed; loaders stream from disk within the VRAM/RAM budget.

#### E0.C Experiment & evaluation harness
- [ ] **E0.10** Config management (Hydra) + seed control for reproducibility · mlops · M · P0
- [ ] **E0.11** Experiment tracking (W&B or MLflow) wired into the train loop · mlops · S · P0 · deps: E0.10
- [ ] **E0.12** Metric library: relative L2, energy-spectrum error, conservation-law violation, rollout-stability horizon · eval · M · P0
- [ ] **E0.13** Baseline registry + auto-generated leaderboard doc · eval · M · P1 · deps: E0.11, E0.12
- [ ] **E0.14** Model registry + checkpoint versioning + ONNX export · mlops · M · P1

*Acceptance:* any experiment reproducible from one config + seed; leaderboard updates automatically from tracked runs.

#### E0.D Compute & CI for ML
- [ ] **E0.15** SLURM job templates for per-node training on mjolnir/legion · infra · S · P1
- [ ] **E0.16** GPU-memory profiler + batch-size autotuner for the 16 GB budget · infra · S · P2 · deps: E0.15
- [ ] **E0.17** CI smoke tests: tiny-model overfit test per family (regression guard) · infra · M · P1

*Acceptance:* `sbatch` template trains a smoke model end-to-end; CI runs overfit tests on every PR.

---

## M1 - Surrogates & Neural Operators

#### E1.A Neural operators
- [ ] **E1.1** FNO (2D) baseline, single-step field prediction · operators · M · P0 · deps: E0.8, E0.12
- [ ] **E1.2** Factorised/tensorised FNO for 3D scaling within VRAM · operators · L · P1 · deps: E1.1
- [ ] **E1.3** DeepONet (branch/trunk) for parametric operator learning · operators · M · P1 · deps: E0.8
- [ ] **E1.4** Geometry-aware operator (GINO / graph-FNO) for irregular domains · operators · L · P2 · deps: E1.1
- [ ] **E1.5** Transformer operator (Transolver-style) baseline · operators · L · P2 · deps: E1.1
- [ ] **E1.6** Autoregressive rollout wrapper + stability regularisation (pushforward / noise injection) · operators · M · P0 · deps: E1.1

*Acceptance:* FNO beats interpolation baseline on held-out Re; rollout stable to target horizon on Taylor-Green.

#### E1.B Reduced-order / latent-dynamics models
- [ ] **E1.7** POD/DMD baseline ROM + reconstruction metrics · rom · S · P1 · deps: E0.12
- [ ] **E1.8** Convolutional autoencoder for field compression · rom · M · P1 · deps: E0.8
- [ ] **E1.9** Latent time-stepper (Transformer/GRU in latent space) · rom · M · P1 · deps: E1.8
- [ ] **E1.10** Neural-ODE latent dynamics variant · rom · L · P2 · deps: E1.8
- [ ] **E1.11** Koopman-based linear latent operator · rom · L · P2 · deps: E1.8

*Acceptance:* latent stepper reconstructs a full-field rollout under error budget at target speed-up.

#### E1.C Super-resolution / upscaling
- [ ] **E1.12** CNN super-resolution, coarse -> fine fields · operators · M · P1 · deps: E0.8
- [ ] **E1.13** Physics-consistent SR (divergence-free / conservation constraints) · closures · M · P2 · deps: E1.12
- [ ] **E1.14** Diffusion-based SR variant (links to M4) · generative · L · P2 · deps: E4.1

*Acceptance:* SR recovers spectra beyond the coarse cutoff without violating continuity beyond tolerance.

---

## M2 - Hybrid & Physics-Constrained ML

#### E2.A Physics-informed family (PINNs and beyond)
- [ ] **E2.1** Baseline PINN, ADE first then NS (aligns with the fp-carleman ADE-first staging) · feature · M · P1 · deps: E0.10
- [ ] **E2.2** Hard-constraint architectures (BC / divergence built into the network) · research · M · P2 · deps: E2.1
- [ ] **E2.3** Causal / curriculum training + loss balancing (NTK / gradnorm) · research · M · P2 · deps: E2.1
- [ ] **E2.4** Head-to-head study: PINN vs operator vs ROM on identical cases · eval · M · P1 · deps: E1.1, E2.1

*Acceptance:* PINN reproduces the Von Karman regime; comparison table populated across families.

#### E2.B Learned closures (solver-in-the-loop) — the moat
- [ ] **E2.5** Pluggable ML collision/closure module conforming to the solver's collision interface · closures/integration · M · P0 · deps: E0.1
- [ ] **E2.6** A-priori closure training (offline targets) · closures · M · P1 · deps: E2.5, E0.8
- [ ] **E2.7** A-posteriori training: backprop through K solver steps into the closure · closures/integration · L · P0 · deps: E2.5, E0.2
- [ ] **E2.8** Learned LES/subgrid closure for under-resolved LBM · closures · L · P1 · deps: E2.7
- [ ] **E2.9** Stability + out-of-distribution generalisation audit of learned closures · eval · M · P0 · deps: E2.7

*Acceptance:* a-posteriori-trained closure keeps a coarse run accurate over target horizon and beats a-priori on stability.

#### E2.C Hybrid ML-corrected solvers
- [ ] **E2.10** Residual/correction model added to the solver update (learned defect correction) · integration · M · P1 · deps: E2.5
- [ ] **E2.11** ML-accelerated timestepping (predict-then-correct) with fallback guard · integration · L · P2 · deps: E2.10
- [ ] **E2.12** Solver/ML switching policy + error monitor · integration · M · P2 · deps: E2.10

*Acceptance:* hybrid solver reaches target accuracy at measured wall-clock speed-up with bounded error.

---

## M3 - Uncertainty, Inverse Problems & Data Assimilation

#### E3.A Uncertainty quantification
- [ ] **E3.1** Deep-ensembles wrapper, generic across model families · uq · M · P0 · deps: E0.14
- [ ] **E3.2** MC-dropout + heteroscedastic / evidential heads · uq · M · P1 · deps: E3.1
- [ ] **E3.3** Bayesian NN (variational / Laplace) operator variant · uq · L · P2 · deps: E1.1
- [ ] **E3.4** Conformal prediction for calibrated field-wise error bars · uq · M · P1 · deps: E3.1
- [ ] **E3.5** UQ calibration + sharpness benchmark (reliability diagrams, CRPS) · eval · M · P0 · deps: E3.1

*Acceptance:* predictive intervals calibrated (coverage within tolerance) on held-out regimes; UQ report auto-generated.

#### E3.B Inverse problems (through the differentiable solver)
- [ ] **E3.6** Gradient-based parameter inversion (viscosity / forcing) via E0.2 · inverse · M · P0 · deps: E0.2
- [ ] **E3.7** Geometry/topology inversion (differentiable masks) · inverse · L · P1 · deps: E3.6
- [ ] **E3.8** Bayesian inversion (SVGD / HMC) for posterior over parameters · inverse/uq · L · P2 · deps: E3.6, E3.1

*Acceptance:* recovers known parameters/geometry from synthetic observations within error budget.

#### E3.C Data assimilation
- [ ] **E3.9** 4D-Var-style DA using the solver adjoint · inverse · L · P1 · deps: E0.2
- [ ] **E3.10** Ensemble-Kalman / ML-hybrid DA · inverse/uq · L · P2 · deps: E3.1, E3.9
- [ ] **E3.11** Active-learning loop: uncertainty-driven solver querying · uq/data · M · P2 · deps: E3.5, E0.5

*Acceptance:* DA reduces state error vs a no-assimilation baseline on a twin experiment.

---

## M4 - Generative, Graph & Spatiotemporal Models

#### E4.A Generative models
- [ ] **E4.1** Conditional diffusion model for flow-field generation / inpainting · generative · L · P1 · deps: E0.8
- [ ] **E4.2** Parameter-conditioned diffusion surrogate (sample fields given Re / BCs) · generative · L · P2 · deps: E4.1
- [ ] **E4.3** Flow-matching / score-based variant + sampling-speed study · generative · L · P2 · deps: E4.1
- [ ] **E4.4** GAN / VAE baselines for comparison + data augmentation · generative · M · P2 · deps: E0.8
- [ ] **E4.5** Generative UQ: ensemble-of-samples calibration (links E3.5) · eval · M · P2 · deps: E4.1, E3.5

*Acceptance:* diffusion surrogate produces physically plausible fields (spectra, conservation) with calibrated spread.

#### E4.B Graph neural simulators (unstructured / particle, ties to LBM-DEM)
- [ ] **E4.6** GNN mesh/particle data adapter from LBM-DEM particle state · graph/data · M · P1 · deps: E0.6
- [ ] **E4.7** MeshGraphNet-style GNN simulator baseline · graph · L · P1 · deps: E4.6
- [ ] **E4.8** Rollout stability + noise-injection training for GNN · graph · M · P1 · deps: E4.7
- [ ] **E4.9** GNN for particle-laden flow (hydrocyclone regime) · graph · L · P2 · deps: E4.7

*Acceptance:* GNN simulator matches solver trajectories over horizon on a particle test case.

#### E4.C Spatiotemporal sequence models
- [ ] **E4.10** ConvLSTM / attention baseline for multi-step forecasting · sequence · M · P2 · deps: E0.8
- [ ] **E4.11** Spatiotemporal transformer for long rollouts · sequence · L · P2 · deps: E4.10
- [ ] **E4.12** Long-horizon rollout benchmark across families · eval · M · P1 · deps: E1.6, E4.11

*Acceptance:* best sequence model extends stable-rollout horizon vs the autoregressive operator baseline.

---

## M5 - Discovery, Control & Foundation Models

#### E5.A Equation / closure discovery
- [ ] **E5.1** SINDy sparse-regression for effective equations from data · discovery · M · P1 · deps: E0.5
- [ ] **E5.2** Symbolic regression for interpretable closures (ties to E2.8) · discovery/closures · L · P2 · deps: E2.8
- [ ] **E5.3** Discovery validation vs known limits (interpretability report) · eval/docs · M · P2 · deps: E5.1

*Acceptance:* recovers known governing terms on canonical cases; discovered closure runs inside the solver.

#### E5.B Reinforcement learning: control & adaptivity
- [ ] **E5.4** RL environment wrapping the differentiable solver (Gym API) · rl/integration · M · P1 · deps: E0.1
- [ ] **E5.5** RL for active flow control (drag reduction / mixing) · rl · L · P2 · deps: E5.4
- [ ] **E5.6** RL / learned policy for adaptive Carleman truncation order (bake-off vs the fp-carleman PINN) · rl · L · P1 · deps: E5.4
- [ ] **E5.7** RL for adaptive timestep / mesh / solver-switching · rl/integration · L · P2 · deps: E5.4, E2.12

*Acceptance:* RL policy improves the control objective / cost vs a fixed baseline on the target case.

#### E5.C Multi-fidelity, transfer & foundation models
- [ ] **E5.8** Multi-fidelity model (cheap + expensive data fusion) · foundation · L · P2 · deps: E0.7
- [ ] **E5.9** Transfer / fine-tuning across regimes and geometries · foundation · L · P2 · deps: E1.2
- [ ] **E5.10** PDE foundation-model pretraining across a multi-regime corpus · foundation · XL · P2 · deps: E0.6, E1.2
- [ ] **E5.11** Fine-tune foundation model to the hydrocyclone / particle regime · foundation · L · P2 · deps: E5.10

*Acceptance:* pretrained-then-finetuned model beats from-scratch on a low-data target regime.

---

## 3. Cross-cutting notes

**Integration back into the solver and the quantum track**
- Learned closures (E2.B) share the *exact* collision interface used by the quantum/Carleman collision on Track C. A learned closure and a quantum closure are therefore swappable at one seam.
- Adaptive-truncation RL (E5.6) is a direct alternative to the fp-carleman PINN-predicts-truncation-order approach. Run them as a bake-off and report both.

**Compute reality**
- Training budget is mjolnir's A2000 Ada (16 GB) plus prosumer nodes, on standalone SLURM (no cross-node distributed training). Every model must train per-node: mixed precision (E0.4), gradient checkpointing (E0.2), streaming data (E0.8), and 3D via factorised operators (E1.2).
- The XL foundation-model item (E5.10) will need external HPC. Flag it as compute-gated; do not schedule it against the homelab.

**Output mapping (each milestone yields at least one paper)**
- M1 -> operator / surrogate methods paper
- M2 -> differentiable learned-closure paper (the flagship differentiability result)
- M3 -> UQ + inverse / data-assimilation paper
- M4 -> generative or GNN-simulator paper
- M5 -> discovery or RL-control paper; foundation model as a longer-horizon output
- Whole component -> JOSS software paper alongside the core solver

---

## 4. Suggested first sprint (2-3 weeks)

Start with the substrate slice. Nothing else unblocks without it.

1. **E0.1** nn.Module wrapper (P0)
2. **E0.2** gradient checkpointing (P0)
3. **E0.3** gradient-accuracy benchmark (P0)
4. **E0.5 + E0.6** dataset generator + schema (P0)
5. **E0.10 + E0.11** Hydra config + experiment tracking (P0)
6. **E1.1** FNO baseline as the first real model to prove the pipeline end-to-end (P0)

Result: a working *generate data -> train FNO -> tracked, reproducible, differentiable* loop. That is the spine everything else hangs off.
