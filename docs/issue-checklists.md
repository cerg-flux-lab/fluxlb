# Issue sub-task checklists

Deep breakdown of every board issue (M0-M5) into concrete sub-tasks. Paste each block into its GitHub issue body, or run an updated board script to include them.

**Legend**
- `[ ]` sub-task (unchecked)
- **`[T1]`** = scientific-logic decision. Tier 1 under the Intellectual Ownership Framework: yours to own and defend; Claude explains and suggests only. Untagged sub-tasks are engineering / boilerplate (Tier 2/3).

---

## M0 - ML Foundations & MLOps

### [E0.1] nn.Module wrapper around the time loop
- [ ] Define `LBMSolver(nn.Module)` holding lattice, collision and boundaries as submodules
- [ ] Implement `run(f, n_steps)` returning final populations (optionally the trajectory)
- [ ] Add `differentiable: bool` toggling grad tracking vs a `torch.no_grad` fast path
- [ ] Register lattice constants as buffers so `.to(device)` moves them
- [ ] Unit test: forward parity between differentiable and eval modes on a small grid
- [ ] Docstring + minimal runnable example

### [E0.2] Gradient checkpointing across time steps
- [ ] Wrap the per-step (or per-chunk) update in `torch.utils.checkpoint`
- [ ] Add `checkpoint_every: int` to trade memory against recompute
- [ ] Verify gradients unchanged vs non-checkpointed on a small case
- [ ] Benchmark peak VRAM vs step count at several `checkpoint_every`
- [ ] Document the memory/compute trade-off

### [E0.3] Gradient-accuracy benchmark (autodiff vs finite-difference)
- [ ] **[T1]** Choose the scalar objective (e.g. cavity centreline error)
- [ ] Implement a finite-difference gradient wrt one parameter
- [ ] Compare autodiff vs FD; report relative error
- [ ] Sweep step count; plot memory and gradient error
- [ ] Add as a CI regression test with tolerance

### [E0.4] Mixed-precision path (fp32 compute, fp64 accumulation)
- [ ] **[T1]** Decide which quantities require fp64 accumulation (density, momentum)
- [ ] Implement the dtype policy and guard sensitive reductions in fp64
- [ ] Compare stability vs full-fp64 on Taylor-Green over N steps
- [ ] Integrate autocast where safe + test
- [ ] Document the precision policy

### [E0.5] Solver-to-dataset generator
- [ ] Define a run-spec schema (Re, geometry, BCs, dt, duration, sampling stride)
- [ ] Batch-run the solver over a parameter grid
- [ ] Save fields + metadata per run
- [ ] Add resumability / skip-existing
- [ ] Smoke test: generate a tiny dataset from a seed

### [E0.6] Storage schema (Zarr/HDF5)
- [ ] Choose Zarr vs HDF5 and a chunking layout
- [ ] Define the group/array hierarchy keyed on run metadata
- [ ] Write reader/writer utilities
- [ ] Store units and normalisation statistics
- [ ] Round-trip test (write then read equals original)

### [E0.7] Palabos result ingester
- [ ] Map Palabos output (VTK/raw) fields onto the schema
- [ ] **[T1]** Reconcile coordinate, unit and non-dimensionalisation conventions between Palabos and FluxLB
- [ ] Convert a hydrocyclone run as a fixture
- [ ] Validate against a known Palabos summary statistic
- [ ] Document the mapping

### [E0.8] Streaming DataLoaders
- [ ] Dataset class reading lazily from the store
- [ ] On-the-fly normalisation using stored stats
- [ ] **[T1]** Decide which augmentations are physically valid (lattice-symmetric rotations/reflections)
- [ ] Prefetch + pinned memory; verify throughput
- [ ] Test: batch shapes/dtypes and normalisation invertibility

### [E0.9] Deterministic splitter with regime stratification
- [ ] Define stratification keys (Re band, geometry)
- [ ] Seeded split with no time leakage from the same run
- [ ] Persist a split manifest
- [ ] Test: disjoint sets, reproducible

### [E0.10] Config management (Hydra) + seeds
- [ ] Config groups: solver, data, model, train, eval
- [ ] Global seed + deterministic flags
- [ ] Config composition + CLI overrides
- [ ] Snapshot the resolved config per run
- [ ] Test: same config+seed reproduces

### [E0.11] Experiment tracking
- [ ] Initialise tracker (W&B/MLflow) from config
- [ ] Log metrics, config, git SHA, artefacts
- [ ] Offline-mode fallback
- [ ] Test: a run appears with the expected fields

### [E0.12] Metric library
- [ ] Implement relative L2 and MAE
- [ ] **[T1]** Define the energy spectrum and its binning; implement spectral error
- [ ] **[T1]** Define the conservation-violation metric (mass/momentum drift)
- [ ] Implement rollout-stability horizon (first step exceeding threshold)
- [ ] Unit tests on analytic fields

### [E0.13] Baseline registry + leaderboard
- [ ] Pull tracked runs into a table
- [ ] Rank per benchmark/metric
- [ ] Auto-write a markdown/HTML leaderboard
- [ ] Scheduled/CI regeneration
- [ ] Doc: how to add a baseline

### [E0.14] Model registry + ONNX export
- [ ] Checkpoint save/load with config + metrics
- [ ] Semantic version tags
- [ ] ONNX export (where applicable) + parity check
- [ ] Test: round-trip load reproduces outputs

### [E0.15] SLURM job templates
- [ ] `train.slurm` parametrised by config
- [ ] Resource requests sized for the A2000 (16 GB)
- [ ] Logging + requeue on preemption
- [ ] Per-node only (document the no-cross-node constraint)
- [ ] Dry-run test

### [E0.16] Memory profiler + batch-size autotuner
- [ ] Hook `torch.cuda.max_memory_allocated`
- [ ] Binary-search batch size under a VRAM cap
- [ ] Persist the chosen batch per model
- [ ] Test on a small model

### [E0.17] CI smoke tests
- [ ] Tiny-overfit test per family (loss -> near zero on 1-2 samples)
- [ ] Mark `@pytest.mark.smoke`; run on every PR
- [ ] Cache tiny fixtures
- [ ] Wire fail-fast into CI

---

## M1 - Surrogates & Neural Operators

### [E1.1] FNO (2D) baseline
- [ ] **[T1]** Choose mode count and network width/depth
- [ ] Spectral conv layer (rfft/irfft, mode truncation) + FNO block stack with lifting/projection
- [ ] Training loop for single-step prediction
- [ ] Compare vs interpolation baseline on held-out Re
- [ ] Register best model + example notebook

### [E1.2] Factorised/tensorised FNO for 3D
- [ ] **[T1]** Choose the factorisation and mode budget for 3D
- [ ] Implement factorised spectral weights
- [ ] Fit on the A2000 (checkpointing, mode budget)
- [ ] Validate on a 3D case + throughput/memory report

### [E1.3] DeepONet
- [ ] **[T1]** Design branch (input function) and trunk (coordinate) networks
- [ ] Wire the parametric training set
- [ ] Evaluate across unseen parameters
- [ ] Compare vs FNO on the same task

### [E1.4] Geometry-aware operator (GINO / graph-FNO)
- [ ] **[T1]** Define graph construction from geometry
- [ ] Implement the graph-spectral operator layer
- [ ] Irregular-domain test case
- [ ] Accuracy vs grid FNO

### [E1.5] Transformer operator
- [ ] **[T1]** Define field/patch tokenisation
- [ ] Implement the attention operator block
- [ ] Train + evaluate on a standard case
- [ ] Cost vs accuracy vs FNO

### [E1.6] Autoregressive rollout wrapper
- [ ] Implement the autoregressive loop utility
- [ ] **[T1]** Choose the stabilisation strategy (pushforward trick / training-time noise)
- [ ] Measure the stable horizon
- [ ] Compare with vs without stabilisation

### [E1.7] POD/DMD ROM
- [ ] **[T1]** Choose POD mode count / truncation
- [ ] Compute POD modes and fit the DMD operator
- [ ] Reconstruction + prediction metrics
- [ ] Add as a leaderboard baseline

### [E1.8] Convolutional autoencoder
- [ ] **[T1]** Design encoder/decoder and latent size
- [ ] Reconstruction training + latent-size sweep
- [ ] Compression vs error curve
- [ ] Export the encoder for E1.9

### [E1.9] Latent time-stepper
- [ ] **[T1]** Choose the latent sequence model (Transformer/GRU)
- [ ] Train on latent trajectories
- [ ] Full-field rollout via decode
- [ ] Compare speed-up + error

### [E1.10] Neural-ODE latent dynamics
- [ ] **[T1]** Design the ODE-function net and choose the solver
- [ ] Train with adjoint/backprop
- [ ] Stability over horizon
- [ ] Compare vs the discrete stepper

### [E1.11] Koopman linear latent operator
- [ ] **[T1]** Formulate the linear latent operator objective
- [ ] Fit + eigen-analysis of the operator
- [ ] Long-horizon linear prediction
- [ ] Link discovered modes to E5.1

### [E1.12] CNN super-resolution
- [ ] **[T1]** Define the physical coarsening operator that builds coarse/fine pairs
- [ ] Implement + train the SR CNN
- [ ] Spectral-recovery evaluation
- [ ] Baseline vs bicubic

### [E1.13] Physics-consistent SR
- [ ] **[T1]** Choose the constraint (divergence-free projection / conservation penalty)
- [ ] Implement the constraint or projection layer
- [ ] Constraint-violation metric before/after
- [ ] Compare vs unconstrained SR

### [E1.14] Diffusion-based SR
- [ ] **[T1]** Define coarse-field conditioning
- [ ] Reuse the E4.1 diffusion backbone
- [ ] Sample quality + spectra
- [ ] Note sampling cost

---

## M2 - Hybrid & Physics-Constrained ML

### [E2.1] Baseline PINN (ADE then NS)
- [ ] **[T1]** Formulate the ADE residual loss
- [ ] **[T1]** Choose the collocation sampling strategy
- [ ] **[T1]** Extend to the NS residual + incompressibility
- [ ] Reproduce the Von Karman regime
- [ ] Compare training-stability tricks

### [E2.2] Hard-constraint architectures
- [ ] **[T1]** Choose the BC-satisfying ansatz / output transform
- [ ] **[T1]** Choose the divergence-free construction (e.g. stream-function)
- [ ] Verify constraints hold exactly
- [ ] Accuracy vs soft-constraint PINN

### [E2.3] Causal / curriculum training
- [ ] **[T1]** Choose causal weighting / time-marching curriculum
- [ ] **[T1]** Choose the loss-balancing scheme (NTK or gradnorm)
- [ ] Ablate vs baseline PINN
- [ ] Report convergence gains

### [E2.4] Head-to-head: PINN vs operator vs ROM
- [ ] Fix identical cases + metrics
- [ ] Run PINN, FNO and ROM under the harness
- [ ] Produce the comparison table + plots
- [ ] **[T1]** Write the findings and interpretation

### [E2.5] Pluggable ML collision/closure module
- [ ] Conform to the `CollisionOperator` interface
- [ ] **[T1]** Decide what the module predicts (full collision vs correction)
- [ ] Drop-in swap test (runs inside the solver)
- [ ] Interface-compliance test

### [E2.6] A-priori closure training
- [ ] **[T1]** Define offline targets (filtered fine/DNS data)
- [ ] Train the closure to targets
- [ ] Offline error evaluation
- [ ] Wire in for later a-posteriori training

### [E2.7] A-posteriori closure training
- [ ] Backprop through K solver steps into the closure (uses E0.2)
- [ ] **[T1]** Choose the unroll length K and the loss
- [ ] Train + compare vs a-priori
- [ ] Stability over horizon

### [E2.8] Learned LES / subgrid closure
- [ ] **[T1]** Define the subgrid target / closure form
- [ ] Train on under-resolved runs
- [ ] A-posteriori accuracy vs resolved reference
- [ ] Compare vs a Smagorinsky baseline

### [E2.9] Stability + OOD audit of learned closures
- [ ] Define the OOD Re/geometry set
- [ ] Long-rollout stability tests
- [ ] **[T1]** Analyse failure modes
- [ ] Report + guardrails

### [E2.10] Residual/correction model
- [ ] Add a learned correction to the solver update
- [ ] Train the correction in-loop
- [ ] Accuracy/speed trade-off
- [ ] Fallback when the correction is unreliable

### [E2.11] ML-accelerated timestepping
- [ ] **[T1]** Design the predict-then-correct scheme
- [ ] Error monitor + guard
- [ ] Speed-up vs accuracy
- [ ] Safety-fallback test

### [E2.12] Solver/ML switching policy
- [ ] **[T1]** Design the error estimator driving the switch
- [ ] Implement the policy (threshold/hysteresis)
- [ ] Demonstrate bounded error
- [ ] Log switch events

---

## M3 - Uncertainty, Inverse & Data Assimilation

### [E3.1] Deep-ensembles wrapper
- [ ] Ensemble wrapper over any model family
- [ ] Parallel/seeded training
- [ ] Mean + variance prediction
- [ ] Hand calibration off to E3.5

### [E3.2] MC-dropout + evidential heads
- [ ] **[T1]** Choose the heteroscedastic / evidential formulation
- [ ] Implement dropout-at-inference + the uncertainty head
- [ ] Compare uncertainty quality
- [ ] Cost vs ensembles

### [E3.3] Bayesian NN operator
- [ ] **[T1]** Choose variational vs Laplace approximation
- [ ] Train + posterior predictive
- [ ] Calibration vs ensembles
- [ ] Cost note

### [E3.4] Conformal prediction
- [ ] **[T1]** Choose the calibration set + nonconformity score
- [ ] Produce field-wise prediction bands
- [ ] Coverage check
- [ ] Compare sharpness

### [E3.5] UQ calibration + sharpness benchmark
- [ ] **[T1]** Choose the calibration metrics (reliability, coverage, CRPS)
- [ ] Implement sharpness metrics
- [ ] Auto-generate a UQ report
- [ ] Rank UQ methods

### [E3.6] Gradient-based parameter inversion
- [ ] **[T1]** Define the observation objective
- [ ] Gradient inversion via E0.2
- [ ] Recover known parameters (twin test)
- [ ] Robustness to observation noise

### [E3.7] Geometry/topology inversion
- [ ] **[T1]** Parametrise geometry (differentiable mask / level-set)
- [ ] **[T1]** Choose the regularisation
- [ ] Recover a known geometry
- [ ] Report error

### [E3.8] Bayesian inversion
- [ ] **[T1]** Choose the sampler (SVGD / HMC)
- [ ] Posterior recovery on a twin experiment
- [ ] Uncertainty in inferred parameters
- [ ] Cost/scaling note

### [E3.9] 4D-Var-style DA
- [ ] **[T1]** Define the cost functional; adjoint via autodiff
- [ ] **[T1]** Choose the assimilation window
- [ ] Twin-experiment error reduction
- [ ] Compare to no-DA

### [E3.10] Ensemble-Kalman / ML-hybrid DA
- [ ] **[T1]** Choose the ensemble-DA scheme
- [ ] Implement + an ML-hybrid variant
- [ ] Twin experiment
- [ ] Compare to 4D-Var

### [E3.11] Active-learning loop
- [ ] **[T1]** Define the acquisition function from UQ
- [ ] Implement the query-generate-retrain loop
- [ ] Sample-efficiency vs random
- [ ] Budgeted comparison

---

## M4 - Generative, Graph & Spatiotemporal

### [E4.1] Conditional diffusion model
- [ ] **[T1]** Choose the noise schedule and backbone (U-Net)
- [ ] **[T1]** Choose the conditioning mechanism
- [ ] Train + sample
- [ ] Physical-plausibility evaluation (spectra, conservation)

### [E4.2] Parameter-conditioned diffusion surrogate
- [ ] **[T1]** Define conditioning on Re/BCs
- [ ] Train across regimes
- [ ] Sample fidelity per regime
- [ ] Compare vs an FNO surrogate

### [E4.3] Flow-matching / score-based variant
- [ ] **[T1]** Choose the flow-matching / score objective
- [ ] Implement sampler + speed study
- [ ] Quality vs diffusion
- [ ] Cost table

### [E4.4] GAN/VAE baselines
- [ ] **[T1]** Choose VAE + GAN architectures
- [ ] Train + physical/perceptual metric
- [ ] Build the augmentation pipeline
- [ ] Compare vs diffusion

### [E4.5] Generative UQ
- [ ] Ensemble-of-samples spread
- [ ] Calibration (reuse E3.5)
- [ ] Coverage of true fields
- [ ] Report

### [E4.6] GNN mesh/particle data adapter
- [ ] **[T1]** Define graph construction (nodes/edges/features) from LBM-DEM state
- [ ] **[T1]** Choose neighbour construction + cutoffs
- [ ] Batch graphs
- [ ] Round-trip / consistency test

### [E4.7] MeshGraphNet-style GNN simulator
- [ ] **[T1]** Design the encode-process-decode GNN
- [ ] Train on trajectories
- [ ] Trajectory match vs solver
- [ ] Save model

### [E4.8] GNN rollout stability
- [ ] **[T1]** Choose the noise-injection scheme
- [ ] Long-rollout stability
- [ ] Compare with vs without
- [ ] Horizon metric

### [E4.9] GNN for particle-laden flow
- [ ] **[T1]** Design two-way coupling features
- [ ] Train on the hydrocyclone regime
- [ ] **[T1]** Define the separation-accuracy metric
- [ ] Report

### [E4.10] ConvLSTM / attention baseline
- [ ] **[T1]** Choose the ConvLSTM/attention architecture
- [ ] Multi-step training
- [ ] Forecast accuracy
- [ ] Add as a baseline

### [E4.11] Spatiotemporal transformer
- [ ] **[T1]** Design the space-time attention
- [ ] Train for long rollouts
- [ ] Horizon vs cost
- [ ] Compare vs ConvLSTM

### [E4.12] Long-horizon rollout benchmark
- [ ] **[T1]** Define the common long-rollout protocol
- [ ] Run operator and sequence models
- [ ] Horizon comparison table
- [ ] **[T1]** Write the findings

---

## M5 - Discovery, Control & Foundation Models

### [E5.1] SINDy sparse-regression
- [ ] **[T1]** Choose the candidate function library
- [ ] **[T1]** Choose the sparse-regression method (e.g. STLSQ) and threshold
- [ ] Recover known terms on a canonical case
- [ ] **[T1]** Report and interpret the discovered equations

### [E5.2] Symbolic regression for closures
- [ ] **[T1]** Set up SR on learned-closure input/output
- [ ] Produce a complexity/accuracy Pareto front
- [ ] Run a discovered closure in the solver
- [ ] **[T1]** Interpret the result

### [E5.3] Discovery validation
- [ ] **[T1]** Compare discovered forms vs known asymptotic limits
- [ ] Robustness to noise
- [ ] **[T1]** Write the interpretability report

### [E5.4] RL environment
- [ ] Gym-API wrapper on the differentiable solver
- [ ] **[T1]** Design state, action and reward
- [ ] Step/reset + seeding
- [ ] Sanity policy test

### [E5.5] RL for active flow control
- [ ] **[T1]** Define the control objective (drag reduction / mixing)
- [ ] Train the policy (PPO/SAC)
- [ ] Improvement vs a fixed baseline
- [ ] Robustness

### [E5.6] RL for adaptive Carleman truncation
- [ ] **[T1]** Design the reward (accuracy vs cost of truncation order)
- [ ] Train the policy on the ADE/NS staging
- [ ] Bake-off vs the fp-carleman PINN truncation predictor
- [ ] **[T1]** Report which approach wins where

### [E5.7] RL for adaptive timestep/mesh/switching
- [ ] **[T1]** Define the action space (dt / mesh / switch)
- [ ] **[T1]** Define the reward (accuracy vs cost)
- [ ] Train + evaluate
- [ ] Compare to a fixed schedule

### [E5.8] Multi-fidelity model
- [ ] **[T1]** Choose the multi-fidelity method (MF-DNN / co-kriging)
- [ ] Fuse cheap + expensive data
- [ ] Accuracy vs single-fidelity
- [ ] Cost saving

### [E5.9] Transfer / fine-tuning
- [ ] **[T1]** Define the fine-tuning protocol across regimes
- [ ] Low-data target evaluation
- [ ] Vs from-scratch
- [ ] Report

### [E5.10] PDE foundation-model pretraining
- [ ] **[T1]** Assemble the multi-regime corpus and its coverage
- [ ] **[T1]** Choose the pretraining objective
- [ ] Compute plan (external HPC, not the homelab)
- [ ] Checkpoints + evaluation harness

### [E5.11] Foundation-model fine-tune
- [ ] Fine-tune to the hydrocyclone/particle regime
- [ ] Low-data performance
- [ ] Vs from-scratch and vs E5.9
- [ ] Report
