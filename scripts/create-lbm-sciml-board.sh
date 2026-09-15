#!/usr/bin/env bash
# Copyright 2026 Muaaz Bhamjee
# SPDX-License-Identifier: Apache-2.0
#
# create-lbm-sciml-board.sh
# Stands up the CERG-FLUX Lab differentiable-LBM SciML board on GitHub:
# labels, milestones (M0-M5), and all 77 issues with bodies, deps and priority labels.
#
# Prerequisites:
#   - gh CLI installed and authenticated:            gh auth login
#   - if adding issues to a Project (v2), grant scope: gh auth refresh -s project
#   - the target repo must already exist.
#
# Usage:
#   1. Edit OWNER / REPO / PROJECT_NUMBER below.
#   2. Preview without writing anything:   DRY_RUN=1 ./create-lbm-sciml-board.sh
#   3. Create for real:                    ./create-lbm-sciml-board.sh
#
# The script is re-runnable: it skips labels/milestones/issues that already exist
# (issues matched by their [E?.?] key in the title).
#
# Notes:
#   - Effort and epic are written into each issue body; Priority is applied as a label.
#     If you would rather have Effort/Priority as Projects v2 custom fields, that needs a
#     GraphQL step (get field + option IDs, then update item field values) - ask and I'll add it.
#   - Dependencies are written in each body as "Depends on: ...". Native GitHub "blocked by"
#     relations aren't reliably settable from the gh CLI yet; add them in the Project UI or
#     via GraphQL if you want the hard links.

set -euo pipefail

# ===== Config (EDIT THESE) =====
OWNER="cerg-flux-lab"     # GitHub org or username
REPO="fluxlb"         # repository name
PROJECT_NUMBER=""         # Projects v2 number to add issues to; leave empty to skip
DRY_RUN="${DRY_RUN:-0}"   # DRY_RUN=1 prints commands instead of running them
CHECKLISTS_FILE="${CHECKLISTS_FILE:-docs/issue-checklists.md}"  # sub-task source; absent = skip appending
# ================================

gh_run() { if [[ "$DRY_RUN" == "1" ]]; then echo "+ gh $*"; else gh "$@"; fi; }

# Preflight (read-only)
gh repo view "$OWNER/$REPO" >/dev/null 2>&1 || { echo "ERROR: repo $OWNER/$REPO not found or no access."; exit 1; }

# ---- Labels ----
mklabel() { gh_run label create "$1" --color "$2" --description "$3" --force --repo "$OWNER/$REPO"; }

echo ">> Labels"
mklabel "type:feature"   "1d76db" "Concrete, buildable capability"
mklabel "type:research"  "0e8a16" "Open-ended / experimental"
mklabel "type:infra"     "b60205" "Plumbing, tooling, pipelines"
mklabel "type:eval"      "fbca04" "Benchmarks and evaluation"
mklabel "type:docs"      "0075ca" "Documentation and reports"
mklabel "type:epic"      "6f42c1" "Epic (grouping)"
for a in data mlops operators rom closures generative graph sequence uq inverse rl discovery foundation integration; do
  mklabel "area:$a" "c5def5" "Area: $a"
done
mklabel "P0" "d93f0b" "Blocker"
mklabel "P1" "e99695" "Core"
mklabel "P2" "c2e0c6" "Nice-to-have"

# ---- Milestones ----
ensure_milestone() {
  local title="$1" desc="$2"
  if gh api "repos/$OWNER/$REPO/milestones?state=all" --jq '.[].title' 2>/dev/null | grep -Fxq "$title"; then
    echo "   milestone exists: $title"
  else
    gh_run api -X POST "repos/$OWNER/$REPO/milestones" -f title="$title" -f description="$desc" >/dev/null
    echo "   created milestone: $title"
  fi
}

echo ">> Milestones"
ensure_milestone "M0 - ML Foundations & MLOps"              "Prerequisite substrate: differentiable interface, data pipeline, MLOps."
ensure_milestone "M1 - Surrogates & Neural Operators"       "Workhorse forward-prediction models."
ensure_milestone "M2 - Hybrid & Physics-Constrained ML"     "PINN family plus learned closures trained in the loop."
ensure_milestone "M3 - Uncertainty, Inverse & Data Assimilation" "Lab-core: UQ, inverse problems, data assimilation."
ensure_milestone "M4 - Generative, Graph & Spatiotemporal"  "Distributions, unstructured/particle, long rollouts."
ensure_milestone "M5 - Discovery, Control & Foundation Models" "Interpretability, RL, transfer, pretraining."

# ---- Issues ----
# MS, EPIC, ACC are set before each epic block and read by mkissue.

# Pull the sub-task block for a given key out of CHECKLISTS_FILE (empty if unavailable).
CHECKLISTS_OK=0
if [[ -n "$CHECKLISTS_FILE" && -f "$CHECKLISTS_FILE" ]]; then
  CHECKLISTS_OK=1
elif [[ -n "$CHECKLISTS_FILE" ]]; then
  echo "NOTE: $CHECKLISTS_FILE not found; issues will be created without sub-task checklists."
fi
extract_checklist() {
  local key="$1"
  [[ "$CHECKLISTS_OK" == "1" ]] || return 0
  awk -v hdr="### [$key] " '
    index($0, hdr)==1 { grab=1; next }
    grab && (/^### \[/ || /^## / || /^---/) { exit }
    grab { print }
  ' "$CHECKLISTS_FILE"
}

mkissue() {
  local key="$1" title="$2" labels="$3" effort="$4" deps="$5" goal="$6"
  local full_title="[$key] $title"
  if gh issue list --repo "$OWNER/$REPO" --state all --limit 500 --search "in:title \"[$key]\"" \
        --json title --jq '.[].title' 2>/dev/null | grep -Fq "[$key]"; then
    echo "   issue exists: $full_title"; return 0
  fi
  local body
  body=$(cat <<EOF
**Goal.** $goal

**Epic.** $EPIC | **Effort:** $effort | **Depends on:** ${deps:-none}

**Acceptance (epic-level).** $ACC
EOF
)
  local checklist; checklist="$(extract_checklist "$key")"
  if [[ -n "$checklist" ]]; then
    body="$body

---

### Sub-tasks
_\`[T1]\` = scientific-logic decision you own (see CLAUDE.md)._

$checklist"
  fi
  local label_args=(); IFS=',' read -ra L <<< "$labels"
  for l in "${L[@]}"; do label_args+=(--label "$l"); done

  if [[ -n "$PROJECT_NUMBER" && "$DRY_RUN" != "1" ]]; then
    local url; url=$(gh issue create --repo "$OWNER/$REPO" --title "$full_title" --body "$body" --milestone "$MS" "${label_args[@]}")
    gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$url" >/dev/null
  else
    gh_run issue create --repo "$OWNER/$REPO" --title "$full_title" --body "$body" --milestone "$MS" "${label_args[@]}" >/dev/null
  fi
  echo "   created: $full_title"
}

echo ">> Issues"

# ===== M0 =====
MS="M0 - ML Foundations & MLOps"

EPIC="E0.A Differentiable solver ML interface"
ACC="Gradients match finite-difference to tolerance on a lid-driven-cavity inverse test; BPTT over N steps fits in 16 GB VRAM via checkpointing."
mkissue E0.1 "nn.Module wrapper around the time loop (differentiable / eval modes)" "type:infra,area:integration,P0" M ""            "Expose the LBM time loop as a first-class trainable nn.Module with switchable differentiable and inference modes."
mkissue E0.2 "Gradient checkpointing across time steps (memory-bounded BPTT)"       "type:infra,P0"                M "E0.1"        "Checkpoint the rollout so backprop-through-time stays within VRAM."
mkissue E0.3 "Gradient-accuracy benchmark (autodiff vs finite-difference)"          "type:eval,P0"                 S "E0.2"        "Verify gradient correctness and profile memory against step count."
mkissue E0.4 "Mixed-precision path (fp32 compute, fp64 accumulation)"               "type:infra,P1"                M "E0.1"        "Stable reduced-precision training given the prosumer-GPU fp64 penalty."

EPIC="E0.B Data pipeline"
ACC="A single documented command regenerates a benchmark dataset from seed; loaders stream from disk within the VRAM/RAM budget."
mkissue E0.5 "Solver-to-dataset generator (parametrised runs -> fields)"            "type:infra,area:data,P0"      M ""            "Turn parametrised solver runs into labelled field datasets."
mkissue E0.6 "Storage schema in Zarr/HDF5 with metadata"                            "type:infra,area:data,P0"      M "E0.5"        "Chunked array store keyed on Re, geometry, BCs and dt."
mkissue E0.7 "Palabos result ingester (cross-source dataset)"                       "type:infra,area:data,P1"      M "E0.6"        "Import hydrocyclone Palabos runs into the same schema for cross-source training and validation."
mkissue E0.8 "Streaming DataLoaders with normalisation and augmentation"            "type:infra,area:data,P0"      M "E0.6"        "On-the-fly normalisation, augmentation and disk streaming."
mkissue E0.9 "Deterministic train/val/test splitter with regime stratification"    "type:infra,area:data,P0"      S "E0.6"        "Reproducible splits stratified by flow regime."

EPIC="E0.C Experiment & evaluation harness"
ACC="Any experiment reproducible from one config plus seed; leaderboard updates automatically from tracked runs."
mkissue E0.10 "Config management (Hydra) + seed control"                            "type:infra,area:mlops,P0"     M ""            "Hydra configs and global seeding for reproducibility."
mkissue E0.11 "Experiment tracking (W&B or MLflow) in the train loop"               "type:infra,area:mlops,P0"     S "E0.10"       "Log metrics, configs and artefacts per run."
mkissue E0.12 "Metric library (rel-L2, spectra, conservation, rollout horizon)"     "type:eval,P0"                 M ""            "Shared metrics used by every model family."
mkissue E0.13 "Baseline registry + auto-generated leaderboard"                      "type:eval,type:docs,P1"       M "E0.11, E0.12" "Central comparison surface across models."
mkissue E0.14 "Model registry + checkpoint versioning + ONNX export"                "type:infra,area:mlops,P1"     M ""            "Version, store and export trained models."

EPIC="E0.D Compute & CI for ML"
ACC="sbatch template trains a smoke model end-to-end; CI runs overfit tests on every PR."
mkissue E0.15 "SLURM job templates for per-node training"                           "type:infra,P1"                S ""            "Ready-to-use sbatch templates for mjolnir and legion (standalone nodes)."
mkissue E0.16 "GPU-memory profiler + batch-size autotuner (16 GB)"                  "type:infra,P2"                S "E0.15"       "Fit batch sizes to the A2000 budget automatically."
mkissue E0.17 "CI smoke tests (tiny-model overfit per family)"                      "type:infra,type:eval,P1"      M ""            "Regression guard: each family overfits a tiny sample in CI."

# ===== M1 =====
MS="M1 - Surrogates & Neural Operators"

EPIC="E1.A Neural operators"
ACC="FNO beats the interpolation baseline on held-out Re; rollout stable to target horizon on Taylor-Green."
mkissue E1.1 "FNO (2D) baseline, single-step field prediction"                      "type:feature,area:operators,P0" M "E0.8, E0.12" "First operator surrogate; proves the training pipeline end-to-end."
mkissue E1.2 "Factorised/tensorised FNO for 3D within VRAM"                          "type:feature,area:operators,P1" L "E1.1"      "Scale operators to 3D under the memory budget."
mkissue E1.3 "DeepONet (branch/trunk) for parametric operator learning"             "type:feature,area:operators,P1" M "E0.8"      "Learn solution operators across parameters."
mkissue E1.4 "Geometry-aware operator (GINO / graph-FNO)"                           "type:research,area:operators,P2" L "E1.1"    "Operators on irregular domains."
mkissue E1.5 "Transformer operator (Transolver-style) baseline"                     "type:research,area:operators,P2" L "E1.1"    "Attention-based operator learning."
mkissue E1.6 "Autoregressive rollout wrapper + stability regularisation"            "type:feature,area:operators,P0" M "E1.1"     "Stable multi-step rollout via pushforward / noise injection."

EPIC="E1.B Reduced-order / latent-dynamics models"
ACC="Latent stepper reconstructs a full-field rollout under the error budget at target speed-up."
mkissue E1.7  "POD/DMD baseline ROM + reconstruction metrics"                       "type:feature,area:rom,P1"     S "E0.12"       "Classical ROM baseline."
mkissue E1.8  "Convolutional autoencoder for field compression"                     "type:feature,area:rom,P1"     M "E0.8"        "Learn a compact latent field representation."
mkissue E1.9  "Latent time-stepper (Transformer/GRU)"                               "type:feature,area:rom,P1"     M "E1.8"        "Advance the dynamics in latent space."
mkissue E1.10 "Neural-ODE latent dynamics variant"                                  "type:research,area:rom,P2"    L "E1.8"        "Continuous-time latent dynamics."
mkissue E1.11 "Koopman-based linear latent operator"                                "type:research,area:rom,P2"    L "E1.8"        "Linear latent evolution (ties to discovery)."

EPIC="E1.C Super-resolution / upscaling"
ACC="Super-resolution recovers spectra beyond the coarse cutoff without violating continuity beyond tolerance."
mkissue E1.12 "CNN super-resolution (coarse -> fine)"                               "type:feature,area:operators,P1" M "E0.8"     "Learned upscaling of coarse fields."
mkissue E1.13 "Physics-consistent SR (divergence-free / conservation)"             "type:research,area:closures,P2" M "E1.12"    "Constrain super-resolution to respect physics."
mkissue E1.14 "Diffusion-based SR variant"                                          "type:research,area:generative,P2" L "E4.1"  "Generative super-resolution (links to M4)."

# ===== M2 =====
MS="M2 - Hybrid & Physics-Constrained ML"

EPIC="E2.A Physics-informed family (PINNs and beyond)"
ACC="PINN reproduces the Von Karman regime; comparison table populated across families."
mkissue E2.1 "Baseline PINN, ADE then NS (fp-carleman staging)"                     "type:feature,P1"              M "E0.10"       "Physics-informed baseline: advection-diffusion first, then Navier-Stokes."
mkissue E2.2 "Hard-constraint architectures (BC/divergence built in)"              "type:research,P2"             M "E2.1"        "Bake boundary and divergence constraints into the network."
mkissue E2.3 "Causal/curriculum training + loss balancing (NTK/gradnorm)"          "type:research,P2"             M "E2.1"        "Improve PINN trainability."
mkissue E2.4 "Head-to-head: PINN vs operator vs ROM"                                "type:eval,P1"                 M "E1.1, E2.1"  "Fair comparison across families on identical cases."

EPIC="E2.B Learned closures (solver-in-the-loop) - the moat"
ACC="A-posteriori-trained closure keeps a coarse run accurate over target horizon and beats a-priori on stability."
mkissue E2.5 "Pluggable ML collision/closure module (solver interface)"             "type:feature,area:closures,area:integration,P0" M "E0.1" "ML closure conforming to the swappable collision interface shared with the quantum track."
mkissue E2.6 "A-priori closure training (offline targets)"                          "type:feature,area:closures,P1" M "E2.5, E0.8" "Train the closure on offline targets."
mkissue E2.7 "A-posteriori training (backprop through K solver steps)"              "type:research,area:closures,area:integration,P0" L "E2.5, E0.2" "Train the closure in the loop through the differentiable solver."
mkissue E2.8 "Learned LES/subgrid closure for under-resolved LBM"                   "type:research,area:closures,P1" L "E2.7"     "Data-driven subgrid model."
mkissue E2.9 "Stability + OOD generalisation audit of learned closures"             "type:eval,area:closures,P0"   M "E2.7"        "Test closures beyond the training Re range."

EPIC="E2.C Hybrid ML-corrected solvers"
ACC="Hybrid solver reaches target accuracy at measured wall-clock speed-up with bounded error."
mkissue E2.10 "Residual/correction model on the solver update"                      "type:feature,area:integration,P1" M "E2.5"   "Learned defect correction."
mkissue E2.11 "ML-accelerated timestepping (predict-then-correct, guarded)"         "type:research,area:integration,P2" L "E2.10" "Speed up stepping with a fallback guard."
mkissue E2.12 "Solver/ML switching policy + error monitor"                          "type:feature,area:integration,P2" M "E2.10"  "Decide when to trust ML versus the solver."

# ===== M3 =====
MS="M3 - Uncertainty, Inverse & Data Assimilation"

EPIC="E3.A Uncertainty quantification"
ACC="Predictive intervals calibrated (coverage within tolerance) on held-out regimes; UQ report auto-generated."
mkissue E3.1 "Deep-ensembles wrapper (generic)"                                     "type:feature,area:uq,P0"      M "E0.14"       "Ensembling across any model family."
mkissue E3.2 "MC-dropout + heteroscedastic/evidential heads"                        "type:feature,area:uq,P1"      M "E3.1"        "Cheap aleatoric and epistemic estimates."
mkissue E3.3 "Bayesian NN (variational/Laplace) operator variant"                  "type:research,area:uq,P2"     L "E1.1"        "Posterior over operator weights."
mkissue E3.4 "Conformal prediction (calibrated field-wise bars)"                    "type:research,area:uq,P1"     M "E3.1"        "Distribution-free calibrated intervals."
mkissue E3.5 "UQ calibration + sharpness benchmark (reliability, CRPS)"             "type:eval,area:uq,P0"         M "E3.1"        "Measure calibration quality."

EPIC="E3.B Inverse problems (through the differentiable solver)"
ACC="Recovers known parameters/geometry from synthetic observations within the error budget."
mkissue E3.6 "Gradient-based parameter inversion (viscosity/forcing)"               "type:feature,area:inverse,P0" M "E0.2"        "Invert parameters through the solver."
mkissue E3.7 "Geometry/topology inversion (differentiable masks)"                   "type:research,area:inverse,P1" L "E3.6"      "Recover geometry via differentiable masks."
mkissue E3.8 "Bayesian inversion (SVGD/HMC)"                                        "type:research,area:inverse,area:uq,P2" L "E3.6, E3.1" "Posterior over parameters."

EPIC="E3.C Data assimilation"
ACC="DA reduces state error versus a no-assimilation baseline on a twin experiment."
mkissue E3.9  "4D-Var-style DA using the solver adjoint"                            "type:research,area:inverse,P1" L "E0.2"      "Variational data assimilation."
mkissue E3.10 "Ensemble-Kalman / ML-hybrid DA"                                      "type:research,area:inverse,area:uq,P2" L "E3.1, E3.9" "Ensemble data assimilation with ML."
mkissue E3.11 "Active-learning loop (uncertainty-driven querying)"                  "type:research,area:uq,area:data,P2" M "E3.5, E0.5" "Query the solver where the model is most uncertain."

# ===== M4 =====
MS="M4 - Generative, Graph & Spatiotemporal"

EPIC="E4.A Generative models"
ACC="Diffusion surrogate produces physically plausible fields (spectra, conservation) with calibrated spread."
mkissue E4.1 "Conditional diffusion model (generation/inpainting)"                  "type:research,area:generative,P1" L "E0.8"   "Generative flow-field model."
mkissue E4.2 "Parameter-conditioned diffusion surrogate"                            "type:research,area:generative,P2" L "E4.1"   "Sample fields given Re and BCs."
mkissue E4.3 "Flow-matching/score-based variant + sampling-speed study"             "type:research,area:generative,P2" L "E4.1"   "Faster generative sampling."
mkissue E4.4 "GAN/VAE baselines + data augmentation"                                "type:feature,area:generative,P2" M "E0.8"    "Comparison baselines and data augmentation."
mkissue E4.5 "Generative UQ (ensemble-of-samples calibration)"                      "type:eval,area:generative,area:uq,P2" M "E4.1, E3.5" "Calibrate generative spread."

EPIC="E4.B Graph neural simulators (unstructured / particle, ties to LBM-DEM)"
ACC="GNN simulator matches solver trajectories over horizon on a particle test case."
mkissue E4.6 "GNN mesh/particle data adapter (from LBM-DEM)"                        "type:infra,area:graph,area:data,P1" M "E0.6" "Convert particle/mesh state to graphs."
mkissue E4.7 "MeshGraphNet-style GNN simulator baseline"                            "type:research,area:graph,P1"  L "E4.6"        "Learned unstructured simulator."
mkissue E4.8 "Rollout stability + noise-injection training (GNN)"                   "type:research,area:graph,P1"  M "E4.7"        "Stable GNN rollouts."
mkissue E4.9 "GNN for particle-laden flow (hydrocyclone)"                           "type:research,area:graph,P2"  L "E4.7"        "Apply the GNN simulator to the hydrocyclone regime."

EPIC="E4.C Spatiotemporal sequence models"
ACC="Best sequence model extends the stable-rollout horizon versus the autoregressive operator baseline."
mkissue E4.10 "ConvLSTM/attention multi-step forecasting baseline"                  "type:feature,area:sequence,P2" M "E0.8"      "Sequence-model baseline."
mkissue E4.11 "Spatiotemporal transformer for long rollouts"                        "type:research,area:sequence,P2" L "E4.10"    "Long-horizon forecasting."
mkissue E4.12 "Long-horizon rollout benchmark across families"                      "type:eval,area:sequence,P1"   M "E1.6, E4.11" "Compare rollout horizons across families."

# ===== M5 =====
MS="M5 - Discovery, Control & Foundation Models"

EPIC="E5.A Equation / closure discovery"
ACC="Recovers known governing terms on canonical cases; discovered closure runs inside the solver."
mkissue E5.1 "SINDy sparse-regression for effective equations"                      "type:research,area:discovery,P1" M "E0.5"    "Discover governing terms from data."
mkissue E5.2 "Symbolic regression for interpretable closures"                       "type:research,area:discovery,area:closures,P2" L "E2.8" "Human-readable closures."
mkissue E5.3 "Discovery validation vs known limits (report)"                        "type:eval,area:discovery,type:docs,P2" M "E5.1" "Validate and document discovered forms."

EPIC="E5.B Reinforcement learning: control & adaptivity"
ACC="RL policy improves the control objective/cost versus a fixed baseline on the target case."
mkissue E5.4 "RL environment wrapping the differentiable solver (Gym)"              "type:infra,area:rl,area:integration,P1" M "E0.1" "Gym-API environment on the solver."
mkissue E5.5 "RL for active flow control (drag/mixing)"                             "type:research,area:rl,P2"     L "E5.4"        "Learn control policies."
mkissue E5.6 "RL/learned policy for adaptive Carleman truncation (bake-off vs fp-carleman PINN)" "type:research,area:rl,P1" L "E5.4" "Alternative to the PINN truncation-order predictor; compare head-to-head."
mkissue E5.7 "RL for adaptive timestep/mesh/solver-switching"                       "type:research,area:rl,area:integration,P2" L "E5.4, E2.12" "Learn adaptivity policies."

EPIC="E5.C Multi-fidelity, transfer & foundation models"
ACC="Pretrained-then-finetuned model beats from-scratch on a low-data target regime."
mkissue E5.8  "Multi-fidelity model (cheap+expensive fusion)"                       "type:research,area:foundation,P2" L "E0.7"   "Fuse cheap and expensive data."
mkissue E5.9  "Transfer/fine-tuning across regimes and geometries"                  "type:research,area:foundation,P2" L "E1.2"   "Cross-regime transfer."
mkissue E5.10 "PDE foundation-model pretraining (multi-regime corpus)"              "type:research,area:foundation,P2" XL "E0.6, E1.2" "Large pretraining. Compute-gated: needs external HPC, not the homelab."
mkissue E5.11 "Fine-tune foundation model to hydrocyclone/particle regime"          "type:research,area:foundation,P2" L "E5.10"  "Adapt the pretrained model to the target regime."

echo ">> Done."
[[ -z "$PROJECT_NUMBER" ]] && echo "   (PROJECT_NUMBER was empty; issues created but not added to a Project board.)"
