# ISA Threshold Finder - Implementation TODO

## Overview

This tracks implementation progress against the design in `plan/`. The goal is a system that:
1. Benchmarks algorithm variants with tunable parameters
2. Computes stats from input data (len, density, etc.)
3. Finds crossover points where one variant becomes faster
4. Supports two modes: `bench()` (quick) and `search()` (full grid)

---

## Phase 1: Measurement Quality ✅ COMPLETE

**Priority: HIGH** - Foundation for accurate benchmarks

- [x] **1.1 Implement `Measurer` struct**
  - Location: `vortex-threshold-traits/src/measure.rs`
  - [x] `black_box` on BOTH inputs AND outputs
  - [x] Warmup phase (~100ms default)
  - [x] Batched iterations (~100µs per batch)
  - [x] Input generation outside timed section
  - [x] Drops outside timed section

- [x] **1.2 Implement `BatchSize` enum**
  - [x] `Small`, `Large`, `PerIteration`, `NumIterations(usize)`

- [x] **1.3 Statistical analysis**
  - [x] IQR outlier removal
  - [x] Bootstrap confidence intervals (95%)
  - [x] `MeasurementResult` struct with median, mean, stddev, CI bounds, sample count

---

## Phase 2: Stats-Based API ✅ COMPLETE

**Priority: HIGH** - Core API redesign

- [x] **2.1 Define `Stats` concept**
  - Location: `vortex-threshold-traits/src/stats.rs`
  - [x] `StatsPoint` - dynamic stats container with named dimensions
  - [x] Stats computed from Data via user-provided closure

- [x] **2.2 Implement `StatsGrid`**
  - [x] Builder for multi-dimensional stats grids
  - [x] `.dimension("len", Scale::log2(6, 20))`
  - [x] `.dimension("density", Scale::steps(0.0, 1.0, 0.1))`
  - [x] Iterator over all stats combinations (`StatsGridIter`)

- [x] **2.3 Implement `Scale` enum**
  - Location: `vortex-threshold-traits/src/scale.rs`
  - [x] `Scale::log2`, `Scale::log`, `Scale::linear`, `Scale::steps`, `Scale::explicit`

- [x] **2.4 Implement `StatsBench` builder**
  - Location: `vortex-threshold-traits/src/bench.rs`
  - [x] `.stats()`, `.generate()`, `.stats_grid()`, `.baseline()`, `.variant()`

---

## Phase 3: Per-Variant Parameters

**Priority: MEDIUM** - Enables parameter tuning per variant

- [ ] **3.1 Design `ParamGrid` trait**
- [ ] **3.2 Implement `ParamGrid` derive macro** (new crate)
- [ ] **3.3 Update `StatsBench` with `.variant_with_params::<P>()`**

---

## Phase 4: Two Modes (Bench vs Search) ⚠️ PARTIAL

- [x] **4.1 `BenchRunner`** - `.at()`, `.run()`, `.print()`
- [x] **4.2 `SearchRunner`** - grid search, winners detection
- [x] **4.3 Entry points** - `.bench()`, `.search()`
- [ ] `.refine()` - binary search refinement
- [ ] Crossover detection

---

## Phase 5: Output & Display ⚠️ PARTIAL

- [x] Basic terminal output with print()
- [ ] Divan-style aligned columns
- [ ] JSON export (`.save()`, `.to_json()`)
- [ ] Comparison mode

---

## Phase 6-7: Refinement, Storage & CI

- [ ] Binary search refinement
- [ ] SQLite storage
- [ ] Code generation
- [ ] CI workflow

---

## Current Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Measurement | **DONE** | `Measurer`, `BatchSize`, IQR, bootstrap CI |
| 2. Stats API | **DONE** | `Scale`, `StatsGrid`, `StatsBench` builder |
| 3. ParamGrid | Not started | No derive macro yet |
| 4. Two Modes | **Partial** | Basic BenchRunner/SearchRunner work |
| 5. Output | **Partial** | Basic print(), no JSON/save |
| 6. Refinement | Not started | |
| 7. Storage/CI | Not started | Trait exists only |

---

## Next Steps

1. ~~**Phase 1** (Measurement) - DONE~~
2. ~~**Phase 2** (Stats API) - DONE~~
3. **Phase 3** (ParamGrid derive) - enables tunable per-variant params
4. **Phase 5** (Output) - JSON export, better terminal display
