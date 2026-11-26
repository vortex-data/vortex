// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `array_ops` fuzz target as ephyra properties; see `vortex_fuzz::ephyra`.
//!
//! - `array_ops`: a random search, the fixed-seed check that pull requests run. Set
//!   `EPHYRA_TESTS`, `EPHYRA_BUDGET_MS` or `EPHYRA_SEED`, and replay a failure with its
//!   `EPHYRA_REPLAY=seed:iteration` line.
//! - `array_ops_corpus`: the coverage-guided search over `EPHYRA_CORPUS_DIR` that CI runs.
//! - `array_ops_plan`: replays the plan files that `EPHYRA_PLAN` names, such as the crash files the
//!   searches write to `EPHYRA_CRASH_DIR`.

#![cfg(test)]

use vortex_fuzz::ephyra::array_ops::ArrayOpsPlan;
use vortex_fuzz::ephyra::array_ops::check;
use vortex_fuzz::ephyra::run_corpus;
use vortex_fuzz::ephyra::run_plans;
use vortex_fuzz::ephyra::run_random;

#[test]
fn array_ops() {
    run_random::<ArrayOpsPlan, _>("array_ops", 500, check);
}

#[test]
#[ignore = "coverage-guided search over EPHYRA_CORPUS_DIR, run by CI"]
fn array_ops_corpus() {
    run_corpus("array_ops_corpus", check);
}

#[test]
#[ignore = "replays the plans that EPHYRA_PLAN names"]
fn array_ops_plan() {
    run_plans(check);
}
