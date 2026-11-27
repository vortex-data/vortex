// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `array_ops` fuzz target as ephyra properties; see `vortex_fuzz::ephyra`.
//!
//! - `array_ops`: a random search. Pull requests run it with a fixed seed, and the scheduled CI
//!   search with `EPHYRA_BUDGET_MS` and a fresh seed per worker. Replay a failure with its
//!   `EPHYRA_REPLAY=seed:iteration` line.
//! - `array_ops_plan`: replays the plan files that `VORTEX_FUZZ_PLAN` names, such as the crash files the
//!   searches write to `VORTEX_FUZZ_CRASH_DIR`.

#![cfg(test)]

use vortex_fuzz::ephyra::ChoicePlan;
use vortex_fuzz::ephyra::array_ops::check;
use vortex_fuzz::ephyra::run_plans;
use vortex_fuzz::ephyra::run_random;

#[test]
fn array_ops() {
    run_random::<ChoicePlan, _>("array_ops", 500, check);
}

#[test]
#[ignore = "replays the plans that VORTEX_FUZZ_PLAN names"]
fn array_ops_plan() {
    run_plans(check);
}
