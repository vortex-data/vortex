// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `file_io` fuzz target as ephyra properties; see `vortex_fuzz::ephyra`. The tests mirror
//! `array_ops_ephyra.rs`: `file_io` is the random search that pull requests and the scheduled CI
//! search run, and `file_io_plan` replays plan files.

#![cfg(test)]

use vortex_fuzz::ephyra::ChoicePlan;
use vortex_fuzz::ephyra::file_io::check;
use vortex_fuzz::ephyra::run_plans;
use vortex_fuzz::ephyra::run_random;

#[test]
fn file_io() {
    run_random::<ChoicePlan, _>("file_io", 200, check);
}

#[test]
#[ignore = "replays the plans that VORTEX_FUZZ_PLAN names"]
fn file_io_plan() {
    run_plans(check);
}
