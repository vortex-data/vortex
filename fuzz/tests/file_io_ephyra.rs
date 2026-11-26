// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `file_io` fuzz target as ephyra properties; see `vortex_fuzz::ephyra`. The tests mirror
//! `array_ops_ephyra.rs`: `file_io` is a random search, `file_io_corpus` the coverage-guided search
//! CI runs, and `file_io_plan` replays plan files.

#![cfg(test)]

use vortex_fuzz::ephyra::file_io::FileIoPlan;
use vortex_fuzz::ephyra::file_io::check;
use vortex_fuzz::ephyra::run_corpus;
use vortex_fuzz::ephyra::run_plans;
use vortex_fuzz::ephyra::run_random;

#[test]
fn file_io() {
    run_random::<FileIoPlan, _>("file_io", 200, check);
}

#[test]
#[ignore = "coverage-guided search over VORTEX_FUZZ_CORPUS_DIR, run by CI"]
fn file_io_corpus() {
    run_corpus("file_io_corpus", check);
}

#[test]
#[ignore = "replays the plans that VORTEX_FUZZ_PLAN names"]
fn file_io_plan() {
    run_plans(check);
}
