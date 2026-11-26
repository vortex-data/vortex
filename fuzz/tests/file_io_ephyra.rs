// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `file_io` fuzz target as ephyra properties: a [`FileIoPlan`] is written to a Vortex file,
//! scanned back with a projection and a filter, and compared with the same expressions applied in
//! memory. The three tests mirror `array_ops_ephyra.rs`: `file_io` is a random search,
//! `file_io_corpus` the coverage-guided search CI runs, and `file_io_plan` replays plan files.

#![cfg(test)]

use ephyra::TestResult;
use vortex_fuzz::FileIoPlan;
use vortex_fuzz::ephyra_driver::PlanOutcome;
use vortex_fuzz::ephyra_driver::check_plan;
use vortex_fuzz::ephyra_driver::dtype_kind;
use vortex_fuzz::ephyra_driver::run_corpus;
use vortex_fuzz::ephyra_driver::run_plans;
use vortex_fuzz::run_file_io;

#[ephyra::test(tests = 200)]
fn file_io(plan: FileIoPlan) -> TestResult {
    check(plan)
}

#[test]
#[ignore = "coverage-guided search over EPHYRA_CORPUS_DIR, run by CI"]
fn file_io_corpus() {
    run_corpus("file_io_corpus", check);
}

#[test]
#[ignore = "replays the plans that EPHYRA_PLAN names"]
fn file_io_plan() {
    run_plans(check);
}

fn check(plan: FileIoPlan) -> TestResult {
    check_plan(&plan, || {
        let Some(action) = plan.materialize() else {
            return PlanOutcome::Discard("the plan's bytes do not decode to an array");
        };
        ephyra::label(format!("dtype: {}", dtype_kind(action.array.dtype())));
        ephyra::label(format!(
            "scan: {} projection, {} filter",
            if action.projection_expr.is_some() {
                "with"
            } else {
                "no"
            },
            if action.filter_expr.is_some() {
                "with"
            } else {
                "no"
            },
        ));
        ephyra::feature("array_len", action.array.len() as f64);
        ephyra::cover(action.array.is_empty(), "empty array");
        if run_file_io(action) {
            PlanOutcome::Pass
        } else {
            PlanOutcome::Discard("the file format does not support the array's dtype")
        }
    })
}
