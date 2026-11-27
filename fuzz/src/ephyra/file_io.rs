// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `file_io` target as an ephyra property: [`check`] runs the libFuzzer target's round trip on
//! the case a [`ChoicePlan`] generates.

use ephyra::TestResult;

use super::ChoicePlan;
use super::PlanOutcome;
use super::check_plan;
use super::dtype_kind;
use crate::FuzzFileAction;
use crate::run_file_io;

/// Generate a `file_io` case from `plan`, write it to a Vortex file, and compare the scan with the
/// same expressions applied in memory.
pub fn check(plan: ChoicePlan) -> TestResult {
    check_plan(&plan, || {
        let Ok(action) = FuzzFileAction::generate(&mut plan.source()) else {
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
