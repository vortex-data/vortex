// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `array_ops` target as an ephyra property: [`check`] runs the libFuzzer target's checks on
//! the case a [`ChoicePlan`] generates.

use ephyra::TestResult;

use super::ChoicePlan;
use super::PlanOutcome;
use super::check_plan;
use super::dtype_kind;
use crate::FuzzArrayAction;
use crate::array::ActionType;
use crate::error::VortexFuzzError;
use crate::run_fuzz_action;

/// Generate an `array_ops` case from `plan` and check it.
///
/// Failures are signed by error kind, failing action and dtype, such as
/// `array mismatch after Slice on u16`, so each bug files one issue.
pub fn check(plan: ChoicePlan) -> TestResult {
    check_plan(&plan, || {
        let Ok(action) = FuzzArrayAction::generate(&mut plan.source()) else {
            return PlanOutcome::Discard("the plan generates no applicable action");
        };
        let dtype = dtype_kind(action.array.dtype());
        let steps = action
            .actions
            .iter()
            .map(|(step, _)| ActionType::from(step))
            .collect::<Vec<_>>();
        ephyra::label(format!("dtype: {dtype}"));
        ephyra::feature("array_len", action.array.len() as f64);
        ephyra::feature("actions", action.actions.len() as f64);
        for step in &steps {
            ephyra::label(format!("action: {step:?}"));
        }
        ephyra::cover(action.array.is_empty(), "empty array");
        ephyra::cover(action.array.len() > 256, "array longer than 256");
        let case = format!(
            "{} array of length {}, actions {steps:?}",
            action.array.dtype(),
            action.array.len()
        );
        match run_fuzz_action(action) {
            Ok(_) => PlanOutcome::Pass,
            Err(e) => {
                // The full message holds arrays and a backtrace; group failures by kind and step.
                let (kind, step) = error_kind(&e);
                let step = step.and_then(|i| steps.get(i));
                PlanOutcome::Fail {
                    signature: match step {
                        Some(step) => format!("{kind} after {step:?} on {dtype}"),
                        None => format!("{kind} on {dtype}"),
                    },
                    message: format!("{case}\n{e}"),
                }
            }
        }
    })
}

/// The error's variant and, when it has one, the index of the step that failed.
fn error_kind(error: &VortexFuzzError) -> (&'static str, Option<usize>) {
    match error {
        VortexFuzzError::ScalarMismatch(.., step, _) => ("scalar mismatch", Some(*step)),
        VortexFuzzError::SearchSortedError(.., step, _) => ("search_sorted mismatch", Some(*step)),
        VortexFuzzError::MinMaxMismatch(.., step, _) => ("min_max mismatch", Some(*step)),
        VortexFuzzError::ArrayNotEqual(.., step, _) => ("array mismatch", Some(*step)),
        VortexFuzzError::DTypeMismatch(.., step, _) => ("dtype mismatch", Some(*step)),
        VortexFuzzError::LengthMismatch(.., step, _) => ("length mismatch", Some(*step)),
        VortexFuzzError::VortexError(..) => ("vortex error", None),
    }
}
