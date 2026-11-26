// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `array_ops` fuzz target as ephyra properties.
//!
//! Inputs are [`ArrayOpsPlan`]s rather than raw bytes, so a failure shrinks to a short list of
//! actions on a small array.
//!
//! - `array_ops` is a random search: set `EPHYRA_TESTS`, `EPHYRA_BUDGET_MS` or `EPHYRA_SEED`, and
//!   replay a failure with its `EPHYRA_REPLAY=seed:iteration` line.
//! - `array_ops_corpus` is the coverage-guided search over `EPHYRA_CORPUS_DIR` that CI runs; see
//!   `vortex_fuzz::ephyra_driver`.
//! - `array_ops_plan` replays the plan files that `EPHYRA_PLAN` names, such as the crash files
//!   that the searches write to `EPHYRA_CRASH_DIR`.

#![cfg(test)]

use ephyra::Config;
use ephyra::TestResult;
use vortex_fuzz::ActionType;
use vortex_fuzz::ArrayOpsPlan;
use vortex_fuzz::ephyra_driver::PlanOutcome;
use vortex_fuzz::ephyra_driver::check_plan;
use vortex_fuzz::ephyra_driver::dtype_kind;
use vortex_fuzz::ephyra_driver::finish;
use vortex_fuzz::ephyra_driver::run_corpus;
use vortex_fuzz::ephyra_driver::run_plans;
use vortex_fuzz::error::VortexFuzzError;
use vortex_fuzz::run_fuzz_action;

#[test]
fn array_ops() {
    let config = Config::default().name("array_ops").tests(500);
    finish(ephyra::check(&config, ephyra::any::<ArrayOpsPlan>(), check));
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

fn check(plan: ArrayOpsPlan) -> TestResult {
    check_plan(&plan, || {
        let Some(action) = plan.materialize() else {
            return PlanOutcome::Discard("no action applies to the generated array");
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
                    message: e.to_string(),
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
        _ => ("other error", None),
    }
}
