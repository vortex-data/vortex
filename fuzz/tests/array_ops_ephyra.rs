// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `array_ops` fuzz target as an ephyra property.
//!
//! Inputs are [`ArrayOpsPlan`]s rather than raw bytes, so a failure shrinks to a short list of
//! actions on a small array. Set `EPHYRA_TESTS`, `EPHYRA_BUDGET_MS` or `EPHYRA_SEED` to change the
//! run, and replay a reported failure with its `EPHYRA_REPLAY=seed:iteration` line. Failures are
//! saved under `fuzz/.ephyra/regressions/` and replayed first on the next run.

#![cfg(test)]

use ephyra::TestResult;
use vortex_array::dtype::DType;
use vortex_fuzz::ActionType;
use vortex_fuzz::ArrayOpsPlan;
use vortex_fuzz::run_fuzz_action;

#[ephyra::test(tests = 500)]
fn array_ops(plan: ArrayOpsPlan) -> TestResult {
    let Some(action) = plan.materialize() else {
        return TestResult::discard("no action applies to the generated array");
    };
    ephyra::label(format!("dtype: {}", dtype_kind(action.array.dtype())));
    ephyra::feature("array_len", action.array.len() as f64);
    ephyra::feature("actions", action.actions.len() as f64);
    for (step, _) in &action.actions {
        ephyra::label(format!("action: {:?}", ActionType::from(step)));
    }
    ephyra::cover(action.array.is_empty(), "empty array");
    ephyra::cover(action.array.len() > 256, "array longer than 256");
    match run_fuzz_action(action) {
        Ok(_) => TestResult::Pass,
        Err(e) => TestResult::fail(e.to_string()),
    }
}

fn dtype_kind(dtype: &DType) -> String {
    match dtype {
        DType::Null => "null".into(),
        DType::Bool(_) => "bool".into(),
        DType::Primitive(ptype, _) => ptype.to_string(),
        DType::Decimal(..) => "decimal".into(),
        DType::Utf8(_) => "utf8".into(),
        DType::Binary(_) => "binary".into(),
        DType::List(..) => "list".into(),
        DType::FixedSizeList(..) => "fixed_size_list".into(),
        DType::Struct(..) => "struct".into(),
        DType::Map(..) => "map".into(),
        DType::Union(..) => "union".into(),
        DType::Variant(_) => "variant".into(),
        DType::Extension(_) => "extension".into(),
    }
}
