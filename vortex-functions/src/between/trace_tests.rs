// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Snapshot tests tracing how `between` executes over run-end arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::trace::trace_op;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::between::Between;
use crate::between::BetweenOptions;
use crate::between::StrictComparison;

fn execution_ctx() -> ExecutionCtx {
    let session = VortexSession::empty().with::<ArraySession>();
    crate::initialize(&session);
    ExecutionCtx::new(session)
}

/// Run-end encoding of `[1, 1, 1, 2, 2, 3, 3, 3, 3]`: ends `[3, 5, 9]`, values `[1, 2, 3]`.
fn runend_array() -> VortexResult<ArrayRef> {
    Ok(RunEnd::encode(
        PrimitiveArray::from_iter([1i32, 1, 1, 2, 2, 3, 3, 3, 3]).into_array(),
        &mut execution_ctx(),
    )?
    .into_array())
}

/// Executing a slice over a run-end array yields the run-end array, whose parent-reduce rule then
/// pushes the suspended between into the run values instead of decoding every row.
#[test]
fn trace_between_on_sliced_runend() -> VortexResult<()> {
    let sliced = runend_array()?.slice(1..8)?;
    let between = Between::try_new(
        sliced,
        ConstantArray::new(2i32, 7).into_array(),
        ConstantArray::new(3i32, 7).into_array(),
        BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::Strict,
        },
    )?
    .into_array();

    let traced = trace_op(|| {
        between
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        BoolArray::from_iter([false, false, true, true, false, false, false]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.between(bool, len=7)
      iter 0 current=vortex.between(bool, len=7) builder_active=false
        ExecuteSlot slot=0 parent=vortex.between(bool, len=7) child=vortex.slice(i32, len=7)
      iter 1 current=vortex.slice(i32, len=7) stack_parent=vortex.between(bool, len=7) slot=0 builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.slice(i32, len=7) child=vortex.runend(i32, len=9) -> vortex.runend(i32, len=7)
      iter 2 current=vortex.runend(i32, len=7) stack_parent=vortex.between(bool, len=7) slot=0 builder_active=false
        reduce_parent session[0]:RunEndScalarFnRule slot=0 parent=vortex.between(bool, len=7) child=vortex.runend(i32, len=7) -> vortex.runend(bool, len=7)
      iter 3 current=vortex.runend(bool, len=7) builder_active=false
        ExecuteSlot slot=1 parent=vortex.runend(bool, len=7) child=vortex.between(bool, len=3)
      iter 4 current=vortex.between(bool, len=3) stack_parent=vortex.runend(bool, len=7) slot=1 builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.between(bool, len=3) child=vortex.primitive(i32, len=3) -> vortex.bool(bool, len=3)
      iter 5 current=vortex.bool(bool, len=3) stack_parent=vortex.runend(bool, len=7) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.runend(bool, len=7)
      iter 6 current=vortex.runend(bool, len=7) builder_active=false
        Done array=vortex.bool(bool, len=7)
      iter 7 current=vortex.bool(bool, len=7) builder_active=false
      return output=vortex.bool(bool, len=7)
    ");

    Ok(())
}
