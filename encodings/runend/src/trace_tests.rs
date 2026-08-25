// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Snapshot tests tracing how reductions and executions flow through run-end arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::test_harness::trace::trace_op;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::RunEnd;

fn execution_ctx() -> ExecutionCtx {
    let session = VortexSession::empty().with::<ArraySession>();
    // Execute-parent kernels live in the session registry; without registering RunEnd's own
    // kernels the traces below would show plain canonicalization instead of the pushdowns.
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

#[test]
fn trace_compare_on_runend() -> VortexResult<()> {
    let runend = runend_array()?;
    let rhs = ConstantArray::new(Scalar::from(2i32), runend.len()).into_array();
    let compared = Binary::try_new(runend, rhs, Operator::Eq)?.into_array();

    let traced = trace_op(|| compared.optimize())?;
    insta::assert_snapshot!(traced.trace.to_string(), @"
    optimize root=vortex.binary(bool, len=9) session=false
      reduce_parent static:RunEndScalarFnRule slot=0 parent=vortex.binary(bool, len=9) child=vortex.runend(i32, len=9) -> vortex.runend(bool, len=9)
      done output=vortex.runend(bool, len=9)
    ");

    let optimized = traced.output;
    let traced = trace_op(|| {
        optimized
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        BoolArray::from_iter([false, false, false, true, true, false, false, false, false]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.runend(bool, len=9)
      iter 0 current=vortex.runend(bool, len=9) builder_active=false
    execute_until target=AnyCanonical root=vortex.binary(bool, len=3)
      iter 0 current=vortex.binary(bool, len=3) builder_active=false
        Done array=vortex.bool(bool, len=3)
      iter 1 current=vortex.bool(bool, len=3) builder_active=false
      return output=vortex.bool(bool, len=3)
        Done array=vortex.bool(bool, len=9)
      iter 1 current=vortex.bool(bool, len=9) builder_active=false
      return output=vortex.bool(bool, len=9)
    ");

    Ok(())
}

#[test]
fn trace_filter_on_runend() -> VortexResult<()> {
    let runend = runend_array()?;
    let filtered = FilterArray::try_new(
        runend,
        Mask::from_iter([true, false, false, true, true, false, false, true, false]),
    )?
    .into_array();

    let traced = trace_op(|| {
        filtered
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([1i32, 2, 2, 3]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.filter(i32, len=4)
      iter 0 current=vortex.filter(i32, len=4) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=0 parent=vortex.filter(i32, len=4) child=vortex.runend(i32, len=9) -> vortex.dict(i32, len=4)
      iter 1 current=vortex.dict(i32, len=4) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(i32, len=4) child=vortex.primitive(i32, len=3) -> vortex.primitive(i32, len=4)
      iter 2 current=vortex.primitive(i32, len=4) builder_active=false
      return output=vortex.primitive(i32, len=4)
    ");

    Ok(())
}

#[test]
fn trace_take_on_runend() -> VortexResult<()> {
    let runend = runend_array()?;
    // A take is expressed as a `DictArray` whose codes are the take indices.
    let indices = PrimitiveArray::from_iter([8u64, 0, 4, 4]).into_array();
    let take = DictArray::try_new(indices, runend)?.into_array();

    let traced = trace_op(|| {
        take.execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([3i32, 1, 2, 2]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.dict(i32, len=4)
      iter 0 current=vortex.dict(i32, len=4) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(i32, len=4) child=vortex.runend(i32, len=9) -> vortex.dict(i32, len=4)
      iter 1 current=vortex.dict(i32, len=4) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(i32, len=4) child=vortex.primitive(i32, len=3) -> vortex.primitive(i32, len=4)
      iter 2 current=vortex.primitive(i32, len=4) builder_active=false
      return output=vortex.primitive(i32, len=4)
    ");

    Ok(())
}
