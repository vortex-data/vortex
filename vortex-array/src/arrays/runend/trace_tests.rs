// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Snapshot tests tracing how reductions and executions flow through run-end arrays.

use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::FilterArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::SliceArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::runend::RunEnd;
use crate::assert_arrays_eq;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::lower::Lower;
use crate::scalar_fn::fns::operators::Operator;
use crate::session::ArraySession;
use crate::test_harness::trace::trace_op;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn execution_ctx() -> ExecutionCtx {
    ExecutionCtx::new(VortexSession::empty().with::<ArraySession>())
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

    let traced = trace_op(|| compared.optimize_ctx(&SESSION))?;
    insta::assert_snapshot!(traced.trace.to_string(), @"
    optimize root=vortex.binary(bool, len=9)
      reduce_parent session[0]:RunEndScalarFnRule slot=0 parent=vortex.binary(bool, len=9) child=vortex.runend(i32, len=9) -> vortex.runend(bool, len=9)
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
        ExecuteSlot slot=1 parent=vortex.runend(bool, len=9) child=vortex.binary(bool, len=3)
      iter 1 current=vortex.binary(bool, len=3) stack_parent=vortex.runend(bool, len=9) slot=1 builder_active=false
        Done array=vortex.bool(bool, len=3)
      iter 2 current=vortex.bool(bool, len=3) stack_parent=vortex.runend(bool, len=9) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.runend(bool, len=9)
      iter 3 current=vortex.runend(bool, len=9) builder_active=false
        Done array=vortex.bool(bool, len=9)
      iter 4 current=vortex.bool(bool, len=9) builder_active=false
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

#[test]
fn lower_on_sliced_runend_reaches_dict_values() -> VortexResult<()> {
    // Rows: ["Great", "Great", "LAKES", "Great", "Great", null].
    let dict = DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 1, 0, 2]).into_array(),
        VarBinViewArray::from_iter_nullable_str([Some("Great"), Some("LAKES"), None]).into_array(),
    )?
    .into_array();
    let runend = RunEnd::try_new(
        PrimitiveArray::from_iter([2u32, 3, 5, 6]).into_array(),
        dict,
        &mut execution_ctx(),
    )?
    .into_array();
    let sliced = SliceArray::new(runend, 1..5).into_array();
    let lowered = ScalarFnArray::try_new(Lower.bind(EmptyOptions), [sliced])?.into_array();

    let optimized = lowered.optimize_ctx(&SESSION)?;
    insta::assert_snapshot!(optimized.display_tree_encodings_only().to_string(), @"
    root: vortex.runend(utf8?, len=4)
      ends: vortex.primitive(u32, len=3)
      values: vortex.dict(utf8?, len=3)
        codes: vortex.primitive(u32, len=3)
        values: vortex.lower(utf8?, len=3)
          input: vortex.varbinview(utf8?, len=3)
            validity: vortex.bool(bool, len=3)
    ");

    assert_arrays_eq!(
        optimized,
        VarBinViewArray::from_iter_nullable_str([
            Some("great"),
            Some("lakes"),
            Some("great"),
            Some("great")
        ]),
        &mut execution_ctx()
    );
    Ok(())
}
