// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use rstest::fixture;
use rstest::rstest;
use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::vtable::NotSupported;
use crate::array::vtable::ValidityVTable;
use crate::array::vtable::with_empty_buffers;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::Filter;
use crate::arrays::FilterArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::operators::Operator;
use crate::serde::ArrayChildren;
use crate::session::ArraySession;
use crate::test_harness::trace::TraceOptions;
use crate::test_harness::trace::TraceResolution;
use crate::test_harness::trace::trace_op;
use crate::test_harness::trace::trace_op_with;
use crate::validity::Validity;

#[fixture]
fn stack_parent_fixture() -> VortexResult<ArrayRef> {
    stack_parent(stack_child()?)
}

/// Build a session with the `StackChild` parent kernels registered.
///
/// The declining kernel is registered first so strict trace snapshots can assert that both
/// session kernels are attempted in registration order.
#[fixture]
fn stack_parent_session() -> VortexSession {
    let session = VortexSession::empty().with::<ArraySession>();
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(StackParent.id(), StackChild, StackDeclineKernel);
    kernels.register_execute_parent_kernel(StackParent.id(), StackChild, StackParentKernel);
    drop(kernels);
    session
}

fn stack_child() -> VortexResult<ArrayRef> {
    Ok(
        Array::try_from_parts(ArrayParts::new(StackChild, test_dtype(), 3, StackChildData))?
            .into_array(),
    )
}

fn stack_parent(child: ArrayRef) -> VortexResult<ArrayRef> {
    Ok(Array::try_from_parts(
        ArrayParts::new(
            StackParent,
            child.dtype().clone(),
            child.len(),
            StackParentData,
        )
        .with_slots(smallvec![Some(child)]),
    )?
    .into_array())
}

fn test_dtype() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

#[derive(Clone, Debug)]
struct StackParent;

#[derive(Clone, Debug)]
struct StackParentData;

impl Display for StackParentData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("stack-parent")
    }
}

impl ArrayHash for StackParentData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
}

impl ArrayEq for StackParentData {
    fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
        true
    }
}

impl ValidityVTable<StackParent> for StackParent {
    fn validity(_array: ArrayView<'_, StackParent>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for StackParent {
    type TypedArrayData = StackParentData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.stack-parent");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype == &test_dtype(), "unexpected stack parent dtype");
        vortex_ensure!(len == 3, "unexpected stack parent length");
        vortex_ensure!(slots.len() == 1, "stack parent must have one child slot");
        let Some(child) = &slots[0] else {
            vortex_bail!("stack parent child slot is missing");
        };
        vortex_ensure!(child.dtype() == dtype, "stack parent child dtype mismatch");
        vortex_ensure!(child.len() == len, "stack parent child length mismatch");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StackParent buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("StackParent cannot be deserialized")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("StackParent slot index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let Some(child) = array.slots()[0].as_ref() else {
            vortex_bail!("stack parent child slot is missing");
        };
        if !child.is::<Primitive>() {
            return Ok(ExecutionResult::execute_slot::<Primitive>(array, 0));
        }

        Ok(ExecutionResult::done(child.clone()))
    }
}

#[derive(Clone, Debug)]
struct StackChild;

#[derive(Clone, Debug)]
struct StackChildData;

impl Display for StackChildData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("stack-child")
    }
}

impl ArrayHash for StackChildData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _eq_mode: EqMode) {}
}

impl ArrayEq for StackChildData {
    fn array_eq(&self, _other: &Self, _eq_mode: EqMode) -> bool {
        true
    }
}

impl ValidityVTable<StackChild> for StackChild {
    fn validity(_array: ArrayView<'_, StackChild>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl VTable for StackChild {
    type TypedArrayData = StackChildData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.stack-child");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype == &test_dtype(), "unexpected stack child dtype");
        vortex_ensure!(len == 3, "unexpected stack child length");
        vortex_ensure!(slots.is_empty(), "stack child must not have slots");
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StackChild buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("StackChild cannot be deserialized")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("StackChild slot index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        debug_assert!(array.slots().is_empty());
        Ok(ExecutionResult::done(PrimitiveArray::from_iter([
            99i32, 99, 99,
        ])))
    }
}

#[derive(Debug)]
struct StackDeclineKernel;

impl ExecuteParentKernel<StackChild> for StackDeclineKernel {
    type Parent = StackParent;

    fn execute_parent(
        &self,
        _array: ArrayView<'_, StackChild>,
        _parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(None)
    }
}

#[derive(Debug)]
struct StackParentKernel;

impl ExecuteParentKernel<StackChild> for StackParentKernel {
    type Parent = StackParent;

    fn execute_parent(
        &self,
        _array: ArrayView<'_, StackChild>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent
            .slots()
            .get(child_idx)
            .is_some_and(|slot| slot.is_none())
        {
            return Ok(Some(PrimitiveArray::from_iter([1i32, 2, 3]).into_array()));
        }

        Ok(None)
    }
}

#[test]
fn trace_optimize_reduce_fixpoint() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter([0i32, 1, 2, 3]).into_array();
    let filter = FilterArray::try_new(values.clone(), Mask::new_true(values.len()))?.into_array();

    let traced = trace_op(|| filter.optimize())?;

    assert!(traced.output.is::<Primitive>());
    assert_arrays_eq!(traced.output, values, &mut execution_ctx());
    insta::assert_snapshot!(traced.trace.to_string(), @r"
optimize root=vortex.filter(i32, len=4) session=false
  reduce TrivialFilterRule: vortex.filter(i32, len=4) -> vortex.primitive(i32, len=4)
  done output=vortex.primitive(i32, len=4)
");

    Ok(())
}

#[test]
fn trace_optimize_parent_reduce_fixpoint_attempts() -> VortexResult<()> {
    let values = PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5]).into_array();
    let inner = FilterArray::try_new(
        values,
        Mask::from_iter([true, false, true, true, false, true]),
    )?
    .into_array();
    let outer =
        FilterArray::try_new(inner, Mask::from_iter([false, true, true, false]))?.into_array();

    let traced = trace_op_with(
        TraceOptions {
            resolution: TraceResolution::ExecutedOnly,
        },
        || outer.optimize(),
    )?;

    let optimized_filter = traced.output.as_::<Filter>();
    assert!(optimized_filter.child().is::<Primitive>());
    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([2i32, 3]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @r"
    optimize root=vortex.filter(i32, len=2) session=false
      reduce_parent static:FilterReduceAdaptor(Filter) slot=0 parent=vortex.filter(i32, len=2) child=vortex.filter(i32, len=4) -> vortex.filter(i32, len=2)
      done output=vortex.filter(i32, len=2)
    ");

    let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());
    let traced = trace_op_with(
        TraceOptions {
            resolution: TraceResolution::ExecutedOnly,
        },
        || outer.execute::<Canonical>(&mut ctx),
    )?;

    insta::assert_snapshot!(traced.trace.to_string(), @r"
    execute_until target=AnyCanonical root=vortex.filter(i32, len=2)
      iter 0 current=vortex.filter(i32, len=2) builder_active=false
        ExecuteSlot slot=0 parent=vortex.filter(i32, len=2) child=vortex.filter(i32, len=4)
      iter 1 current=vortex.filter(i32, len=4) stack_parent=vortex.filter(i32, len=2) slot=0 builder_active=false
        Done array=vortex.primitive(i32, len=4)
      iter 2 current=vortex.primitive(i32, len=4) stack_parent=vortex.filter(i32, len=2) slot=0 builder_active=false
        pop_frame slot=0 output=vortex.filter(i32, len=2)
      iter 3 current=vortex.filter(i32, len=2) builder_active=false
        Done array=vortex.primitive(i32, len=2)
      iter 4 current=vortex.primitive(i32, len=2) builder_active=false
      return output=vortex.primitive(i32, len=2)
    ");

    Ok(())
}

#[rstest]
fn trace_execution_stack_parent_kernel_attempts(
    stack_parent_fixture: VortexResult<ArrayRef>,
    stack_parent_session: VortexSession,
) -> VortexResult<()> {
    let mut ctx = ExecutionCtx::new(stack_parent_session);
    let parent = stack_parent_fixture?;

    let traced = trace_op_with(
        TraceOptions {
            resolution: TraceResolution::Attempts,
        },
        || parent.execute::<PrimitiveArray>(&mut ctx),
    )?;

    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([1i32, 2, 3]),
        &mut ctx
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.test.stack-parent(i32, len=3)
      iter 0 current=vortex.test.stack-parent(i32, len=3) builder_active=false
        done_check target=false canonical=false
        child_execute_parent attempt slot=0 parent=vortex.test.stack-parent(i32, len=3) child=vortex.test.stack-child(i32, len=3) source=session[0] kernel=execute_parent_fn outcome=declined
        child_execute_parent attempt slot=0 parent=vortex.test.stack-parent(i32, len=3) child=vortex.test.stack-child(i32, len=3) source=session[1] kernel=execute_parent_fn outcome=declined
        child_execute_parent none current=vortex.test.stack-parent(i32, len=3)
        execute encoding=vortex.test.stack-parent(i32, len=3)
        ExecuteSlot slot=0 parent=vortex.test.stack-parent(i32, len=3) child=vortex.test.stack-child(i32, len=3)
      iter 1 current=vortex.test.stack-child(i32, len=3) stack_parent=vortex.test.stack-parent(i32, len=3) slot=0 builder_active=false
        done_check target=false canonical=false
        stack_execute_parent attempt slot=0 parent=vortex.test.stack-parent(i32, len=3) child=vortex.test.stack-child(i32, len=3) source=session[0] kernel=execute_parent_fn outcome=declined
        stack_execute_parent applied slot=0 parent=vortex.test.stack-parent(i32, len=3) child=vortex.test.stack-child(i32, len=3) source=session[1] kernel=execute_parent_fn output=vortex.primitive(i32, len=3)
    optimize root=vortex.primitive(i32, len=3) session=true
      loop input=vortex.primitive(i32, len=3)
        reduce none array=vortex.primitive(i32, len=3)
        reduce_parent none array=vortex.primitive(i32, len=3)
      done output=vortex.primitive(i32, len=3) changed=false
        optimize_ctx input=vortex.primitive(i32, len=3) output=vortex.primitive(i32, len=3) changed=false
      iter 2 current=vortex.primitive(i32, len=3) builder_active=false
        done_check target=true canonical=true
      return output=vortex.primitive(i32, len=3)
    ");

    Ok(())
}

#[test]
fn trace_execution_chunked_append_child_flow() -> VortexResult<()> {
    let chunks = vec![
        PrimitiveArray::from_iter([1i32, 2]).into_array(),
        PrimitiveArray::from_iter([3i32]).into_array(),
        PrimitiveArray::from_iter([4i32, 5]).into_array(),
    ];
    let dtype = chunks[0].dtype().clone();
    let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());

    let traced = trace_op(|| {
        chunked
            .execute::<Canonical>(&mut ctx)
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]),
        &mut ctx
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.chunked(i32, len=5)
      iter 0 current=vortex.chunked(i32, len=5) builder_active=false
        builder start array=vortex.chunked(i32, len=5)
        AppendChild slot=1 parent=vortex.chunked(i32, len=5) child=vortex.primitive(i32, len=2)
        builder append child=vortex.primitive(i32, len=2)
      iter 1 current=vortex.chunked(i32, len=5) builder_active=true
        AppendChild slot=2 parent=vortex.chunked(i32, len=5) child=vortex.primitive(i32, len=1)
        builder append child=vortex.primitive(i32, len=1)
      iter 2 current=vortex.chunked(i32, len=5) builder_active=true
        AppendChild slot=3 parent=vortex.chunked(i32, len=5) child=vortex.primitive(i32, len=2)
        builder append child=vortex.primitive(i32, len=2)
      iter 3 current=vortex.chunked(i32, len=5) builder_active=true
        Done array=vortex.primitive(i32, len=0)
        builder finish output=vortex.primitive(i32, len=5)
      iter 4 current=vortex.primitive(i32, len=5) builder_active=false
      return output=vortex.primitive(i32, len=5)
    ");

    Ok(())
}

/// A dictionary of strings: codes `[0, 1, 2, 1, 0, 2]` over values
/// `["alpha", "beta", "gamma"]`.
fn dict_of_strings() -> VortexResult<ArrayRef> {
    Ok(DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 1, 2, 1, 0, 2]).into_array(),
        VarBinViewArray::from_iter_str(["alpha", "beta", "gamma"]).into_array(),
    )?
    .into_array())
}

fn execution_ctx() -> ExecutionCtx {
    ExecutionCtx::new(VortexSession::empty().with::<ArraySession>())
}

#[test]
fn trace_take_on_chunked() -> VortexResult<()> {
    let chunked = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_iter([10i32, 11]).into_array(),
            PrimitiveArray::from_iter([12i32, 13, 14]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();

    // A take is expressed as a `DictArray` whose codes are the take indices.
    let indices = PrimitiveArray::from_iter([4u64, 0, 2, 2]).into_array();
    let take = DictArray::try_new(indices, chunked)?.into_array();

    // No reduce rule rewrites a take over a chunked array: the work all happens at execution
    // time, so the optimizer trace is empty.
    let traced = trace_op(|| take.optimize())?;
    insta::assert_snapshot!(traced.trace.to_string(), @"");

    let optimized = traced.output;
    let traced = trace_op(|| {
        optimized
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        PrimitiveArray::from_iter([14i32, 10, 12, 12]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.dict(i32, len=4)
      iter 0 current=vortex.dict(i32, len=4) builder_active=false
    execute_until target=AnyCanonical root=vortex.chunked(i32, len=4)
      iter 0 current=vortex.chunked(i32, len=4) builder_active=false
        builder start array=vortex.chunked(i32, len=4)
        AppendChild slot=1 parent=vortex.chunked(i32, len=4) child=vortex.dict(i32, len=1)
        builder append child=vortex.dict(i32, len=1)
      iter 1 current=vortex.chunked(i32, len=4) builder_active=true
        AppendChild slot=2 parent=vortex.chunked(i32, len=4) child=vortex.dict(i32, len=3)
        builder append child=vortex.dict(i32, len=3)
      iter 2 current=vortex.chunked(i32, len=4) builder_active=true
        Done array=vortex.primitive(i32, len=0)
        builder finish output=vortex.primitive(i32, len=4)
      iter 3 current=vortex.primitive(i32, len=4) builder_active=false
      return output=vortex.primitive(i32, len=4)
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(i32, len=4) child=vortex.chunked(i32, len=5) -> vortex.dict(i32, len=4)
      iter 1 current=vortex.dict(i32, len=4) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(i32, len=4) child=vortex.primitive(i32, len=4) -> vortex.primitive(i32, len=4)
      iter 2 current=vortex.primitive(i32, len=4) builder_active=false
      return output=vortex.primitive(i32, len=4)
    ");

    Ok(())
}

#[test]
fn trace_filter_on_struct_with_complex_children() -> VortexResult<()> {
    let names = DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 1, 2, 1, 0]).into_array(),
        VarBinViewArray::from_iter_str(["alpha", "beta", "gamma"]).into_array(),
    )?
    .into_array();
    let scores = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_iter([1i64, 2]).into_array(),
            PrimitiveArray::from_iter([3i64, 4, 5]).into_array(),
        ],
        DType::Primitive(PType::I64, Nullability::NonNullable),
    )?
    .into_array();
    let struct_ = StructArray::from_fields(&[("name", names), ("score", scores)])?.into_array();

    let filtered =
        FilterArray::try_new(struct_, Mask::from_iter([true, false, true, true, false]))?
            .into_array();

    let traced = trace_op(|| filtered.optimize())?;
    insta::assert_snapshot!(traced.trace.to_string(), @"
    optimize root=vortex.filter({name=utf8, score=i64}, len=3) session=false
      optimize root=vortex.filter(utf8, len=3) session=false
        reduce_parent static:FilterReduceAdaptor(Dict) slot=0 parent=vortex.filter(utf8, len=3) child=vortex.dict(utf8, len=5) -> vortex.dict(utf8, len=3)
        done output=vortex.dict(utf8, len=3)
      reduce FilterStructRule: vortex.filter({name=utf8, score=i64}, len=3) -> vortex.struct({name=utf8, score=i64}, len=3)
      done output=vortex.struct({name=utf8, score=i64}, len=3)
    ");

    let optimized = traced.output;
    let traced = trace_op(|| {
        optimized
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    let expected = StructArray::from_fields(&[
        (
            "name",
            VarBinViewArray::from_iter_str(["alpha", "gamma", "beta"]).into_array(),
        ),
        (
            "score",
            PrimitiveArray::from_iter([1i64, 3, 4]).into_array(),
        ),
    ])?
    .into_array();
    assert_arrays_eq!(traced.output, expected, &mut execution_ctx());
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.struct({name=utf8, score=i64}, len=3)
      iter 0 current=vortex.struct({name=utf8, score=i64}, len=3) builder_active=false
      return output=vortex.struct({name=utf8, score=i64}, len=3)
    ");

    Ok(())
}

#[test]
fn trace_compare_on_dict() -> VortexResult<()> {
    let dict = DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 1, 2, 1, 0]).into_array(),
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
    )?
    .into_array();
    let rhs = ConstantArray::new(Scalar::from(20i32), dict.len()).into_array();

    let compared = Binary.try_new_array(dict.len(), Operator::Eq, [dict, rhs])?;

    let traced = trace_op(|| compared.optimize())?;
    insta::assert_snapshot!(traced.trace.to_string(), @"
    optimize root=vortex.binary(bool, len=5) session=false
      reduce_parent static:DictionaryScalarFnValuesPushDownRule slot=0 parent=vortex.binary(bool, len=5) child=vortex.dict(i32, len=5) -> vortex.dict(bool, len=5)
      done output=vortex.dict(bool, len=5)
    ");

    let optimized = traced.output;
    let traced = trace_op(|| {
        optimized
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        BoolArray::from_iter([false, true, false, true, false]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.dict(bool, len=5)
      iter 0 current=vortex.dict(bool, len=5) builder_active=false
        ExecuteSlot slot=1 parent=vortex.dict(bool, len=5) child=vortex.binary(bool, len=3)
      iter 1 current=vortex.binary(bool, len=3) stack_parent=vortex.dict(bool, len=5) slot=1 builder_active=false
        Done array=vortex.bool(bool, len=3)
      iter 2 current=vortex.bool(bool, len=3) stack_parent=vortex.dict(bool, len=5) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.dict(bool, len=5)
      iter 3 current=vortex.dict(bool, len=5) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(bool, len=5) child=vortex.bool(bool, len=3) -> vortex.bool(bool, len=5)
      iter 4 current=vortex.bool(bool, len=5) builder_active=false
      return output=vortex.bool(bool, len=5)
    ");

    Ok(())
}

#[test]
fn trace_like_on_dict() -> VortexResult<()> {
    let strings = dict_of_strings()?;
    let pattern = ConstantArray::new(Scalar::from("b%"), strings.len()).into_array();

    let like = Like.try_new_array(
        strings.len(),
        LikeOptions {
            negated: false,
            case_insensitive: false,
        },
        [strings, pattern],
    )?;

    let traced = trace_op(|| like.optimize())?;
    insta::assert_snapshot!(traced.trace.to_string(), @"
    optimize root=vortex.like(bool, len=6) session=false
      reduce_parent static:LikeReduceAdaptor(Dict) slot=0 parent=vortex.like(bool, len=6) child=vortex.dict(utf8, len=6) -> vortex.dict(bool, len=6)
      done output=vortex.dict(bool, len=6)
    ");

    let optimized = traced.output;
    let traced = trace_op(|| {
        optimized
            .execute::<Canonical>(&mut execution_ctx())
            .map(IntoArray::into_array)
    })?;

    assert_arrays_eq!(
        traced.output,
        BoolArray::from_iter([false, true, false, true, false, false]),
        &mut execution_ctx()
    );
    insta::assert_snapshot!(traced.trace.to_string(), @"
    execute_until target=AnyCanonical root=vortex.dict(bool, len=6)
      iter 0 current=vortex.dict(bool, len=6) builder_active=false
        ExecuteSlot slot=1 parent=vortex.dict(bool, len=6) child=vortex.like(bool, len=3)
      iter 1 current=vortex.like(bool, len=3) stack_parent=vortex.dict(bool, len=6) slot=1 builder_active=false
        Done array=vortex.bool(bool, len=3)
      iter 2 current=vortex.bool(bool, len=3) stack_parent=vortex.dict(bool, len=6) slot=1 builder_active=false
        pop_frame slot=1 output=vortex.dict(bool, len=6)
      iter 3 current=vortex.dict(bool, len=6) builder_active=false
        child_execute_parent session[0]:execute_parent_fn slot=1 parent=vortex.dict(bool, len=6) child=vortex.bool(bool, len=3) -> vortex.bool(bool, len=6)
      iter 4 current=vortex.bool(bool, len=6) builder_active=false
      return output=vortex.bool(bool, len=6)
    ");

    Ok(())
}
