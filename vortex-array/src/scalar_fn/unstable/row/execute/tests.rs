// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Verifies ownership of RowFn output allocations independently of input decoding.

use std::mem::MaybeUninit;
use std::ops::BitOrAssign;

use rstest::rstest;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::DenseAttempt;
use super::execute_bool_dense_attempt;
use super::execute_owned;
use super::execute_owned_bool;
use super::execute_owned_dense_attempt;
use super::execute_owned_infallible;
use super::execute_owned_infallible_bool;
use super::execute_owned_infallible_filtered;
use super::execute_owned_infallible_valid_rows;
use super::execute_sink;
use super::execute_sink_filtered;
use super::execute_sink_valid_rows;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Bool;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::memory::MemorySessionExt;
use crate::memory::test_allocator::tracking_allocator;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::FailureEvidence;
use crate::scalar_fn::unstable::row::FixedSizeListSink;
use crate::scalar_fn::unstable::row::InitializedElement;
use crate::scalar_fn::unstable::row::OutputBuffer;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::SinkResult;
use crate::scalar_fn::unstable::row::UninitElementSink;
use crate::scalar_fn::unstable::row::Utf8Sink;
use crate::validity::Validity;

#[derive(Clone, Copy)]
enum Traversal {
    Infallible,
    Fallible,
    DenseAttempt,
    Selected,
    Filtered,
}

fn collect_owned<Out: OutputElement, Fail: FailureEvidence>(
    traversal: Traversal,
    args: &VecExecutionArgs,
    valid: &MaskValuesRef,
    ctx: &mut ExecutionCtx,
    apply: impl Fn(i64) -> Out,
) -> VortexResult<ArrayRef> {
    match traversal {
        Traversal::Infallible => execute_owned_infallible::<(i64,), Out, ()>(
            args,
            ctx,
            |_| (),
            |_, (value,)| apply(value),
        ),
        Traversal::Fallible => execute_owned::<(i64,), Out, (), Fail>(
            args,
            ctx,
            |_| (),
            |_, (value,)| (apply(value), Fail::default()),
            |_| Ok(()),
        ),
        Traversal::DenseAttempt => {
            match execute_owned_dense_attempt::<(i64,), Out, (), Fail>(
                args,
                ctx,
                |_| (),
                |_, (value,)| (apply(value), Fail::default()),
                |_| Ok(()),
            )? {
                DenseAttempt::Values(values) => Ok(values),
                DenseAttempt::DeferredError(error) => Err(error),
            }
        }
        Traversal::Selected => execute_owned_infallible_valid_rows::<(i64,), Out, ()>(
            args,
            valid,
            ctx,
            |_| (),
            |_, (value,)| apply(value),
        )?
        .ok_or_else(|| vortex_err!("canonical input must support selected rows")),
        Traversal::Filtered => execute_owned_infallible_filtered::<(i64,), Out, ()>(
            args,
            valid,
            ctx,
            |_| (),
            |_, (value,)| apply(value),
        ),
    }
}

fn canonical_args(traversal: Traversal, constant: bool) -> VecExecutionArgs {
    let values = if matches!(traversal, Traversal::Filtered) {
        vec![2_i64, 4]
    } else {
        vec![2_i64, 3, 4]
    };
    let len = values.len();
    let input = if constant {
        ConstantArray::new(2_i64, len).into_array()
    } else {
        PrimitiveArray::from_iter(values).into_array()
    };
    VecExecutionArgs::new(vec![input], len)
}

fn selected_rows() -> MaskValuesRef {
    let Mask::Values(valid) = Mask::from_iter([true, false, true]) else {
        unreachable!("the test mask is partially valid");
    };
    valid
}

#[rstest]
#[case::infallible(Traversal::Infallible)]
#[case::fallible(Traversal::Fallible)]
#[case::dense_attempt(Traversal::DenseAttempt)]
#[case::selected(Traversal::Selected)]
#[case::filtered(Traversal::Filtered)]
fn owned_payload_uses_context_allocator(
    #[case] traversal: Traversal,
    #[values(false, true)] constant: bool,
    #[values(false, true)] boolean: bool,
) -> VortexResult<()> {
    // Prepare canonical inputs and the selection before creating either measured allocator.
    let args = canonical_args(traversal, constant);
    let valid = selected_rows();
    let (session_allocator, session_tracker) = tracking_allocator();
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .with_allocator(session_allocator)
        .create_execution_ctx()
        .with_allocator(allocator);

    let output = if boolean {
        collect_owned::<_, bool>(traversal, &args, &valid, &mut ctx, |value| value % 2 == 0)?
    } else {
        collect_owned::<_, bool>(traversal, &args, &valid, &mut ctx, |value| value)?
    };
    if boolean {
        tracker.assert_owns(output.as_::<Bool>().to_bit_buffer().inner().as_slice());
    } else {
        tracker.assert_owns(output.as_::<Primitive>().as_slice::<i64>());
    }
    assert_eq!(session_tracker.live_allocations(), 0);
    assert_eq!(tracker.live_allocations(), 1);
    drop(output);
    assert_eq!(tracker.live_allocations(), 0);
    Ok(())
}

#[rstest]
#[case::infallible(Traversal::Infallible)]
#[case::fallible(Traversal::Fallible)]
#[case::dense_attempt(Traversal::DenseAttempt)]
fn packed_boolean_payload_uses_allocator(
    #[case] traversal: Traversal,
    #[values(false, true)] multiversioned: bool,
) -> VortexResult<()> {
    let args = canonical_args(Traversal::Infallible, false);
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .create_execution_ctx()
        .with_allocator(allocator);

    let output = match (multiversioned, traversal) {
        (false, Traversal::Infallible) => {
            execute_owned_infallible_bool::<(i64,), false>(&args, &mut ctx, |(value,)| value > 2)?
        }
        (true, Traversal::Infallible) => {
            execute_owned_infallible_bool::<(i64,), true>(&args, &mut ctx, |(value,)| value > 2)?
        }
        (false, Traversal::Fallible) => execute_owned_bool::<(i64,), (), bool, false>(
            &args,
            &mut ctx,
            |_| (),
            |_, (value,)| (value > 2, false),
            |_| Ok(()),
        )?,
        (true, Traversal::Fallible) => execute_owned_bool::<(i64,), (), bool, true>(
            &args,
            &mut ctx,
            |_| (),
            |_, (value,)| (value > 2, false),
            |_| Ok(()),
        )?,
        (multiversioned, Traversal::DenseAttempt) => {
            let attempt = if multiversioned {
                execute_bool_dense_attempt::<(i64,), (), bool, true>(
                    &args,
                    &mut ctx,
                    |_| (),
                    |_, (value,)| (value > 2, false),
                    |_| Ok(()),
                )?
            } else {
                execute_bool_dense_attempt::<(i64,), (), bool, false>(
                    &args,
                    &mut ctx,
                    |_| (),
                    |_, (value,)| (value > 2, false),
                    |_| Ok(()),
                )?
            };
            match attempt {
                DenseAttempt::Values(values) => values,
                DenseAttempt::DeferredError(error) => return Err(error),
            }
        }
        _ => vortex_bail!("this test traversal requires packed Boolean output"),
    };
    tracker.assert_owns(output.as_::<Bool>().to_bit_buffer().inner().as_slice());
    assert_eq!(tracker.live_allocations(), 1);
    Ok(())
}

fn collect_sink<Sink, ApplyResult>(
    traversal: Traversal,
    args: &VecExecutionArgs,
    valid: &MaskValuesRef,
    params: &Sink::Params,
    ctx: &mut ExecutionCtx,
    apply: impl Fn(Sink::Row<'_>) -> ApplyResult,
) -> VortexResult<ArrayRef>
where
    Sink: OutputSink,
    ApplyResult: SinkResult<WriteToken = Sink::WriteToken>,
{
    match traversal {
        Traversal::Infallible => execute_sink::<(i64,), (), Sink, ApplyResult>(
            args,
            params,
            ctx,
            |_| (),
            |_, _, row| apply(row),
        ),
        Traversal::Selected => execute_sink_valid_rows::<(i64,), (), Sink, ApplyResult>(
            args,
            valid,
            params,
            ctx,
            |_| (),
            |_, _, row| apply(row),
        )?
        .ok_or_else(|| vortex_err!("canonical input must support selected rows")),
        Traversal::Filtered => execute_sink_filtered::<(i64,), (), Sink, ApplyResult>(
            args,
            valid,
            params,
            ctx,
            |_| (),
            |_, _, row| apply(row),
        ),
        _ => vortex_bail!("this test traversal requires an owned output"),
    }
}

#[rstest]
#[case::dense(Traversal::Infallible)]
#[case::selected(Traversal::Selected)]
#[case::filtered(Traversal::Filtered)]
fn sink_payloads_use_allocator(#[case] traversal: Traversal) -> VortexResult<()> {
    let args = canonical_args(traversal, false);
    let valid = selected_rows();
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .create_execution_ctx()
        .with_allocator(allocator);

    let scalar = collect_sink::<UninitElementSink<i64>, _>(
        traversal,
        &args,
        &valid,
        &(),
        &mut ctx,
        |row| {
            // SAFETY: writes the supplied slot and returns its token without modifying the slot.
            unsafe { InitializedElement::write(row, 42) }
        },
    )?;
    tracker.assert_owns(scalar.as_::<Primitive>().as_slice::<i64>());

    let strings = collect_sink::<Utf8Sink, _>(traversal, &args, &valid, &(), &mut ctx, |row| {
        row.write("an external UTF-8 payload")
    })?;
    let strings = strings.as_::<VarBinView>();
    tracker.assert_owns(strings.views());
    assert!(!strings.data_buffers().is_empty());
    for buffer in strings.data_buffers().iter() {
        tracker.assert_owns(buffer.as_host().as_slice());
    }
    Ok(())
}

#[test]
fn primitive_finish_reuses_allocation() {
    let (allocator, tracker) = tracking_allocator();
    let mut values = i64::allocate(3, &allocator);
    let slots = &mut values.slots()[..3];
    let ptr = slots.as_ptr().cast::<i64>();

    for (slot, value) in slots.iter_mut().zip([1, 2, 3]) {
        slot.write(value);
    }

    // SAFETY: all three slots were initialized above.
    let output = unsafe { values.finish(3, &allocator) };
    assert_eq!(output.as_::<Primitive>().as_slice::<i64>().as_ptr(), ptr);
    tracker.assert_owns(output.as_::<Primitive>().as_slice::<i64>());
}

#[test]
fn empty_outputs_and_zero_width_rows_do_not_allocate_payloads() -> VortexResult<()> {
    let empty = PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array();
    let args = VecExecutionArgs::new(vec![empty], 0);
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .create_execution_ctx()
        .with_allocator(allocator);

    let primitive =
        execute_owned_infallible::<(i64,), i64, ()>(&args, &mut ctx, |_| (), |_, (value,)| value)?;
    let boolean =
        execute_owned_infallible_bool::<(i64,), false>(&args, &mut ctx, |(value,)| value > 0)?;
    let scalar = execute_sink::<(i64,), (), UninitElementSink<i64>, _>(
        &args,
        &(),
        &mut ctx,
        |_| (),
        |_, _, row| {
            // SAFETY: writes the supplied slot and immediately returns its token.
            unsafe { InitializedElement::write(row, 0) }
        },
    )?;
    let mut lists = FixedSizeListSink::<i64>::with_capacity(3, &0, ctx.allocator())?;
    FixedSizeListSink::<i64>::initialize_skipped_rows(&mut lists.rows());
    // SAFETY: the skipped-row initializer completed for every zero-width row.
    let lists = unsafe { lists.finish() }?;
    let strings = execute_sink::<(i64,), (), Utf8Sink, _>(
        &args,
        &(),
        &mut ctx,
        |_| (),
        |_, _, row| row.write("unused"),
    )?;

    assert!(primitive.is_empty());
    assert!(boolean.is_empty());
    assert!(scalar.is_empty());
    assert!(strings.is_empty());
    assert_eq!(lists.len(), 3);
    assert!(lists.as_::<FixedSizeList>().elements().is_empty());
    assert_eq!(tracker.live_allocations(), 0);
    Ok(())
}

#[test]
fn boolean_sinks_allocate_packed_payloads_with_context_allocator() -> VortexResult<()> {
    let args = canonical_args(Traversal::Infallible, false);
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .create_execution_ctx()
        .with_allocator(allocator);

    let scalar = execute_sink::<(i64,), (), UninitElementSink<bool>, _>(
        &args,
        &(),
        &mut ctx,
        |_| (),
        |_, _, row| {
            // SAFETY: writes the supplied slot and immediately returns its token.
            unsafe { InitializedElement::write(row, true) }
        },
    )?;
    tracker.assert_owns(scalar.as_::<Bool>().to_bit_buffer().inner().as_slice());

    let mut lists = FixedSizeListSink::<bool>::with_capacity(3, &2, ctx.allocator())?;
    FixedSizeListSink::<bool>::initialize_skipped_rows(&mut lists.rows());
    // SAFETY: the skipped-row initializer initialized every element.
    let lists = unsafe { lists.finish() }?;
    tracker.assert_owns(
        lists
            .as_::<FixedSizeList>()
            .elements()
            .as_::<Bool>()
            .to_bit_buffer()
            .inner()
            .as_slice(),
    );
    assert_eq!(tracker.live_allocations(), 2);
    Ok(())
}

#[test]
fn fixed_size_list_payload_uses_allocator() -> VortexResult<()> {
    let (allocator, tracker) = tracking_allocator();
    let mut sink = FixedSizeListSink::<i64>::with_capacity(3, &2, &allocator)?;
    FixedSizeListSink::<i64>::initialize_skipped_rows(&mut sink.rows());
    // SAFETY: the skipped-row initializer initialized every element.
    let output = unsafe { sink.finish() }?;

    tracker.assert_owns(
        output
            .as_::<FixedSizeList>()
            .elements()
            .as_::<Primitive>()
            .as_slice::<i64>(),
    );
    assert_eq!(tracker.live_allocations(), 1);
    Ok(())
}

// Zero-sized outputs require failure evidence that is also zero-sized.
#[derive(Clone, Copy, Default)]
struct NoFailure;

impl BitOrAssign for NoFailure {
    fn bitor_assign(&mut self, _rhs: Self) {}
}

/// A zero-sized element whose collection storage does not depend on Vortex buffers.
#[derive(Clone, Copy, Default)]
struct One;

impl OutputElement for One {
    type Buffer = Vec<MaybeUninit<Self>>;

    fn element_dtype() -> DType {
        i64::element_dtype()
    }

    fn allocate(rows: usize, _allocator: &BufferAllocatorRef) -> Self::Buffer {
        vec![MaybeUninit::uninit(); rows]
    }
}

// SAFETY: the vector retains the same slots, and MaybeUninit permits partial initialization.
unsafe impl OutputBuffer<One> for Vec<MaybeUninit<One>> {
    fn slots(&mut self) -> &mut [MaybeUninit<One>] {
        self.as_mut_slice()
    }

    unsafe fn finish(self, len: usize, allocator: &BufferAllocatorRef) -> ArrayRef {
        let mut values = allocator.with_capacity(len);
        values.extend(std::iter::repeat_n(1_i64, len));
        PrimitiveArray::new(values.freeze(), Validity::NonNullable).into_array()
    }
}

#[rstest]
#[case::infallible(Traversal::Infallible)]
#[case::fallible(Traversal::Fallible)]
#[case::dense_attempt(Traversal::DenseAttempt)]
#[case::selected(Traversal::Selected)]
#[case::filtered(Traversal::Filtered)]
fn zero_sized_output_uses_its_own_storage(#[case] traversal: Traversal) -> VortexResult<()> {
    let args = canonical_args(traversal, false);
    let valid = selected_rows();
    let expected = PrimitiveArray::from_iter([1_i64; 3]).into_array();
    let (allocator, tracker) = tracking_allocator();
    let mut ctx = array_session()
        .create_execution_ctx()
        .with_allocator(allocator);

    let output = collect_owned::<One, NoFailure>(traversal, &args, &valid, &mut ctx, |_| One)?;
    assert_arrays_eq!(&output, &expected, &mut ctx);
    tracker.assert_owns(output.as_::<Primitive>().as_slice::<i64>());

    Ok(())
}
