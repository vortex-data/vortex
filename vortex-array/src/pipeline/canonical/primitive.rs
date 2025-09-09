// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, BufferMut};
use vortex_dtype::{NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::PrimitiveArray;
use crate::pipeline::bits::{BitAlignedChunkedIterator, BitVector, BitView, BitViewMut, MaskSliceIterator};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N, N_WORDS};
use crate::validity::Validity;

pub(super) fn export_primitive_nonnull<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut remaining = len;
    while remaining >= N {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let mask = BitVector::true_until(remaining);
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

pub(super) fn export_primitive_nonnull2<T, Mask>(
    mut mask_iter: Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray>

where T: Element + NativePType,
    Mask: MaskSliceIterator,
{
    let len = mask_iter.len();
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };
    let mut size = 0;

    let mut remaining = len;
    while remaining >= 0 {
        let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
        let dummy_ctx = KernelContext::default();
        // TODO(joe): have iter return true count.
        let view = BitView::new(mask_iter.next_chunk().vortex_expect("mask iterator"));
        size += view.true_count();
        pipeline.step(&dummy_ctx, view, &mut elements_view)?;
        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(len) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

pub(super) fn export_primitive_null<T: Element + NativePType>(
    len: usize,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut mask =
        BufferMut::<usize>::full(0, len.div_ceil(N_WORDS) * N_WORDS).aligned(Alignment::new(1024));

    let mut remaining = len;

    while remaining >= N {
        let head = len - remaining;
        let slice: &mut [usize; N_WORDS] =
            unsafe { extract_step_slice(&mut (mask[head / (u32::BITS as usize)..][..N_WORDS])) };
        let val_view = BitViewMut::new(slice);
        let mut elements_view = ViewMut::new(&mut elements[head..][..N], Some(val_view));
        let dummy_ctx = KernelContext::default();
        pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
        remaining -= N;
    }

    if remaining > 0 {
        let head = len - remaining;
        let slice: &mut [usize; N_WORDS] =
            unsafe { extract_step_slice(&mut mask[head / (u32::BITS as usize)..][..N_WORDS]) };

        let val_view = BitViewMut::new(slice);
        let mut elements_view = ViewMut::new(&mut elements[head..][..N], Some(val_view));
        let mask = BitVector::true_until(remaining);
        let dummy_ctx = KernelContext::default();

        pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
    }

    unsafe { elements.set_len(len) };

    let abuf = arrow_buffer::Buffer::from(mask.freeze().into_inner());
    let buf = BooleanBuffer::new(abuf, 0, len);
    let mask = Mask::from_buffer(buf);
    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::from_mask(mask, Nullability::Nullable),
    ))
}

unsafe fn extract_step_slice(slice: &mut [usize]) -> &mut [usize; N_WORDS] {
    unsafe { &mut *(slice.as_mut_ptr() as *mut [usize; N_WORDS]) }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    
    use arrow_buffer::BooleanBuffer;
    use vortex_dtype::PType;
    
    use super::*;
    use crate::pipeline::bits::BitAlignedChunkedIterator;

    struct StepCountingKernel {
        step_count: Rc<RefCell<usize>>,
    }

    impl StepCountingKernel {
        fn new() -> (Self, Rc<RefCell<usize>>) {
            let counter = Rc::new(RefCell::new(0));
            (
                StepCountingKernel {
                    step_count: counter.clone(),
                },
                counter,
            )
        }
    }

    impl Kernel for StepCountingKernel {
        fn step(
            &mut self,
            _ctx: &KernelContext,
            _selected: BitView,
            _out: &mut ViewMut,
        ) -> VortexResult<()> {
            *self.step_count.borrow_mut() += 1;
            Ok(())
        }
    }

    #[test]
    fn test_export_primitive_nonnull2_step_calls() {
        // Test various sizes to verify step call counts
        let test_cases = [
            (512, 1),    // Less than N (1024), should call step once
            (1024, 1),   // Exactly N, should call step once
            (1536, 2),   // More than N but less than 2*N, should call step twice
            (2048, 2),   // Exactly 2*N, should call step twice
            (3000, 3),   // Should call step three times
        ];

        for (total_bits, expected_steps) in test_cases {
            // Create a boolean buffer with all true values
            let buffer = BooleanBuffer::new_set(total_bits);
            let mask_iter = BitAlignedChunkedIterator::new(&buffer);

            let (mut kernel, step_counter) = StepCountingKernel::new();
            
            let _result = export_primitive_nonnull2::<u32, _>(mask_iter, &mut kernel);
            
            let actual_steps = *step_counter.borrow();
            assert_eq!(
                actual_steps, expected_steps,
                "For {} bits, expected {} steps but got {}",
                total_bits, expected_steps, actual_steps
            );
        }
    }
}

pub(super) fn export_primitive_nonnull_masked<T: Element + NativePType>(
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    let len = mask.len();
    let capacity = mask.true_count().next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mask_buffer = mask.to_boolean_buffer();
    let mut mask_iter = BitAlignedChunkedIterator::new(&mask_buffer);

    let mut offset = 0;
    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[offset..][..N], None);

        let dummy_ctx = KernelContext::default();
        let mask_view = BitView::new(&mask_iter.next_chunk().vortex_expect("mask iterator"));
        pipeline.step(&dummy_ctx, mask_view, &mut elements_view)?;
        offset += mask_view.true_count();

        remaining = remaining.saturating_sub(N);
    }

    assert_eq!(mask.true_count(), offset);

    unsafe { elements.set_len(offset) };

    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}
