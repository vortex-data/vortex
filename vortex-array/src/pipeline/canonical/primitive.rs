// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, BufferMut};
use vortex_dtype::{NativePType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::arrays::PrimitiveArray;
use crate::pipeline::bits::{BitAlignedChunkedIterator, BitView, BitViewMut, MaskSliceIterator};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N, N_WORDS};
use crate::validity::Validity;

// pub(super) fn export_primitive_nonnull<T: Element + NativePType>(
//     len: usize,
//     pipeline: &mut dyn Kernel,
// ) -> VortexResult<PrimitiveArray> {
//     let capacity = len.next_multiple_of(N) + N;
//
//     let mut elements = BufferMut::<T>::with_capacity(capacity);
//     unsafe { elements.set_len(capacity) };
//
//     let mut remaining = len;
//     while remaining >= N {
//         let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
//         let dummy_ctx = KernelContext::default();
//         pipeline.step(&dummy_ctx, BitView::all_true(), &mut elements_view)?;
//         remaining -= N;
//     }
//
//     if remaining > 0 {
//         let mut elements_view = ViewMut::new(&mut elements[len - remaining..][..N], None);
//         let mask = BitVector::true_until(remaining);
//         let dummy_ctx = KernelContext::default();
//         pipeline.step(&dummy_ctx, mask.as_view(), &mut elements_view)?;
//     }
//
//     unsafe { elements.set_len(len) };
//
//     Ok(PrimitiveArray::new(
//         elements.freeze(),
//         Validity::NonNullable,
//     ))
// }

pub(super) fn export_primitive_nonnull<T, Mask>(
    mut mask_iter: Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray>
where
    T: Element + NativePType,
    Mask: MaskSliceIterator,
{
    let len = mask_iter.len();
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };
    let mut element_count = 0;

    let mut remaining = len;
    while remaining > 0 {
        let mut elements_view = ViewMut::new(&mut elements[element_count..][..N], None);
        let dummy_ctx = KernelContext::default();
        // TODO(joe): have iter return true count.
        let view = BitView::new(mask_iter.next_chunk().vortex_expect("mask iterator"));
        element_count += view.true_count();
        pipeline.step(&dummy_ctx, view, &mut elements_view)?;
        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(element_count) };


    Ok(PrimitiveArray::new(
        elements.freeze(),
        Validity::NonNullable,
    ))
}

pub(super) fn export_primitive_null<T: Element + NativePType, MaskIter>(
    mut mask_iter: MaskIter,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray>
where
    T: Element + NativePType,
    MaskIter: MaskSliceIterator,
{
    let len = mask_iter.len();
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };

    let mut mask =
        BufferMut::<usize>::full(0, len.div_ceil(N_WORDS) * N_WORDS).aligned(Alignment::new(1024));

    let mut remaining = len;
    let mut element_count = 0;

    while remaining > 0 {
        let head = len - remaining;
        let slice: &mut [usize; N_WORDS] =
            unsafe { extract_step_slice(&mut (mask[head / (u32::BITS as usize)..][..N_WORDS])) };
        let val_view = BitViewMut::new(slice);
        let mut elements_view = ViewMut::new(&mut elements[element_count..][..N], Some(val_view));
        let dummy_ctx = KernelContext::default();
        // TODO(joe): have iter return true count.
        let view = BitView::new(mask_iter.next_chunk().vortex_expect("mask iterator"));
        element_count += view.true_count();
        pipeline.step(&dummy_ctx, view, &mut elements_view)?;
        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(element_count) };

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

    use super::*;

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

    /// Kernel that tracks both step calls and true counts
    struct TrueCountTrackingKernel {
        step_count: Rc<RefCell<usize>>,
        total_true_count: Rc<RefCell<usize>>,
    }

    impl TrueCountTrackingKernel {
        fn new() -> (Self, Rc<RefCell<usize>>, Rc<RefCell<usize>>) {
            let step_counter = Rc::new(RefCell::new(0));
            let true_counter = Rc::new(RefCell::new(0));
            (
                TrueCountTrackingKernel {
                    step_count: step_counter.clone(),
                    total_true_count: true_counter.clone(),
                },
                step_counter,
                true_counter,
            )
        }
    }

    impl Kernel for TrueCountTrackingKernel {
        fn step(
            &mut self,
            _ctx: &KernelContext,
            selected: BitView,
            _out: &mut ViewMut,
        ) -> VortexResult<()> {
            *self.step_count.borrow_mut() += 1;
            *self.total_true_count.borrow_mut() += selected.true_count();
            Ok(())
        }
    }

    /// Simple mock iterator for testing step counts
    struct MockMaskIterator {
        len: usize,
        calls: usize,
        max_calls: usize,
        buffer: Box<[usize; N_WORDS]>,
    }

    impl MockMaskIterator {
        fn new(len: usize) -> Self {
            let max_calls = len.div_ceil(N);
            Self {
                len,
                calls: 0,
                max_calls,
                buffer: Box::new([usize::MAX; N_WORDS]), // All bits set
            }
        }
    }

    impl MaskSliceIterator for MockMaskIterator {
        fn next_chunk(&mut self) -> Option<&[usize; N_WORDS]> {
            if self.calls < self.max_calls {
                self.calls += 1;
                Some(&*self.buffer)
            } else {
                None
            }
        }

        fn len(&self) -> usize {
            self.len
        }
    }

    #[test]
    fn test_export_primitive_nonnull_step_calls() {
        // Test various sizes to verify step call counts
        let test_cases = [
            (512, 1),  // Less than N (1024), should call step once
            (1024, 1), // Exactly N, should call step once
            (1536, 2), // More than N but less than 2*N, should call step twice
            (2048, 2), // Exactly 2*N, should call step twice
        ];

        for (total_bits, expected_steps) in test_cases {
            let mask_iter = MockMaskIterator::new(total_bits);
            let (mut kernel, step_counter) = StepCountingKernel::new();

            // Test the fixed version
            let result = export_primitive_nonnull::<u32, _>(mask_iter, &mut kernel);
            assert!(result.is_ok(), "Fixed function should not fail");

            let actual_steps = *step_counter.borrow();
            assert_eq!(
                actual_steps, expected_steps,
                "For {} bits, expected {} steps but got {}",
                total_bits, expected_steps, actual_steps
            );
        }
    }

    #[test]
    fn test_export_primitive_null_step_calls() {
        // Test various sizes to verify step call counts for nullable primitive export
        let test_cases = [
            (512, 1),  // Less than N (1024), should call step once
            (1024, 1), // Exactly N, should call step once
            (1536, 2), // More than N but less than 2*N, should call step twice
            (2048, 2), // Exactly 2*N, should call step twice
        ];

        for (total_bits, expected_steps) in test_cases {
            let mask_iter = MockMaskIterator::new(total_bits);
            let (mut kernel, step_counter) = StepCountingKernel::new();

            let result = export_primitive_null::<u32, _>(mask_iter, &mut kernel);
            assert!(result.is_ok(), "export_primitive_null should not fail");

            let actual_steps = *step_counter.borrow();
            assert_eq!(
                actual_steps, expected_steps,
                "For {} bits, expected {} steps but got {}",
                total_bits, expected_steps, actual_steps
            );
        }
    }

    #[test]
    fn test_export_primitive_nonnull_masked_step_calls() {
        use arrow_buffer::BooleanBuffer;

        // Test various mask patterns to verify step call counts
        let test_cases = [
            (512, 1),  // Less than N (1024), should call step once
            (1024, 1), // Exactly N, should call step once
            (1536, 2), // More than N but less than 2*N, should call step twice
            (2048, 2), // Exactly 2*N, should call step twice
        ];

        for (total_bits, expected_steps) in test_cases {
            // Create a mask with all bits set to true
            let buffer = BooleanBuffer::new_set(total_bits);
            let mask = Mask::from_buffer(buffer);

            let (mut kernel, step_counter) = StepCountingKernel::new();

            let result = export_primitive_nonnull_masked::<u32>(&mask, &mut kernel);
            assert!(
                result.is_ok(),
                "export_primitive_nonnull_masked should not fail"
            );

            let actual_steps = *step_counter.borrow();
            assert_eq!(
                actual_steps, expected_steps,
                "For {} bits (masked), expected {} steps but got {}",
                total_bits, expected_steps, actual_steps
            );
        }
    }

    #[test]
    fn test_export_primitive_nonnull_masked_partial_mask_step_calls() {
        use arrow_buffer::BooleanBuffer;

        // Test with a mask that has some false values
        let total_bits = 2048;
        let expected_steps = 2; // Should still call step twice since we process N elements at a time

        // Create a mask with alternating true/false pattern
        let mut mask_data = vec![];
        for i in 0..total_bits {
            mask_data.push(i % 2 == 0);
        }

        let buffer = BooleanBuffer::from(mask_data);
        let mask = Mask::from_buffer(buffer);

        let (mut kernel, step_counter) = StepCountingKernel::new();

        let result = export_primitive_nonnull_masked::<u32>(&mask, &mut kernel);
        assert!(
            result.is_ok(),
            "export_primitive_nonnull_masked should not fail with partial mask"
        );

        let actual_steps = *step_counter.borrow();
        assert_eq!(
            actual_steps, expected_steps,
            "For {} bits with partial mask, expected {} steps but got {}",
            total_bits, expected_steps, actual_steps
        );
    }

    #[test]
    fn test_export_primitive_nonnull_true_count_tracking() {
        // Test that the kernel receives correct true counts from BitView
        let test_cases = [
            (1024, 1024), // All bits true
            (2048, 2048), // All bits true, multiple steps
        ];

        for (total_bits, expected_true_count) in test_cases {
            let mask_iter = MockMaskIterator::new(total_bits);
            let (mut kernel, _step_counter, true_counter) = TrueCountTrackingKernel::new();

            let result = export_primitive_nonnull::<u32, _>(mask_iter, &mut kernel);
            assert!(result.is_ok(), "export_primitive_nonnull should not fail");

            let actual_true_count = *true_counter.borrow();
            assert_eq!(
                actual_true_count, expected_true_count,
                "For {} bits, expected true count {} but got {}",
                total_bits, expected_true_count, actual_true_count
            );
        }
    }

    #[test]
    fn test_export_primitive_null_true_count_tracking() {
        // Test that export_primitive_null correctly tracks true counts
        let test_cases = [
            (1024, 1024), // All bits true
            (2048, 2048), // All bits true, multiple steps
        ];

        for (total_bits, expected_true_count) in test_cases {
            let mask_iter = MockMaskIterator::new(total_bits);
            let (mut kernel, _step_counter, true_counter) = TrueCountTrackingKernel::new();

            let result = export_primitive_null::<u32, _>(mask_iter, &mut kernel);
            assert!(result.is_ok(), "export_primitive_null should not fail");

            let actual_true_count = *true_counter.borrow();
            assert_eq!(
                actual_true_count, expected_true_count,
                "For {} bits, expected true count {} but got {}",
                total_bits, expected_true_count, actual_true_count
            );
        }
    }

    #[test]
    fn test_export_primitive_nonnull_masked_true_count_tracking() {
        use arrow_buffer::BooleanBuffer;

        // Test with different mask patterns
        let test_cases = [
            (1024, 1024), // All bits true
            (2048, 1024), // Half bits true (alternating pattern)
        ];

        for (total_bits, expected_true_count) in test_cases {
            let buffer = if expected_true_count == total_bits {
                // All bits true
                BooleanBuffer::new_set(total_bits)
            } else {
                // Alternating true/false pattern
                let mut mask_data = vec![];
                for i in 0..total_bits {
                    mask_data.push(i % 2 == 0);
                }
                BooleanBuffer::from(mask_data)
            };

            let mask = Mask::from_buffer(buffer);
            let (mut kernel, _step_counter, true_counter) = TrueCountTrackingKernel::new();

            let result = export_primitive_nonnull_masked::<u32>(&mask, &mut kernel);
            assert!(
                result.is_ok(),
                "export_primitive_nonnull_masked should not fail"
            );

            let actual_true_count = *true_counter.borrow();
            assert_eq!(
                actual_true_count, expected_true_count,
                "For {} bits (masked), expected true count {} but got {}",
                total_bits, expected_true_count, actual_true_count
            );
        }
    }

    #[test]
    fn test_export_primitive_functions_size_tracking() {
        // Test that the functions correctly accumulate size from true_count()
        // This verifies the `size += view.true_count()` lines in the code

        let total_bits = 1536; // Will result in 2 steps
        // MockMaskIterator returns all bits set, so we expect:
        // Step 1: 1024 true bits (full chunk)
        // Step 2: 1024 true bits (full chunk, even though only 512 bits are meaningful)
        let expected_true_count = 2048; // 2 * N (1024)

        let mask_iter = MockMaskIterator::new(total_bits);
        let (mut kernel, _step_counter, true_counter) = TrueCountTrackingKernel::new();

        // Test export_primitive_nonnull
        let result = export_primitive_nonnull::<u32, _>(mask_iter, &mut kernel);
        assert!(result.is_ok());

        let true_count_nonnull = *true_counter.borrow();
        assert_eq!(true_count_nonnull, expected_true_count);

        // Test export_primitive_null
        let mask_iter2 = MockMaskIterator::new(total_bits);
        let (mut kernel2, _step_counter2, true_counter2) = TrueCountTrackingKernel::new();

        let result2 = export_primitive_null::<u32, _>(mask_iter2, &mut kernel2);
        assert!(result2.is_ok());

        let true_count_null = *true_counter2.borrow();
        assert_eq!(true_count_null, expected_true_count);

        // Both should have same true count since MockMaskIterator returns all true bits
        assert_eq!(true_count_nonnull, true_count_null);
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
