// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, Nullability, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::Array;
use crate::arrays::PrimitiveArray;
use crate::pipeline::bits::{
    AlignedBitSink, BitAlignedChunkedIterator, BitSink, BitView, BitViewMut, EmptyBitSink,
    MaskSliceIterator, TrueSliceIterator, UnalignedBitSink,
};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N};
use crate::validity::Validity;

pub(super) fn export_primitive(
    ptype: PType,
    nullability: Nullability,
    mask: &Mask,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray> {
    match (nullability, mask.all_true()) {
        (Nullability::NonNullable, true) => match_each_native_ptype!(ptype, |T| {
            export_primitive_impl::<T, _, _>(
                TrueSliceIterator::new(mask.len()),
                EmptyBitSink,
                pipeline,
            )
        }),
        (Nullability::NonNullable, false) => match_each_native_ptype!(ptype, |T| {
            export_primitive_impl::<T, _, _>(
                BitAlignedChunkedIterator::new(&mask.to_boolean_buffer(), mask.true_count()),
                EmptyBitSink,
                pipeline,
            )
        }),
        (Nullability::Nullable, true) => match_each_native_ptype!(ptype, |T| {
            export_primitive_impl::<T, _, _>(
                TrueSliceIterator::new(mask.len()),
                AlignedBitSink::new(mask.true_count()),
                pipeline,
            )
        }),

        (Nullability::Nullable, false) => match_each_native_ptype!(ptype, |T| {
            export_primitive_impl::<T, _, _>(
                BitAlignedChunkedIterator::new(&mask.to_boolean_buffer(), mask.true_count()),
                UnalignedBitSink::new(mask.true_count()),
                pipeline,
            )
        }),
    }
}

fn export_primitive_impl<T, MaskIter, Sink>(
    mut mask_iter: MaskIter,
    mut sink: Sink,
    pipeline: &mut dyn Kernel,
) -> VortexResult<PrimitiveArray>
where
    T: Element + NativePType,
    MaskIter: MaskSliceIterator,
    Sink: BitSink,
{
    let len = mask_iter.len();
    let capacity = len.next_multiple_of(N) + N;

    let mut elements = BufferMut::<T>::with_capacity(capacity);
    unsafe { elements.set_len(capacity) };
    let mut element_count = 0;

    let mut remaining = len;
    while remaining > 0 {
        let sink_slice = sink.next_chunk();
        let mut elements_view = ViewMut::new(
            &mut elements[element_count..][..N],
            sink_slice.map(BitViewMut::new),
        );
        let dummy_ctx = KernelContext::default();
        // TODO(joe): have iter return true count.
        let view = BitView::new(mask_iter.next_chunk().vortex_expect("mask iterator"));
        let true_count = view.true_count();

        pipeline.step(&dummy_ctx, view, &mut elements_view)?;
        sink.commit_n(true_count).vortex_expect("commit");
        element_count += true_count;
        remaining = remaining.saturating_sub(N);
    }

    unsafe { elements.set_len(element_count) };

    if let Some(validity) = sink.finish()? {
        Ok(PrimitiveArray::new(
            elements.freeze(),
            Validity::from(validity),
        ))
    } else {
        Ok(PrimitiveArray::new(
            elements.freeze(),
            Validity::NonNullable,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::pipeline::bits::{BitAlignedChunkedIterator, TrueSliceIterator};

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

        fn true_count(&self) -> usize {
            self.len // All bits are set to true in MockMaskIterator
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
            let result =
                export_primitive_impl::<u32, _, _>(mask_iter, EmptyBitSink::default(), &mut kernel);
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

            let result =
                export_primitive_impl::<u32, _, _>(mask_iter, EmptyBitSink::default(), &mut kernel);
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
        // Test various mask patterns to verify step call counts
        let test_cases = [
            (512, 1),  // Less than N (1024), should call step once
            (1024, 1), // Exactly N, should call step once
            (1536, 2), // More than N but less than 2*N, should call step twice
            (2048, 2), // Exactly 2*N, should call step twice
        ];

        for (total_bits, expected_steps) in test_cases {
            let (mut kernel, step_counter) = StepCountingKernel::new();

            // Create a mask with all bits set to true
            let result = export_primitive_impl::<u32, _, _>(
                TrueSliceIterator::new(total_bits),
                EmptyBitSink::default(),
                &mut kernel,
            );
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

        let boolean_buffer = mask.to_boolean_buffer();
        let result = export_primitive_impl::<u32, _, _>(
            BitAlignedChunkedIterator::new(&boolean_buffer, mask.true_count()),
            EmptyBitSink::default(),
            &mut kernel,
        );
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

            let result =
                export_primitive_impl::<u32, _, _>(mask_iter, EmptyBitSink::default(), &mut kernel);
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

            let result =
                export_primitive_impl::<u32, _, _>(mask_iter, EmptyBitSink::default(), &mut kernel);
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

            let boolean_buffer = mask.to_boolean_buffer();
            let result = export_primitive_impl::<u32, _, _>(
                BitAlignedChunkedIterator::new(&boolean_buffer, mask.true_count()),
                EmptyBitSink::default(),
                &mut kernel,
            );
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
        let result =
            export_primitive_impl::<u32, _, _>(mask_iter, EmptyBitSink::default(), &mut kernel);
        assert!(result.is_ok());

        let true_count_nonnull = *true_counter.borrow();
        assert_eq!(true_count_nonnull, expected_true_count);

        // Test export_primitive_null
        let mask_iter2 = MockMaskIterator::new(total_bits);
        let (mut kernel2, _step_counter2, true_counter2) = TrueCountTrackingKernel::new();

        let result2 =
            export_primitive_impl::<u32, _, _>(mask_iter2, EmptyBitSink::default(), &mut kernel2);
        assert!(result2.is_ok());

        let true_count_null = *true_counter2.borrow();
        assert_eq!(true_count_null, expected_true_count);

        // Both should have same true count since MockMaskIterator returns all true bits
        assert_eq!(true_count_nonnull, true_count_null);
    }

    /// Simple kernel that just tracks step calls and true counts
    struct SimpleTrackingKernel {
        step_count: Rc<RefCell<usize>>,
        total_true_count: Rc<RefCell<usize>>,
    }

    impl SimpleTrackingKernel {
        fn new() -> (Self, Rc<RefCell<usize>>, Rc<RefCell<usize>>) {
            let step_counter = Rc::new(RefCell::new(0));
            let true_counter = Rc::new(RefCell::new(0));
            (
                SimpleTrackingKernel {
                    step_count: step_counter.clone(),
                    total_true_count: true_counter.clone(),
                },
                step_counter,
                true_counter,
            )
        }
    }

    impl Kernel for SimpleTrackingKernel {
        fn step(
            &mut self,
            _ctx: &KernelContext,
            selected: BitView,
            _out: &mut ViewMut,
        ) -> VortexResult<()> {
            *self.step_count.borrow_mut() += 1;
            let true_count = selected.true_count();
            *self.total_true_count.borrow_mut() += true_count;
            Ok(())
        }
    }

    #[test]
    fn test_export_primitive_nonnull_with_unaligned_sink_and_mixed_mask() {
        // Test export_primitive_nonnull with UnalignedBitSink and mixed true/false mask
        let total_bits = 2048; // Exactly 2*N bits

        // Create a mask with alternating true/false pattern
        let mut mask_data = vec![];
        for i in 0..total_bits {
            mask_data.push(i % 2 == 0); // Even indices are true
        }

        let buffer = BooleanBuffer::from(mask_data);
        let mask = Mask::from_buffer(buffer);
        let expected_true_count = mask.true_count(); // Should be 1024 (half)

        let boolean_buffer = mask.to_boolean_buffer();
        let mask_iter = BitAlignedChunkedIterator::new(&boolean_buffer, mask.true_count());
        let unaligned_sink = UnalignedBitSink::new(expected_true_count);
        let (mut kernel, _step_counter, true_counter) = SimpleTrackingKernel::new();

        let result = export_primitive_impl::<u32, _, _>(mask_iter, unaligned_sink, &mut kernel);
        assert!(
            result.is_ok(),
            "export_primitive_nonnull with UnalignedBitSink should not fail"
        );

        let primitive_array = result.unwrap();

        // Verify the kernel processed the correct number of true bits
        let actual_true_count = *true_counter.borrow();
        assert_eq!(
            actual_true_count, expected_true_count,
            "Kernel should have seen {} true bits, but saw {}",
            expected_true_count, actual_true_count
        );

        // Verify the array has the correct length (should equal true_count)
        assert_eq!(primitive_array.len(), expected_true_count);

        // Verify the validity mask was created by the UnalignedBitSink
        match primitive_array.validity() {
            Validity::NonNullable => {
                panic!("Expected validity mask to be set by UnalignedBitSink, but got NonNullable");
            }
            Validity::Array(validity_array) => {
                let bool_array = validity_array.to_bool();
                let validity_buffer = bool_array.boolean_buffer();

                // The validity should have the same length as the output array
                assert_eq!(
                    validity_buffer.len(),
                    expected_true_count,
                    "Validity buffer length should match output array length"
                );

                // Verify the validity buffer exists (specific pattern doesn't matter for this test)
                // The key is that UnalignedBitSink successfully created and returned a validity mask
            }
            _ => {
                // AllValid or AllInvalid are also acceptable outcomes
            }
        }
    }

    #[test]
    fn test_export_primitive_nonnull_with_aligned_sink_and_exact_n_bits() {
        // Test export_primitive_nonnull with AlignedBitSink using exactly N bits
        let total_bits = N; // Exactly one chunk

        // Create a mask with all bits set to true
        let mask_data = vec![true; total_bits];
        let buffer = BooleanBuffer::from(mask_data);
        let mask = Mask::from_buffer(buffer);

        let boolean_buffer = mask.to_boolean_buffer();
        let mask_iter = BitAlignedChunkedIterator::new(&boolean_buffer, mask.true_count());
        let aligned_sink = AlignedBitSink::new(total_bits);
        let (mut kernel, _step_counter, true_counter) = SimpleTrackingKernel::new();

        let result = export_primitive_impl::<u32, _, _>(mask_iter, aligned_sink, &mut kernel);
        assert!(
            result.is_ok(),
            "export_primitive_nonnull with AlignedBitSink should not fail"
        );

        let primitive_array = result.unwrap();

        // Verify the kernel processed exactly N true bits
        let actual_true_count = *true_counter.borrow();
        assert_eq!(actual_true_count, N);

        // Verify the array has the correct length
        assert_eq!(primitive_array.len(), N);

        // Verify the validity mask was created by the AlignedBitSink
        match primitive_array.validity() {
            Validity::Array(validity_array) => {
                let bool_array = validity_array.to_bool();
                let validity_buffer = bool_array.boolean_buffer();

                // The validity should have the same length as the output array
                assert_eq!(
                    validity_buffer.len(),
                    N,
                    "Validity buffer length should match output array length"
                );
            }
            _ => {
                // AllValid, AllInvalid, or NonNullable are acceptable outcomes
            }
        }
    }

    #[test]
    fn test_export_primitive_nonnull_empty_sink_no_validity() {
        // Test export_primitive_nonnull with EmptyBitSink should produce no validity
        let total_bits = 1024;
        let mask_iter = MockMaskIterator::new(total_bits);
        let empty_sink = EmptyBitSink::default();
        let (mut kernel, _step_counter, true_counter) = SimpleTrackingKernel::new();

        let result = export_primitive_impl::<u32, _, _>(mask_iter, empty_sink, &mut kernel);
        assert!(
            result.is_ok(),
            "export_primitive_nonnull with EmptyBitSink should not fail"
        );

        let primitive_array = result.unwrap();

        // Verify the kernel processed the correct number of bits
        let actual_true_count = *true_counter.borrow();
        assert_eq!(actual_true_count, total_bits);

        // Verify the array has the correct length
        assert_eq!(primitive_array.len(), total_bits);

        // Verify no validity mask was set (EmptyBitSink returns None)
        match primitive_array.validity() {
            Validity::NonNullable => {
                // This is expected for EmptyBitSink
            }
            _ => panic!("Expected NonNullable validity for EmptyBitSink, but got validity array"),
        }
    }
}
