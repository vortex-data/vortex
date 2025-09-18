// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use fastlanes::BitPacking;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Element, Kernel, KernelContext, N};
use vortex_buffer::Buffer;
use vortex_dtype::PhysicalPType;
use vortex_error::VortexResult;

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
#[derive(Clone)]
pub struct BitPackedUnalignedKernel<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,

    buffer: Buffer<<T as PhysicalPType>::Physical>,
    packed_offset: usize,
    value_offset: u16,
    temp_buffer: [MaybeUninit<<T as PhysicalPType>::Physical>; 1024],
}

impl<T> BitPackedUnalignedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    pub fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
        packed_offset: usize,
        value_offset: u16,
    ) -> Self {
        assert!(value_offset < 1024);
        BitPackedUnalignedKernel::<T> {
            width,
            packed_stride,
            buffer,
            packed_offset,
            value_offset,
            temp_buffer: [const { MaybeUninit::uninit() }; 1024],
        }
    }

    fn unpack_sliced_chunk(
        width: usize,
        packed_chunk: &[<T as PhysicalPType>::Physical],
        temp_buffer: &mut [MaybeUninit<<T as PhysicalPType>::Physical>; 1024],
        output: &mut [<T as PhysicalPType>::Physical],
        source_offset: usize,
    ) {
        unsafe {
            let temp_slice = std::slice::from_raw_parts_mut(
                temp_buffer.as_mut_ptr() as *mut <T as PhysicalPType>::Physical,
                1024,
            );
            BitPacking::unchecked_unpack(width, packed_chunk, temp_slice);

            let copy_count = output.len();
            output.copy_from_slice(&temp_slice[source_offset..source_offset + copy_count]);
        }
    }
}

impl<T> Kernel for BitPackedUnalignedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    #[allow(clippy::unwrap_in_result, clippy::expect_used)]
    fn step(
        &mut self,
        _ctx: &KernelContext,
        selected: BitView,
        physical_out: &mut ViewMut,
    ) -> VortexResult<()> {
        // We re-interpret the output view as the unsigned bitpacked type.
        physical_out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = physical_out.as_slice_mut::<<T as PhysicalPType>::Physical>();
        let packed = &self.buffer.as_slice()[self.packed_offset..];

        let chunk_value_offset = self.value_offset as usize;

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if selected.true_count() > 8 {
            let mut output_idx = 0;

            // Pre-calculate what we need to do
            let first_chunk_needs_slicing = chunk_value_offset > 0;
            let elements_from_first_chunk = (1024 - chunk_value_offset).min(elements.len());

            let elements_after_first = elements.len() - elements_from_first_chunk;
            let full_chunks_count = elements_after_first / 1024;
            let final_chunk_size = elements_after_first % 1024;
            let final_chunk_needs_slicing = final_chunk_size > 0;

            let total_chunks_needed = 1 + full_chunks_count + (final_chunk_needs_slicing as usize);
            let available_chunks = packed.len() / self.packed_stride;
            let actual_chunks_to_process = total_chunks_needed.min(available_chunks);

            // Part 1: Handle first sliced chunk (if there's a value_offset)
            if actual_chunks_to_process > 0 {
                Self::unpack_sliced_chunk(
                    self.width,
                    &packed[0..self.packed_stride],
                    &mut self.temp_buffer,
                    &mut elements[output_idx..output_idx + elements_from_first_chunk],
                    chunk_value_offset,
                );
                output_idx += elements_from_first_chunk;
            }

            // Part 2: Handle all non-sliced full chunks (for loop)
            let last_full_chunk_idx = full_chunks_count + 1;

            for packed_idx in 1..last_full_chunk_idx.min(actual_chunks_to_process) {
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &packed[(packed_idx * self.packed_stride)..][..self.packed_stride],
                        &mut elements[output_idx..output_idx + 1024],
                    );
                }
                output_idx += 1024;
            }

            // Part 3: Handle final sliced chunk (if needed)
            if last_full_chunk_idx < actual_chunks_to_process {
                Self::unpack_sliced_chunk(
                    self.width,
                    &packed[(last_full_chunk_idx * self.packed_stride)..][..self.packed_stride],
                    &mut self.temp_buffer,
                    &mut elements[output_idx..output_idx + final_chunk_size],
                    0,
                );
            }

            let nvecs = (first_chunk_needs_slicing as usize) + full_chunks_count;

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.flatten::<<T as PhysicalPType>::Physical>(&selected);
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let adjusted_idx = idx + chunk_value_offset;
                let chunk_idx = adjusted_idx / 1024;
                let bit_idx = adjusted_idx % 1024;

                let start_idx = chunk_idx * self.packed_stride;
                if start_idx + self.packed_stride <= packed.len() {
                    unsafe {
                        *elements.get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
                            self.width,
                            &packed[start_idx..start_idx + self.packed_stride],
                            bit_idx,
                        );
                    }
                } else {
                    // Not enough packed data - set to default value
                    elements[offset] = Default::default();
                }
                offset += 1;
            });

            let elements_needed = elements.len() + chunk_value_offset;
            let chunks_needed = elements_needed.div_ceil(1024);
            let nvecs = chunks_needed
                .min(packed.len() / self.packed_stride)
                .min(N / 1024);
            self.packed_offset += nvecs * self.packed_stride;
        }

        physical_out.reinterpret_as::<T>();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::pipeline::bits::BitView;
    use vortex_array::pipeline::view::ViewMut;
    use vortex_array::pipeline::{KernelContext, N, N_WORDS};
    use vortex_fastlanes::bitpack_to_best_bit_width;

    use super::*;

    #[test]
    fn test_unaligned_kernel_step_dense() {
        let len = 2048 + 100; // More than one chunk plus extra
        let offset = 7u16;
        let values: Vec<i32> = (0..len).collect();

        let primitive_array: PrimitiveArray = values.clone().into_iter().collect();
        let array = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Create the unaligned kernel
        let packed_stride = array.bit_width() as usize * 32; // i32 FastLanes lanes
        let buffer = Buffer::<u32>::from_byte_buffer(array.packed().clone().into_byte_buffer());
        let mut kernel = BitPackedUnalignedKernel::<i32>::new(
            array.bit_width() as usize,
            packed_stride,
            buffer,
            0,
            offset,
        );

        // Test dense selection (all true)
        let bit_view = BitView::all_true();
        let ctx = KernelContext::default();
        let mut output_data = vec![0i32; N];
        let mut output = ViewMut::new(&mut output_data, None);

        // Call step function
        kernel.step(&ctx, bit_view, &mut output).unwrap();

        // Verify results - should match original values starting from offset
        let expected = &values[offset as usize..][..N];
        assert_eq!(
            output.as_slice::<i32>(),
            expected,
            "Dense unaligned step failed"
        );
    }

    #[test]
    fn test_unaligned_kernel_step_sparse() {
        let len = 1024 + 512; // One full chunk plus partial
        let offset = 15u16;
        let values: Vec<i16> = (0..len).map(|i| (i * 3 + 7) as i16).collect();

        let primitive_array: PrimitiveArray = values.clone().into_iter().collect();
        let array = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Create the unaligned kernel
        let packed_stride = array.bit_width() as usize * 64; // i16 FastLanes lanes
        let buffer = Buffer::<u16>::from_byte_buffer(array.packed().clone().into_byte_buffer());
        let mut kernel = BitPackedUnalignedKernel::<i16>::new(
            array.bit_width() as usize,
            packed_stride,
            buffer,
            0,
            offset,
        );

        // Create sparse selection (every 64th element)
        let selected_indices: Vec<usize> = (0..N).step_by(64).take(8).collect();
        let mut mask_data = [0usize; N_WORDS];
        for &idx in &selected_indices {
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            if word_idx < N_WORDS {
                mask_data[word_idx] |= 1usize << bit_idx;
            }
        }
        let bit_view = BitView::new(&mask_data);

        let ctx = KernelContext::default();
        // ViewMut requires exactly N elements
        let mut output_data = vec![0i16; N];
        let mut output = ViewMut::new(&mut output_data, None);

        // Call step function
        kernel.step(&ctx, bit_view, &mut output).unwrap();

        // Verify results - check only the first few selected values (step function compacts them)
        let output_slice = output.as_slice::<i16>();
        for (i, &idx) in selected_indices.iter().enumerate() {
            let expected_value = values[offset as usize + idx];
            assert_eq!(
                output_slice[i], expected_value,
                "Sparse unaligned step failed at index {}",
                i
            );
        }
    }

    #[rstest]
    #[case(1u16, "small offset")]
    #[case(8u16, "byte-aligned offset")]
    #[case(63u16, "near chunk boundary")]
    #[case(100u16, "mid-chunk offset")]
    fn test_unaligned_kernel_step_different_offsets(
        #[case] offset: u16,
        #[case] description: &str,
    ) {
        let len = N + offset as usize + 100; // Ensure we have enough data
        let values: Vec<i8> = (0..len)
            .map(|i| ((i + offset as usize) % 127) as i8)
            .collect();

        let primitive_array: PrimitiveArray = values.clone().into_iter().collect();
        let array = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Create the unaligned kernel - use proper FastLanes lanes count
        let packed_stride = array.bit_width() as usize * 128; // i8 has 128 lanes in FastLanes
        let buffer = Buffer::<u8>::from_byte_buffer(array.packed().clone().into_byte_buffer());
        let mut kernel = BitPackedUnalignedKernel::<i8>::new(
            array.bit_width() as usize,
            packed_stride,
            buffer,
            0,
            offset,
        );

        // Test with all true mask
        let bit_view = BitView::all_true();
        let ctx = KernelContext::default();
        let mut output_data = vec![0i8; N];
        let mut output = ViewMut::new(&mut output_data, None);

        // Call step function
        kernel.step(&ctx, bit_view, &mut output).unwrap();

        // Verify results - ensure we don't go out of bounds
        let expected = &values[offset as usize..offset as usize + N];
        assert_eq!(
            output.as_slice::<i8>(),
            expected,
            "Unaligned step failed for {}: offset={}",
            description,
            offset
        );
    }
}
