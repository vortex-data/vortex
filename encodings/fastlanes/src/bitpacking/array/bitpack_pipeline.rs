// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::{transmute, transmute_copy};

use fastlanes::{BitPacking, FastLanes};
use static_assertions::const_assert_eq;
use vortex_array::pipeline::{
    BindContext, BitView, Kernel, KernelCtx, N, PipelineInputs, PipelinedNode,
};
use vortex_buffer::Buffer;
use vortex_dtype::{PTypeDowncastExt, PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_mask::MaskMut;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::BitPackedArray;

/// The size of a FastLanes vector of elements.
const FL_VECTOR_SIZE: usize = 1024;

// Bitpacking uses FastLanes decompression, which expects a multiple of 1024 elements.
const_assert_eq!(N, FL_VECTOR_SIZE);

// TODO(connor): Run some benchmarks to actually get a good value.
/// The true count threshold at which it is faster to unpack individual bitpacked values one at a
/// time instead of unpack entire vectors and then filter later.
const SCALAR_UNPACK_THRESHOLD: usize = 7;

impl PipelinedNode for BitPackedArray {
    fn inputs(&self) -> PipelineInputs {
        PipelineInputs::Source
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert!(self.bit_width > 0);

        if self.patches.is_some() {
            unimplemented!(
                "We do not handle patches for bitpacked right now, as this will become a parent patch array"
            );
        }

        match_each_integer_ptype!(self.ptype(), |T| {
            let packed_bit_width = self.bit_width as usize;
            let packed_buffer = Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
                self.packed.clone().into_byte_buffer(),
            );

            if self.offset != 0 {
                // TODO(ngates): the unaligned kernel needs fixing for the non-masked API
                unimplemented!(
                    "Unaligned `BitPackedArray` as a `PipelineSource` is not yet implemented"
                )
            }

            Ok(Box::new(AlignedBitPackedKernel::<T>::new(
                packed_bit_width,
                packed_buffer,
                // FIXME(ngates): if we make sure the mask has offset zero, we know that split_off
                //  inside the kernel is free.
                self.validity.to_mask(self.len()).into_mut(),
            )) as Box<dyn Kernel>)
        })
    }
}

pub struct AlignedBitPackedKernel<BP: PhysicalPType<Physical: BitPacking>> {
    /// The bit width of each bitpacked value.
    ///
    /// This is guaranteed to be less than or equal to the (unpacked) bit-width of `BP`.
    packed_bit_width: usize,

    /// The stride of the bitpacked values, which when fully unpacked will occupy exactly 1024 bits.
    /// This is equal to `1024 * bit_width / BP::Physical::T`
    ///
    /// We store this here so that we do not have to keep calculating this in [`step()`].
    ///
    /// For example, if the `bit_width` is 10 and the physical type is `u16` (which will fill up
    /// `1024 / 16 = 64` lanes), the `packed_stride` will be `10 * 64 = 640`. This ensures we pass
    /// a slice with the correct length to [`BitPacking::unchecked_unpack`].
    ///
    /// [`step()`]: SourceKernel::step
    /// [`BitPacking::unchecked_unpack()`]: BitPacking::unchecked_unpack
    packed_stride: usize,

    /// The buffer containing the bitpacked values.
    packed_buffer: Buffer<BP::Physical>,

    /// The validity mask for the bitpacked array.
    validity: MaskMut,

    /// The total number of bitpacked chunks we have unpacked.
    num_chunks_unpacked: usize,
}

impl<BP: PhysicalPType<Physical: BitPacking>> AlignedBitPackedKernel<BP> {
    pub fn new(
        packed_bit_width: usize,
        // TODO(ngates): hold an iterator over chunks instead of the full buffer?
        packed_buffer: Buffer<BP::Physical>,
        validity: MaskMut,
    ) -> Self {
        let packed_stride =
            packed_bit_width * <<BP as PhysicalPType>::Physical as FastLanes>::LANES;

        assert_eq!(
            packed_stride,
            FL_VECTOR_SIZE * packed_bit_width / BP::Physical::T
        );
        assert!(packed_bit_width <= BP::Physical::T);

        Self {
            packed_bit_width,
            packed_stride,
            packed_buffer,
            validity,
            num_chunks_unpacked: 0,
        }
    }
}

impl<BP: PhysicalPType<Physical: BitPacking>> Kernel for AlignedBitPackedKernel<BP> {
    fn step(
        &mut self,
        _ctx: &mut KernelCtx,
        selection: &BitView,
        out: VectorMut,
    ) -> VortexResult<VectorMut> {
        if selection.true_count() == 0 {
            debug_assert!(out.is_empty());
            return Ok(out);
        }

        let mut output: PVectorMut<BP> = out.into_primitive().downcast();
        debug_assert!(output.is_empty());

        let packed_offset = self.num_chunks_unpacked * self.packed_stride;
        let packed_bytes = &self.packed_buffer[packed_offset..][..self.packed_stride];

        // If the true count is very small (the selection is sparse), we can unpack individual
        // elements directly into the output vector.
        if selection.true_count() < SCALAR_UNPACK_THRESHOLD {
            output.reserve(selection.true_count());

            selection.iter_ones(|idx| {
                if self.validity.value(idx) {
                    // SAFETY:
                    // - The documentation for `packed_bit_width` explains that the size is valid.
                    // - We know that the size of the `next_packed_chunk` we provide is equal to
                    //   `self.packed_stride`, and we explain why this is correct in its
                    //   documentation.
                    let unpacked_value = unsafe {
                        BitPacking::unchecked_unpack_single(
                            self.packed_bit_width,
                            packed_bytes,
                            idx,
                        )
                    };

                    // SAFETY: We just reserved enough capacity to push these values.
                    unsafe { output.push_unchecked(transmute_copy(&unpacked_value)) };
                } else {
                    output.append_nulls(1);
                }
            });

            self.num_chunks_unpacked += 1;
            return Ok(output.into());
        }

        // Otherwise if the mask is dense, it is faster to fully unpack the entire 1024
        // element lane with SIMD / FastLanes and let other nodes in the pipeline decide if they
        // want to perform the selection filter themselves.
        let (mut elements, _validity) = output.into_parts();

        elements.reserve(N);
        // SAFETY: we just reserved enough capacity.
        unsafe { elements.set_len(N) };

        unsafe {
            BitPacking::unchecked_unpack(
                self.packed_bit_width,
                packed_bytes,
                transmute::<&mut [BP], &mut [BP::Physical]>(elements.as_mut()),
            );
        }

        // Prepare the output validity mask for this chunk.
        let mut chunk_validity = self.validity.split_off(N.min(self.validity.capacity()));
        std::mem::swap(&mut self.validity, &mut chunk_validity);

        // For the final chunk, we may have fewer than N elements to unpack.
        // So we just set the length of the output to the correct value.
        unsafe { elements.set_len(chunk_validity.len()) };

        self.num_chunks_unpacked += 1;

        Ok(PVectorMut::new(elements, chunk_validity).into())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;

    use crate::BitPackedArray;

    #[test]
    fn test_bitpack_pipeline_basic() {
        // Create exactly 1024 elements (0 to 1023).
        let values = (0..1024).map(|i| i as u32);
        let primitive = PrimitiveArray::from_iter(values).to_array();

        // Encode with 10-bit width (max value 1023 fits in 10 bits).
        let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
        assert_eq!(bitpacked.bit_width(), 10, "Bit width should be 10");

        // Select all elements.
        let mask = Mask::new_true(1024);

        // This should trigger the pipeline since `BitPackedArray` implements `PipelinedNode`.
        let result = bitpacked.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), 1024, "Result should have 1024 elements");

        let pvector_u32 = result.as_primitive().into_u32();
        let elements = pvector_u32.elements().as_slice();

        for i in 0..1024 {
            assert_eq!(
                elements[i], i as u32,
                "Value at index {} should be {}",
                i, i
            );
        }
    }

    #[ignore = "TODO(connor): need to filter in pipeline driver step"]
    #[test]
    fn test_bitpack_pipeline_dense_75_percent() {
        // Create exactly 1024 elements (0 to 1023).
        let values = (0..1024).map(|i| i as u32);
        let primitive = PrimitiveArray::from_iter(values).to_array();

        // Encode with 10-bit width.
        let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
        assert_eq!(bitpacked.bit_width(), 10, "Bit width should be 10");

        // Select 75% of elements (768 out of 1024) - every element where index % 4 != 0.
        let indices: Vec<usize> = (0..1024).filter(|i| i % 4 != 0).collect();
        assert_eq!(indices.len(), 768, "Should select exactly 768 elements");
        let mask = Mask::from_indices(1024, indices);

        // This should still use the dense path since true_count >= 7.
        let result = bitpacked.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(
            result.len(),
            1024,
            "Result should have 1024 elements (dense path outputs all N elements)"
        );

        let pvector_u32 = result.as_primitive().into_u32();
        let elements = pvector_u32.elements().as_slice();

        // Check that selected elements have correct values.
        // Elements where index % 4 != 0 should have their original values.
        for i in 0..1024 {
            if i % 4 != 0 {
                assert_eq!(
                    elements[i], i as u32,
                    "Selected element at {} should be {}",
                    i, i
                );
            }
            // Note: Unselected elements (where i % 4 == 0) may have undefined values.
        }
    }

    #[test]
    fn test_bitpack_pipeline_sparse_5_elements() {
        // Create exactly 1024 elements (0 to 1023).
        let values = (0..1024).map(|i| i as u32);
        let primitive = PrimitiveArray::from_iter(values).to_array();

        // Encode with 10-bit width.
        let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
        assert_eq!(bitpacked.bit_width(), 10, "Bit width should be 10");

        // Select only 5 elements at specific indices.
        let indices = vec![10, 100, 256, 512, 1000];
        let mask = Mask::from_indices(1024, indices);

        // This should use the sparse path since true_count < 7.
        let result = bitpacked.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), 5, "Result should have 5 elements");

        let pvector_u32 = result.as_primitive().into_u32();
        let elements = pvector_u32.elements().as_slice();

        // Verify the values match the selected indices.
        assert_eq!(elements[0], 10);
        assert_eq!(elements[1], 100);
        assert_eq!(elements[2], 256);
        assert_eq!(elements[3], 512);
        assert_eq!(elements[4], 1000);
    }

    #[test]
    fn test_bitpack_pipeline_sparse_with_nulls() {
        // Create 1024 elements with some nulls.
        let values: Vec<Option<u32>> = (0..1024)
            .map(|i| if i % 100 == 0 { None } else { Some(i as u32) })
            .collect();
        let primitive = PrimitiveArray::from_option_iter(values).to_array();

        // Encode with 10-bit width.
        let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
        assert_eq!(bitpacked.bit_width(), 10, "Bit width should be 10");

        // Select only 5 elements at specific indices, including a null value at index 100.
        let indices = vec![10, 100, 256, 512, 1000];
        let mask = Mask::from_indices(1024, indices);

        // This should use the sparse path since true_count < 7.
        let result = bitpacked.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), 5, "Result should have 5 elements");

        let pvector_u32 = result.as_primitive().into_u32();
        let elements = pvector_u32.elements().as_slice();

        // Verify the values and validity.
        assert_eq!(elements[0], 10);
        assert!(
            pvector_u32.validity().value(0),
            "Element at index 0 should be valid"
        );

        // Index 100 should be null.
        assert!(
            !pvector_u32.validity().value(1),
            "Element at index 1 (original index 100) should be null"
        );

        assert_eq!(elements[2], 256);
        assert!(
            pvector_u32.validity().value(2),
            "Element at index 2 should be valid"
        );

        assert_eq!(elements[3], 512);
        assert!(
            pvector_u32.validity().value(3),
            "Element at index 3 should be valid"
        );

        // Index 1000 should be null.
        assert!(
            !pvector_u32.validity().value(4),
            "Element at index 4 (original index 1000) should be null"
        );
    }

    #[test]
    fn test_bitpack_pipeline_dense_with_nulls() {
        // Create 1024 elements with some nulls.
        let values: Vec<Option<u32>> = (0..1024)
            .map(|i| if i % 100 == 0 { None } else { Some(i as u32) })
            .collect();
        let primitive = PrimitiveArray::from_option_iter(values).to_array();

        // Encode with 10-bit width.
        let bitpacked = BitPackedArray::encode(&primitive, 10).unwrap();
        assert_eq!(bitpacked.bit_width(), 10, "Bit width should be 10");

        // Select all elements (dense path).
        let mask = Mask::new_true(1024);

        // This should use the dense path since true_count >= 7.
        let result = bitpacked.to_array().execute_with_selection(&mask).unwrap();
        assert_eq!(result.len(), 1024, "Result should have 1024 elements");

        let pvector_u32 = result.as_primitive().into_u32();
        let elements = pvector_u32.elements().as_slice();

        // Verify the values and validity.
        for i in 0..1024 {
            if i % 100 == 0 {
                assert!(
                    !pvector_u32.validity().value(i),
                    "Element at index {} should be null",
                    i
                );
            } else {
                assert_eq!(
                    elements[i], i as u32,
                    "Element at index {} should be {}",
                    i, i
                );
                assert!(
                    pvector_u32.validity().value(i),
                    "Element at index {} should be valid",
                    i
                );
            }
        }
    }
}
