// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::Arc;

use fastlanes::{BitPacking, FastLanes};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::operators::{BindContext, Operator, OperatorRef};
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Element, Kernel, KernelContext, N, PipelineVTable, VType};
use vortex_buffer::Buffer;
use vortex_dtype::{PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedVTable};

impl PipelineVTable<BitPackedVTable> for BitPackedVTable {
    fn to_operator(array: &BitPackedArray) -> VortexResult<Option<OperatorRef>> {
        if array.dtype.is_nullable() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }
        if array.patches.is_some() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }

        Ok(Some(Arc::new(array.clone())))
    }
}

impl Operator for BitPackedArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype())
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(&self, _children: Vec<OperatorRef>) -> OperatorRef {
        Arc::new(self.clone())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert!(self.bit_width > 0);
        match_each_integer_ptype!(self.ptype(), |T| {
            let packed_stride =
                self.bit_width as usize * <<T as PhysicalPType>::Physical as FastLanes>::LANES;
            Ok(Box::new(BitPackedKernel::<T>::new(
                self.bit_width as usize,
                packed_stride,
                Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
                    self.packed.clone().into_byte_buffer(),
                ),
                0,
                self.offset,
            )) as Box<dyn Kernel>)
        })
    }
}

impl Hash for BitPackedArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.packed.as_ptr().addr().hash(state);
        self.bit_width.hash(state);
        self.dtype.hash(state);
    }
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
pub(crate) struct BitPackedKernel<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,

    buffer: Buffer<<T as PhysicalPType>::Physical>,
    packed_offset: usize,
    value_offset: u16,
}

impl<T> BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn new(
        width: usize,
        packed_stride: usize,
        buffer: Buffer<<T as PhysicalPType>::Physical>,
        packed_offset: usize,
        value_offset: u16,
    ) -> Self {
        assert!(value_offset < 1024);
        BitPackedKernel::<T> {
            width,
            packed_stride,
            buffer,
            packed_offset,
            value_offset,
        }
    }

    fn unpack_sliced_chunk(
        &self,
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
            BitPacking::unchecked_unpack(self.width, packed_chunk, temp_slice);

            let copy_count = output.len();
            output.copy_from_slice(&temp_slice[source_offset..source_offset + copy_count]);
        }
    }
}

impl<T> Kernel for BitPackedKernel<T>
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
        let mut temp_buffer: [MaybeUninit<<T as PhysicalPType>::Physical>; 1024] =
            [const { MaybeUninit::uninit() }; 1024];
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
            let elements_from_first_chunk = if first_chunk_needs_slicing {
                (1024 - chunk_value_offset).min(elements.len())
            } else {
                0
            };

            let elements_after_first = elements.len() - elements_from_first_chunk;
            let full_chunks_count = elements_after_first / 1024;
            let final_chunk_size = elements_after_first % 1024;
            let final_chunk_needs_slicing = final_chunk_size > 0;

            let total_chunks_needed = (first_chunk_needs_slicing as usize)
                + full_chunks_count
                + (final_chunk_needs_slicing as usize);
            let available_chunks = packed.len() / self.packed_stride;
            let actual_chunks_to_process = total_chunks_needed.min(available_chunks);

            // Part 1: Handle first sliced chunk (if there's a value_offset)
            if first_chunk_needs_slicing && actual_chunks_to_process > 0 {
                self.unpack_sliced_chunk(
                    &packed[0..self.packed_stride],
                    &mut temp_buffer,
                    &mut elements[output_idx..output_idx + elements_from_first_chunk],
                    chunk_value_offset,
                );
                output_idx += elements_from_first_chunk;
            }

            // Part 2: Handle all non-sliced full chunks (for loop)
            let first_full_chunk_idx = if first_chunk_needs_slicing { 1 } else { 0 };
            let last_full_chunk_idx = first_full_chunk_idx + full_chunks_count;

            for packed_idx in
                first_full_chunk_idx..last_full_chunk_idx.min(actual_chunks_to_process)
            {
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
            if final_chunk_needs_slicing && last_full_chunk_idx < actual_chunks_to_process {
                self.unpack_sliced_chunk(
                    &packed[(last_full_chunk_idx * self.packed_stride)..][..self.packed_stride],
                    &mut temp_buffer,
                    &mut elements[output_idx..output_idx + final_chunk_size],
                    0,
                );
            }

            let nvecs = (first_chunk_needs_slicing as usize) + full_chunks_count;

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.select_mask::<<T as PhysicalPType>::Physical>(&selected);
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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::filter;
    use vortex_array::pipeline::export_canonical_pipeline_expr;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use crate::{FoRArray, bitpack_encode, bitpack_to_best_bit_width};

    #[test]
    fn test_bitpacking_pipeline() {
        for frac in [0.5] {
            let len = 10;
            let mut rng = StdRng::seed_from_u64(0);
            let values = (0i16..len)
                .map(|_| rng.random_range(0..100))
                .collect::<BufferMut<_>>();

            let primitive_array = values.clone().into_array().to_primitive().unwrap();
            let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();

            let mask = (0..len)
                .map(|_| rng.random_bool(frac))
                .collect::<BooleanBuffer>();
            let mask = Mask::from_buffer(mask);

            let result = export_canonical_pipeline_expr(
                bitpacked.dtype(),
                bitpacked.len(),
                bitpacked.to_operator().unwrap().unwrap().as_ref(),
                &mask,
            )
            .unwrap()
            .into_array();

            let expect = filter(bitpacked.to_canonical().unwrap().as_ref(), &mask).unwrap();

            assert_eq!(result.len(), expect.len());

            for i in 0..mask.true_count() {
                assert_eq!(
                    result.scalar_at(i),
                    expect.scalar_at(i),
                    "mismatch at index {}",
                    i,
                );
            }
        }
    }

    #[test]
    fn test_bitpacking_offset_simple() {
        // Test a simple case: 1024 + 10 elements, offset by 5
        let len = 1034usize;
        let offset = 5usize;

        let values = (0..len).map(|i| i as i32).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();
        let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Slice to get elements [5, 6, 7, ..., 1033] (1029 elements)
        let sliced = bitpacked.slice(offset, len);

        // Just test first few elements manually
        let val0: i32 = sliced.scalar_at(0).try_into().unwrap();
        let val1: i32 = sliced.scalar_at(1).try_into().unwrap();
        let val1019: i32 = sliced.scalar_at(1019).try_into().unwrap();
        assert_eq!(val0, 5i32);
        assert_eq!(val1, 6i32);
        assert_eq!(val1019, 1024i32); // This should be from second chunk
    }

    #[test]
    fn test_bitpacking_offset_with_partial_last_chunk() {
        // Test case: offset + partial last chunk
        let len = 1030usize; // 1024 + 6 elements 
        let offset = 5usize;

        let values = (0..len).map(|i| i as i32).collect::<BufferMut<_>>();
        let primitive_array = values.into_array().to_primitive().unwrap();
        let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Slice to get elements [5, 6, 7, ..., 1029] (1025 elements)
        let sliced = bitpacked.slice(offset, len);

        // Test values across the boundary and in the last partial chunk
        let val0: i32 = sliced.scalar_at(0).try_into().unwrap();
        let val1019: i32 = sliced.scalar_at(1019).try_into().unwrap(); // First element of second chunk
        let val1024: i32 = sliced.scalar_at(1024).try_into().unwrap(); // Last element (partial chunk)

        assert_eq!(val0, 5i32); // First element
        assert_eq!(val1019, 1024i32); // Element at chunk boundary  
        assert_eq!(val1024, 1029i32); // Last element in partial chunk
    }

    #[test]
    fn test_bitpacking_parent_pipeline() {
        let len = 10;
        let prim = (0i32..len).map(|x| x % 32).collect::<PrimitiveArray>();
        let mask = (0..len).map(|i| i % 32 != 0).collect::<Mask>();
        let bitpack = bitpack_to_best_bit_width(&prim).unwrap();
        let array = FoRArray::try_new(bitpack.to_array(), Scalar::from(100i32)).unwrap();

        let res = export_canonical_pipeline_expr(
            array.dtype(),
            array.len(),
            array.to_operator().unwrap().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_array();

        let expect = filter(array.as_ref(), &mask).unwrap();

        for i in 0..mask.true_count() {
            assert_eq!(res.scalar_at(i), expect.scalar_at(i), "{i}",);
        }
    }

    #[test]
    fn test_bitpacking_pipeline_with_offset() {
        let len = 1028usize;
        let offset = 1023usize;

        // Create simple sequential values for easier debugging
        let values = (0..len).map(|i| i as i32).collect::<PrimitiveArray>();
        let bitpacked = bitpack_encode(&values, 11, None).unwrap();

        let sliced = bitpacked.slice(offset, len);

        let mask = Mask::AllTrue(sliced.len());

        // Run through the pipeline
        let result = export_canonical_pipeline_expr(
            sliced.dtype(),
            sliced.len(),
            sliced.to_operator().unwrap().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_primitive()
        .unwrap();

        // Compare with expected result
        let expect = sliced.to_primitive().unwrap();

        assert_eq!(result.len(), expect.len(), "Length mismatch");
        assert_eq!(
            result.as_slice::<i32>(),
            expect.as_slice::<i32>(),
            "Null count mismatch"
        );
    }

    #[test]
    fn test_bitpacking_pipeline_with_offset_and_mask() {
        let len = 1028usize;
        let offset = 1023usize;

        // Create simple sequential values for easier debugging
        let values = (0..len).map(|i| i as i32).collect::<PrimitiveArray>();
        let bitpacked = bitpack_encode(&values, 11, None).unwrap();

        let sliced = bitpacked.slice(offset, len);

        // Use a simple mask that selects all elements to avoid mask complexity
        let mask = Mask::from_indices(5, vec![0, 2, 4]);

        // Run through the pipeline
        let result = export_canonical_pipeline_expr(
            sliced.dtype(),
            sliced.len(),
            sliced.to_operator().unwrap().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_primitive()
        .unwrap();

        // Compare with expected result
        let expect = filter(sliced.to_canonical().unwrap().as_ref(), &mask)
            .unwrap()
            .to_primitive()
            .unwrap();

        assert_eq!(result.len(), expect.len(), "Length mismatch");
        assert_eq!(
            result.as_slice::<i32>(),
            expect.as_slice::<i32>(),
            "Null count mismatch"
        );
    }

    #[test]
    fn test_bitpacking_pipeline_sparse_selection() {
        // Test with very sparse selection (< 8 elements selected)
        let len = 2048usize;

        let values = (0..len)
            .map(|i| (i as i32) * 3 + 17)
            .collect::<BufferMut<_>>();

        let primitive_array = values.into_array().to_primitive().unwrap();
        let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();

        // Test with offset
        let offset = 7;
        let sliced = bitpacked.slice(offset, len);
        let sliced_mask = Mask::from_buffer(BooleanBuffer::from(
            (0..sliced.len())
                .map(|i| {
                    let orig_idx = i + offset;
                    orig_idx == 10
                        || orig_idx == 500
                        || orig_idx == 1024
                        || orig_idx == 1500
                        || orig_idx == 2047
                })
                .collect::<Vec<bool>>(),
        ));

        let result = export_canonical_pipeline_expr(
            sliced.dtype(),
            sliced.len(),
            sliced.to_operator().unwrap().unwrap().as_ref(),
            &sliced_mask,
        )
        .unwrap()
        .into_array();

        let expect = filter(sliced.to_canonical().unwrap().as_ref(), &sliced_mask).unwrap();

        assert_eq!(result.len(), 5, "Should have exactly 5 selected elements");

        for i in 0..5 {
            assert_eq!(
                result.scalar_at(i),
                expect.scalar_at(i),
                "Sparse selection mismatch at index {}",
                i
            );
        }
    }
}
