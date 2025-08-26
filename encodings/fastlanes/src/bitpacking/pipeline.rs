// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use fastlanes::{BitPacking, FastLanes};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::operators::{BindContext, Operator};
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Element, Kernel, KernelContext, N, PipelineVTable, VType};
use vortex_buffer::Buffer;
use vortex_dtype::{PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedVTable};

impl PipelineVTable<BitPackedVTable> for BitPackedVTable {
    fn to_operator(array: &BitPackedArray) -> VortexResult<Option<Rc<dyn Operator>>> {
        if array.dtype.is_nullable() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }
        if array.patches.is_some() {
            log::trace!("BitPackedVTable does not support nullable arrays");
            return Ok(None);
        }

        Ok(Some(Rc::new(array.clone())))
    }
}

impl Operator for BitPackedArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype())
    }

    fn children(&self) -> &[Rc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, _children: Vec<Rc<dyn Operator>>) -> Rc<dyn Operator> {
        Rc::new(self.clone())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert!(self.bit_width > 0);
        match_each_integer_ptype!(self.ptype(), |T| {
            let packed_stride =
                self.bit_width as usize * <<T as PhysicalPType>::Physical as FastLanes>::LANES;
            Ok(Box::new(BitPackedKernel::<T> {
                width: self.bit_width as usize,
                packed_stride,
                buffer: Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
                    self.packed.clone().into_byte_buffer(),
                ),
                packed_offset: 0,
                value_offset: self.offset as usize,
            }) as Box<dyn Kernel>)
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
    value_offset: usize,
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

        // Handle partial chunk due to value offset
        let chunk_value_offset = self.value_offset % 1024;
        let needs_partial_handling = chunk_value_offset != 0;

        // We compute the number of FastLanes vectors that we need.
        // If we have an offset, we need to read enough chunks to cover all the output elements
        let elements_needed = elements.len() + chunk_value_offset;
        let chunks_needed = elements_needed.div_ceil(1024);
        let nvecs = chunks_needed.min(packed.len() / self.packed_stride).min(N / 1024);

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if selected.true_count() > 8 {
            if needs_partial_handling {
                // For the first chunk with offset, unpack to local buffer then copy
                let mut local_buffer = [unsafe { std::mem::zeroed() }; 1024];
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &packed[..self.packed_stride],
                        &mut local_buffer[..],
                    );
                }

                // Copy from local buffer starting at the offset position
                let copy_count = (1024 - chunk_value_offset).min(elements.len());
                elements[..copy_count].copy_from_slice(
                    &local_buffer[chunk_value_offset..chunk_value_offset + copy_count],
                );

                // Handle remaining full chunks normally
                for i in 1..nvecs {
                    let start_idx = copy_count + (i - 1) * 1024;
                    let remaining_elements = elements.len().saturating_sub(start_idx);
                    let chunk_size = remaining_elements.min(1024);
                    
                    if chunk_size > 0 {
                        unsafe {
                            BitPacking::unchecked_unpack(
                                self.width,
                                &packed[(i * self.packed_stride)..][..self.packed_stride],
                                &mut elements[start_idx..start_idx + chunk_size],
                            );
                        }
                    }
                }
            } else {
                // Normal full chunk unpacking
                for i in 0..nvecs {
                    let start_idx = i * 1024;
                    let end_idx = start_idx + 1024;
                    if end_idx <= elements.len() {
                        unsafe {
                            BitPacking::unchecked_unpack(
                                self.width,
                                &packed[(i * self.packed_stride)..][..self.packed_stride],
                                &mut elements[start_idx..end_idx],
                            );
                        }
                    }
                }
            }

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.select_mask::<<T as PhysicalPType>::Physical>(&selected);
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let adjusted_idx = idx + chunk_value_offset;
                let chunk_idx = adjusted_idx / 1024;
                let bit_idx = adjusted_idx % 1024;

                unsafe {
                    *elements.get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
                        self.width,
                        &packed[(chunk_idx * self.packed_stride)..][..self.packed_stride],
                        bit_idx,
                    );
                }
                offset += 1;
            });

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

    use crate::{FoRArray, bitpack_to_best_bit_width};

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
}
