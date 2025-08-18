// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::{Poll, ready};

use fastlanes::{BitPacking, FastLanes};
use vortex_array::pipeline::PipelineVTable;
use vortex_dtype::{PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_vector::bits::BitView;
use vortex_vector::buffers::BufferHandle;
use vortex_vector::operators::{BindContext, Operator};
use vortex_vector::types::{Element, VType};
use vortex_vector::view::ViewMut;
use vortex_vector::{Kernel, KernelContext, PIPELINE_STEP_COUNT};

use crate::{BitPackedArray, BitPackedVTable};

impl PipelineVTable<BitPackedVTable> for BitPackedVTable {
    fn to_operator(array: &BitPackedArray) -> VortexResult<Option<Arc<dyn Operator>>> {
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

    fn children(&self) -> &[Arc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, _children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        Arc::new(self.clone())
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert!(self.bit_width > 0, "{}", self.as_ref().display_tree());
        match_each_integer_ptype!(self.ptype(), |T| {
            Ok(Box::new(BitPackedKernel::<T> {
                width: self.bit_width as usize,
                packed_stride: self.bit_width as usize
                    * <<T as PhysicalPType>::Physical as FastLanes>::LANES,
                buffer: BufferHandle::new(self.packed.clone())
                    .into_typed::<<T as PhysicalPType>::Physical>(),
                packed_offset: 0,
            }))
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

    buffer: BufferHandle<<T as PhysicalPType>::Physical>,
    packed_offset: usize,
}

impl<T> Kernel for BitPackedKernel<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (PIPELINE_STEP_COUNT / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        // We re-interpret the output view as the unsigned bitpacked type.
        let mut physical_out = out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = physical_out.as_slice_mut::<<T as PhysicalPType>::Physical>();
        let packed = &buffer.as_slice()[self.packed_offset..];

        // We compute the number of FastLanes vectors that we have remaining.
        let nvecs = (PIPELINE_STEP_COUNT / 1024).min(packed.len() / self.packed_stride);

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if selected.true_count() > 8 {
            for i in 0..nvecs {
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &packed[(i * self.packed_stride)..][..self.packed_stride],
                        &mut elements[(i * 1024)..],
                    );
                }
            }

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            physical_out.select_mask::<<T as PhysicalPType>::Physical>(&selected);
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let chunk_idx = idx / 1024;
                let bit_idx = idx % 1024;
                // SAFETY: we verify the bounds of the vector during construction.
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

        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::filter;
    
    use vortex_array::pipeline::canonical::export_canonical_pipeline_expr;
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
                    result.scalar_at(i).unwrap(),
                    expect.scalar_at(i).unwrap(),
                    "mismatch at index {}, fraction {}",
                    i,
                    frac
                );
            }
        }
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
            assert_eq!(
                res.scalar_at(i).unwrap(),
                expect.scalar_at(i).unwrap(),
                "{i}",
            );
        }
    }
}
