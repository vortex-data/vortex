// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod aligned_kernel;
// mod unaligned_kernel;

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use fastlanes::FastLanes;
use vortex_array::operator::{
    LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::pipeline::{BindContext, Kernel, PipelinedOperator, RowSelection};
use vortex_array::vtable::PipelineVTable;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, PhysicalPType, match_each_integer_ptype};
use vortex_error::VortexResult;

use crate::operator::aligned_kernel::BitPackedKernel;
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
        if array.offset != 0 {
            log::trace!("BitPackedVTable does not support non-zero offsets");
            return Ok(None);
        }

        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for BitPackedArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
        self.len.hash(state);
        self.dtype.hash(state);
        self.bit_width.hash(state);
        self.packed.operator_hash(state);
        // We don't care about patches because they're not yet supported by the operator.
        // OperatorHash(&self.patches).hash(state);
        self.validity.operator_hash(state);
    }
}

impl OperatorEq for BitPackedArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.offset == other.offset
            && self.len == other.len
            && self.dtype == other.dtype
            && self.bit_width == other.bit_width
            && self.packed.operator_eq(&other.packed)
            && self.validity.operator_eq(&other.validity)
    }
}

impl Operator for BitPackedArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn bounds(&self) -> LengthBounds {
        self.len.into()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }
}

impl PipelinedOperator for BitPackedArray {
    fn row_selection(&self) -> RowSelection {
        RowSelection::Domain(self.len)
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert!(self.bit_width > 0);
        match_each_integer_ptype!(self.ptype(), |T| {
            let packed_stride =
                self.bit_width as usize * <<T as PhysicalPType>::Physical as FastLanes>::LANES;
            let buffer = Buffer::<<T as PhysicalPType>::Physical>::from_byte_buffer(
                self.packed.clone().into_byte_buffer(),
            );

            if self.offset == 0 {
                Ok(Box::new(BitPackedKernel::<T>::new(
                    self.bit_width as usize,
                    packed_stride,
                    buffer,
                )) as Box<dyn Kernel>)
            } else {
                // TODO(ngates): the unaligned kernel needs fixing for the non-masked API
                // Ok(Box::new(BitPackedUnalignedKernel::<T>::new(
                //     self.bit_width as usize,
                //     packed_stride,
                //     buffer,
                //     0,
                //     self.offset,
                // )) as Box<dyn Kernel>)
                unreachable!("Offset must be zero")
            }
        })
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

//
// #[cfg(test)]
// mod tests {
//     use arrow_buffer::BooleanBuffer;
//     use rand::prelude::StdRng;
//     use rand::{Rng, SeedableRng};
//     use vortex_array::arrays::PrimitiveArray;
//     use vortex_array::compute::filter;
//     use vortex_array::operator::{export_canonical_pipeline_expr, N};
//     use vortex_array::{IntoArray, ToCanonical};
//     use vortex_buffer::BufferMut;
//     use vortex_mask::Mask;
//     use vortex_scalar::Scalar;
//
//     use crate::{bitpack_to_best_bit_width, FoRArray};
//
//     #[test]
//     fn test_bitpacking_pipeline() {
//         let frac = 0.5;
//         let len = 10;
//         let mut rng = StdRng::seed_from_u64(0);
//         let values = (0i16..len)
//             .map(|_| rng.random_range(0..100))
//             .collect::<BufferMut<_>>();
//
//         let primitive_array = values.into_array().to_primitive();
//         let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();
//
//         let mask = (0..len)
//             .map(|_| rng.random_bool(frac))
//             .collect::<BooleanBuffer>();
//         let mask = Mask::from_buffer(mask);
//
//         let result = export_canonical_pipeline_expr(
//             bitpacked.dtype(),
//             bitpacked.len(),
//             bitpacked.to_operator().unwrap().unwrap().as_ref(),
//             &mask,
//         )
//         .unwrap()
//         .into_array();
//
//         let expect = filter(bitpacked.to_canonical().as_ref(), &mask).unwrap();
//
//         assert_eq!(result.len(), expect.len());
//
//         for i in 0..mask.true_count() {
//             assert_eq!(
//                 result.scalar_at(i),
//                 expect.scalar_at(i),
//                 "mismatch at index {}",
//                 i,
//             );
//         }
//     }
//
//     #[test]
//     fn test_bitpacking_offset_simple() {
//         // Test a simple case: 1024 + 10 elements, offset by 5
//         let len = 1034usize;
//         let offset = 5usize;
//
//         let values = (0..len).map(|i| i as i32).collect::<BufferMut<_>>();
//         let primitive_array = values.into_array().to_primitive();
//         let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();
//
//         let sliced = bitpacked.slice(offset..offset + N);
//
//         // Just test first few elements manually
//         let val0: i32 = sliced.scalar_at(0).try_into().unwrap();
//         let val1: i32 = sliced.scalar_at(1).try_into().unwrap();
//         let val1019: i32 = sliced.scalar_at(1019).try_into().unwrap();
//         assert_eq!(val0, 5i32);
//         assert_eq!(val1, 6i32);
//         assert_eq!(val1019, 1024i32); // This should be from second chunk
//     }
//
//     #[test]
//     fn test_bitpacking_offset_with_partial_last_chunk() {
//         // Test case: offset + partial last chunk
//         let len = 1030usize; // 1024 + 6 elements
//         let offset = 5usize;
//
//         let values = (0..len).map(|i| i as i32).collect::<BufferMut<_>>();
//         let primitive_array = values.into_array().to_primitive();
//         let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();
//
//         let sliced = bitpacked.slice(offset..offset + N);
//
//         assert_eq!(i32::try_from(sliced.scalar_at(0)).unwrap(), 5i32); // First element
//         assert_eq!(i32::try_from(sliced.scalar_at(1019)).unwrap(), 1024i32); // Element at chunk boundary
//         assert_eq!(i32::try_from(sliced.scalar_at(1020)).unwrap(), 1025i32); // Element at chunk boundary
//         assert_eq!(i32::try_from(sliced.scalar_at(1023)).unwrap(), 1028i32); // Last element in partial chunk
//     }
//
//     #[test]
//     fn test_bitpacking_parent_pipeline() {
//         let len = 10;
//         let prim = (0i32..len).map(|x| x % 32).collect::<PrimitiveArray>();
//         let mask = (0..len).map(|i| i % 32 != 0).collect::<Mask>();
//         let bitpack = bitpack_to_best_bit_width(&prim).unwrap();
//         let array = FoRArray::try_new(bitpack.to_array(), Scalar::from(100i32)).unwrap();
//
//         let res = export_canonical_pipeline_expr(
//             array.dtype(),
//             array.len(),
//             array.to_operator().unwrap().unwrap().as_ref(),
//             &mask,
//         )
//         .unwrap()
//         .into_array();
//
//         let expect = filter(array.as_ref(), &mask).unwrap();
//
//         for i in 0..mask.true_count() {
//             assert_eq!(res.scalar_at(i), expect.scalar_at(i), "{i}",);
//         }
//     }
//
//     #[test]
//     fn test_bitpacking_pipeline_sparse_selection() {
//         // Test with very sparse selection (< 8 elements selected)
//         let len = 2048usize;
//
//         let values = (0..len)
//             .map(|i| (i as i32) * 3 + 17)
//             .collect::<BufferMut<_>>();
//
//         let primitive_array = values.into_array().to_primitive();
//         let bitpacked = bitpack_to_best_bit_width(&primitive_array).unwrap();
//
//         // Test with offset
//         let offset = 7;
//         let sliced = bitpacked.slice(offset..len);
//         let sliced_mask = Mask::from_buffer(BooleanBuffer::from(
//             (0..sliced.len())
//                 .map(|i| {
//                     let orig_idx = i + offset;
//                     orig_idx == 10
//                         || orig_idx == 500
//                         || orig_idx == 1024
//                         || orig_idx == 1500
//                         || orig_idx == 2047
//                 })
//                 .collect::<Vec<bool>>(),
//         ));
//
//         let result = export_canonical_pipeline_expr(
//             sliced.dtype(),
//             sliced.len(),
//             sliced.to_operator().unwrap().unwrap().as_ref(),
//             &sliced_mask,
//         )
//         .unwrap()
//         .into_array();
//
//         let expect = filter(sliced.to_canonical().as_ref(), &sliced_mask).unwrap();
//
//         assert_eq!(result.len(), 5, "Should have exactly 5 selected elements");
//
//         for i in 0..5 {
//             assert_eq!(
//                 result.scalar_at(i),
//                 expect.scalar_at(i),
//                 "Sparse selection mismatch at index {}",
//                 i
//             );
//         }
//     }
// }
