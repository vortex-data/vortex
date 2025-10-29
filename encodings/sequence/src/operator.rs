// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Mul;

use num_traits::One;
use vortex_array::ArrayRef;
use vortex_array::execution::{BatchKernel, BatchKernelRef, BindCtx};
use vortex_array::vtable::OperatorVTable;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::{PVectorMut, Vector, VectorMutOps};

use crate::{SequenceArray, SequenceVTable};

impl OperatorVTable<SequenceVTable> for SequenceVTable {
    fn bind(
        array: &SequenceArray,
        _selection: Option<&ArrayRef>,
        _ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        Ok(match_each_native_ptype!(array.ptype(), |T| {
            if array.multiplier().as_primitive::<T>() == <T as One>::one() {
                Box::new(SequenceKernel::<T> {
                    base: array.base().as_primitive::<T>(),
                    len: array.len(),
                })
            } else {
                Box::new(MultiplierSequenceKernel::<T> {
                    base: array.base().as_primitive::<T>(),
                    multiplier: array.multiplier().as_primitive::<T>(),
                    len: array.len(),
                })
            }
        }))
    }
}

struct SequenceKernel<T> {
    base: T,
    len: usize,
}

impl<T: NativePType> BatchKernel for SequenceKernel<T> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        Ok(PVectorMut::<T>::from_iter((0..self.len).map(|i| {
            // This should never panic if the SequenceArray was constructed correctly
            let offset = T::from_usize(i).vortex_expect("Overflow converting usize to ptype");
            self.base + offset
        }))
        .freeze()
        .into())
    }
}

struct MultiplierSequenceKernel<T> {
    base: T,
    multiplier: T,
    len: usize,
}

impl<T: NativePType + Mul> BatchKernel for MultiplierSequenceKernel<T> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        Ok(PVectorMut::<T>::from_iter((0..self.len).map(|i| {
            // This should never panic if the SequenceArray was constructed correctly
            let offset = T::from_usize(i).vortex_expect("Overflow converting usize to ptype");
            let scaled = self.multiplier * offset;
            self.base + scaled
        }))
        .freeze()
        .into())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{Nullability, PTypeDowncast};

    use crate::SequenceArray;

    #[test]
    fn test_sequence_operator_unit_multiplier() {
        // Test sequence with multiplier = 1: [2, 3, 4, 5]
        let seq = SequenceArray::typed_new(2i32, 1, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap();
        let prim = result.as_primitive().clone().into_i32();

        assert_eq!(
            prim.elements().as_slice(),
            buffer![2i32, 3, 4, 5].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_multiplier() {
        // Test sequence with multiplier = 3: [5, 8, 11, 14, 17]
        let seq = SequenceArray::typed_new(5i64, 3, Nullability::NonNullable, 5)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap();
        let prim = result.as_primitive().clone().into_i64();

        assert_eq!(
            prim.elements().as_slice(),
            buffer![5i64, 8, 11, 14, 17].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_negative_multiplier() {
        // Test sequence with negative multiplier: [10, 8, 6, 4]
        let seq = SequenceArray::typed_new(10i16, -2, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap();
        let prim = result.as_primitive().clone().into_i16();

        assert_eq!(
            prim.elements().as_slice(),
            buffer![10i16, 8, 6, 4].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_single_element() {
        // Test sequence with single element: [42]
        let seq = SequenceArray::typed_new(42i32, 1, Nullability::NonNullable, 1)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap();
        let prim = result.as_primitive().clone().into_i32();

        assert_eq!(prim.elements().as_slice(), buffer![42i32].as_slice());
    }

    #[test]
    fn test_sequence_operator_u64() {
        // Test with unsigned type: [100, 110, 120, 130]
        let seq = SequenceArray::typed_new(100u64, 10, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap();
        let prim = result.as_primitive().clone().into_u64();

        assert_eq!(
            prim.elements().as_slice(),
            buffer![100u64, 110, 120, 130].as_slice()
        );
    }
}
