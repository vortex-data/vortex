// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Mul;

use num_traits::One;
use vortex_array::ArrayRef;
use vortex_array::execution::{BatchKernel, BatchKernelRef, BindCtx, MaskExecution};
use vortex_array::vtable::OperatorVTable;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::AllOr;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::{Vector, VectorMutOps};

use crate::{SequenceArray, SequenceVTable};

impl OperatorVTable<SequenceVTable> for SequenceVTable {
    fn bind(
        array: &SequenceArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let selection = ctx.bind_selection(array.len(), selection)?;

        Ok(match_each_native_ptype!(array.ptype(), |T| {
            if array.multiplier().as_primitive::<T>() == <T as One>::one() {
                Box::new(SequenceKernel::<T> {
                    base: array.base().as_primitive::<T>(),
                    selection,
                })
            } else {
                Box::new(MultiplierSequenceKernel::<T> {
                    base: array.base().as_primitive::<T>(),
                    multiplier: array.multiplier().as_primitive::<T>(),
                    selection,
                })
            }
        }))
    }
}

struct SequenceKernel<T> {
    base: T,
    selection: MaskExecution,
}

impl<T: NativePType> BatchKernel for SequenceKernel<T> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let selection = self.selection.execute()?;

        let elements = match selection.indices() {
            AllOr::All => PVectorMut::<T>::from_iter((0..selection.len()).map(|i| {
                // This should never panic if the SequenceArray was constructed correctly
                let offset = T::from_usize(i).vortex_expect("Overflow converting usize to ptype");
                self.base + offset
            })),
            AllOr::None => PVectorMut::<T>::with_capacity(0),
            AllOr::Some(indices) => {
                PVectorMut::<T>::from_iter(indices.iter().map(|i| {
                    // This should never panic if the SequenceArray was constructed correctly
                    let offset =
                        T::from_usize(*i).vortex_expect("Overflow converting usize to ptype");
                    self.base + offset
                }))
            }
        };

        Ok(elements.freeze().into())
    }
}

struct MultiplierSequenceKernel<T> {
    base: T,
    multiplier: T,
    selection: MaskExecution,
}

impl<T: NativePType + Mul> BatchKernel for MultiplierSequenceKernel<T> {
    fn execute(self: Box<Self>) -> VortexResult<Vector> {
        let selection = self.selection.execute()?;

        let elements = match selection.indices() {
            AllOr::All => PVectorMut::<T>::from_iter((0..selection.len()).map(|i| {
                // This should never panic if the SequenceArray was constructed correctly
                let offset = T::from_usize(i).vortex_expect("Overflow converting usize to ptype");
                let scaled = self.multiplier * offset;
                self.base + scaled
            })),
            AllOr::None => PVectorMut::<T>::with_capacity(0),
            AllOr::Some(indices) => PVectorMut::<T>::from_iter(indices.iter().map(|&i| {
                // This should never panic if the SequenceArray was constructed correctly
                let offset = T::from_usize(i).vortex_expect("Overflow converting usize to ptype");
                let scaled = self.multiplier * offset;
                self.base + scaled
            })),
        };

        Ok(elements.freeze().into())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_dtype::{Nullability, PTypeDowncast};
    use vortex_vector::VectorOps;

    use crate::SequenceArray;

    #[test]
    fn test_sequence_operator_unit_multiplier() {
        // Test sequence with multiplier = 1: [2, 3, 4, 5]
        let seq = SequenceArray::typed_new(2i32, 1, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap().into_primitive().into_i32();

        assert_eq!(
            result.elements().as_slice(),
            buffer![2i32, 3, 4, 5].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_multiplier() {
        // Test sequence with multiplier = 3: [5, 8, 11, 14, 17]
        let seq = SequenceArray::typed_new(5i64, 3, Nullability::NonNullable, 5)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap().into_primitive().into_i64();

        assert_eq!(
            result.elements().as_slice(),
            buffer![5i64, 8, 11, 14, 17].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_negative_multiplier() {
        // Test sequence with negative multiplier: [10, 8, 6, 4]
        let seq = SequenceArray::typed_new(10i16, -2, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap().into_primitive().into_i16();

        assert_eq!(
            result.elements().as_slice(),
            buffer![10i16, 8, 6, 4].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_single_element() {
        // Test sequence with single element: [42]
        let seq = SequenceArray::typed_new(42i32, 1, Nullability::NonNullable, 1)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap().into_primitive().into_i32();

        assert_eq!(result.elements().as_slice(), buffer![42i32].as_slice());
    }

    #[test]
    fn test_sequence_operator_u64() {
        // Test with unsigned type: [100, 110, 120, 130]
        let seq = SequenceArray::typed_new(100u64, 10, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let result = seq.execute().unwrap().into_primitive().into_u64();

        assert_eq!(
            result.elements().as_slice(),
            buffer![100u64, 110, 120, 130].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_selection_alternating() {
        // Test sequence [10, 11, 12, 13, 14] with selection [1 0 1 0 1] => [10, 12, 14]
        let seq = SequenceArray::typed_new(10i32, 1, Nullability::NonNullable, 5)
            .unwrap()
            .into_array();

        let selection = bitbuffer![1 0 1 0 1].into_array();
        let result = seq
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_i32();

        assert_eq!(
            result.elements().as_slice(),
            buffer![10i32, 12, 14].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_selection_beginning() {
        // Test sequence [5, 8, 11, 14, 17] with selection [1 1 0 0 0] => [5, 8]
        let seq = SequenceArray::typed_new(5i64, 3, Nullability::NonNullable, 5)
            .unwrap()
            .into_array();

        let selection = bitbuffer![1 1 0 0 0].into_array();
        let result = seq
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_i64();

        assert_eq!(result.elements().as_slice(), buffer![5i64, 8].as_slice());
    }

    #[test]
    fn test_sequence_operator_with_selection_end() {
        // Test sequence [100, 110, 120, 130] with selection [0 0 1 1] => [120, 130]
        let seq = SequenceArray::typed_new(100u64, 10, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let selection = bitbuffer![0 0 1 1].into_array();
        let result = seq
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_u64();

        assert_eq!(
            result.elements().as_slice(),
            buffer![120u64, 130].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_selection_none() {
        // Test sequence [2, 3, 4, 5] with selection [0 0 0 0] => []
        let seq = SequenceArray::typed_new(2i32, 1, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let selection = bitbuffer![0 0 0 0].into_array();
        let result = seq.execute_with_selection(Some(&selection)).unwrap();
        assert!(result.is_empty())
    }

    #[test]
    fn test_sequence_operator_with_selection_all() {
        // Test sequence [10, 8, 6, 4] with selection [1 1 1 1] => [10, 8, 6, 4]
        let seq = SequenceArray::typed_new(10i16, -2, Nullability::NonNullable, 4)
            .unwrap()
            .into_array();

        let selection = bitbuffer![1 1 1 1].into_array();
        let result = seq
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_i16();

        assert_eq!(
            result.elements().as_slice(),
            buffer![10i16, 8, 6, 4].as_slice()
        );
    }

    #[test]
    fn test_sequence_operator_with_multiplier_and_selection() {
        // Test sequence [0, 5, 10, 15, 20, 25] with selection [1 0 0 1 0 1] => [0, 15, 25]
        let seq = SequenceArray::typed_new(0i32, 5, Nullability::NonNullable, 6)
            .unwrap()
            .into_array();

        let selection = bitbuffer![1 0 0 1 0 1].into_array();
        let result = seq
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_i32();

        assert_eq!(
            result.elements().as_slice(),
            buffer![0i32, 15, 25].as_slice()
        );
    }
}
