// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use itertools::Itertools as _;
use num_traits::NumCast;
use vortex_array::arrays::{BooleanBufferBuilder, ConstantArray};
use vortex_array::compute::{Operator, compare, fill_null, filter, sub_scalar};
use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{ArrayVTable, NotSupported, VTable, ValidityVTable};
use vortex_array::{Array, ArrayRef, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_ensure};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

mod canonical;
mod compute;
mod ops;
mod serde;

vtable!(Sparse);

impl VTable for SparseVTable {
    type Array = SparseArray;
    type Encoding = SparseEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.sparse")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(SparseEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct SparseArray {
    patches: Patches,
    fill_value: Scalar,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct SparseEncoding;

impl SparseArray {
    pub fn try_new(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            indices.len() == values.len(),
            "Mismatched indices {} and values {} length",
            indices.len(),
            values.len()
        );

        vortex_ensure!(
            indices.statistics().compute_is_strict_sorted() == Some(true),
            "SparseArray: indices must be strict-sorted"
        );

        // Verify the indices are all in the valid range
        if !indices.is_empty() {
            let last_index = usize::try_from(&indices.scalar_at(indices.len() - 1))?;

            vortex_ensure!(
                last_index < len,
                "Array length was {len} but the last index is {last_index}"
            );
        }

        let patches = Patches::new(len, 0, indices, values);

        Ok(Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        })
    }

    /// Build a new SparseArray from an existing set of patches.
    pub fn try_new_from_patches(patches: Patches, fill_value: Scalar) -> VortexResult<Self> {
        vortex_ensure!(
            fill_value.dtype() == patches.values().dtype(),
            "fill value, {:?}, should be instance of values dtype, {} but was {}.",
            fill_value,
            patches.values().dtype(),
            fill_value.dtype(),
        );

        Ok(Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(patches: Patches, fill_value: Scalar) -> Self {
        Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        }
    }

    #[inline]
    pub fn patches(&self) -> &Patches {
        &self.patches
    }

    #[inline]
    pub fn resolved_patches(&self) -> Patches {
        let patches = self.patches();
        let indices_offset = Scalar::from(patches.offset())
            .cast(patches.indices().dtype())
            .vortex_expect("Patches offset must cast to the indices dtype");
        let indices = sub_scalar(patches.indices(), indices_offset)
            .vortex_expect("must be able to subtract offset from indices");
        Patches::new(patches.array_len(), 0, indices, patches.values().clone())
    }

    #[inline]
    pub fn fill_scalar(&self) -> &Scalar {
        &self.fill_value
    }

    /// Encode given array as a SparseArray.
    ///
    /// Optionally provided fill value will be respected if the array is less than 90% null.
    pub fn encode(array: &dyn Array, fill_value: Option<Scalar>) -> VortexResult<ArrayRef> {
        if let Some(fill_value) = fill_value.as_ref()
            && array.dtype() != fill_value.dtype()
        {
            vortex_bail!(
                "Array and fill value types must match. got {} and {}",
                array.dtype(),
                fill_value.dtype()
            )
        }
        let mask = array.validity_mask();

        if mask.all_false() {
            // Array is constant NULL
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        } else if mask.false_count() as f64 > (0.9 * mask.len() as f64) {
            // Array is dominated by NULL but has non-NULL values
            let non_null_values = filter(array, &mask)?;
            let non_null_indices = match mask.indices() {
                AllOr::All => {
                    // We already know that the mask is 90%+ false
                    unreachable!("Mask is mostly null")
                }
                AllOr::None => {
                    // we know there are some non-NULL values
                    unreachable!("Mask is mostly null but not all null")
                }
                AllOr::Some(values) => {
                    let buffer: Buffer<u32> = values
                        .iter()
                        .map(|&v| v.try_into().vortex_expect("indices must fit in u32"))
                        .collect();

                    buffer.into_array()
                }
            };

            return Ok(SparseArray::try_new(
                non_null_indices,
                non_null_values,
                array.len(),
                Scalar::null(array.dtype().clone()),
            )?
            .into_array());
        }

        let fill = if let Some(fill) = fill_value {
            fill
        } else {
            // TODO(robert): Support other dtypes, only thing missing is getting most common value out of the array
            let (top_pvalue, _) = array
                .to_primitive()
                .top_value()?
                .vortex_expect("Non empty or all null array");

            Scalar::primitive_value(top_pvalue, top_pvalue.ptype(), array.dtype().nullability())
        };

        let fill_array = ConstantArray::new(fill.clone(), array.len()).into_array();
        let non_top_mask = Mask::from_buffer(
            fill_null(
                &compare(array, &fill_array, Operator::NotEq)?,
                &Scalar::bool(true, Nullability::NonNullable),
            )?
            .to_bool()
            .boolean_buffer()
            .clone(),
        );

        let non_top_values = filter(array, &non_top_mask)?;

        let indices: Buffer<u64> = match non_top_mask {
            Mask::AllTrue(count) => {
                // all true -> complete slice
                (0u64..count as u64).collect()
            }
            Mask::AllFalse(_) => {
                // All values are equal to the top value
                return Ok(fill_array);
            }
            Mask::Values(values) => values.indices().iter().map(|v| *v as u64).collect(),
        };

        SparseArray::try_new(indices.into_array(), non_top_values, array.len(), fill)
            .map(|a| a.into_array())
    }
}

impl ArrayVTable<SparseVTable> for SparseVTable {
    fn len(array: &SparseArray) -> usize {
        array.patches.array_len()
    }

    fn dtype(array: &SparseArray) -> &DType {
        array.fill_scalar().dtype()
    }

    fn stats(array: &SparseArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl ValidityVTable<SparseVTable> for SparseVTable {
    fn is_valid(array: &SparseArray, index: usize) -> bool {
        match array.patches().get_patched(index) {
            None => array.fill_scalar().is_valid(),
            Some(patch_value) => patch_value.is_valid(),
        }
    }

    fn all_valid(array: &SparseArray) -> bool {
        if array.fill_scalar().is_null() {
            // We need _all_ values to be patched, and all patches to be valid
            return array.patches().values().len() == array.len()
                && array.patches().values().all_valid();
        }

        array.patches().values().all_valid()
    }

    fn all_invalid(array: &SparseArray) -> bool {
        if !array.fill_scalar().is_null() {
            // We need _all_ values to be patched, and all patches to be invalid
            return array.patches().values().len() == array.len()
                && array.patches().values().all_invalid();
        }

        array.patches().values().all_invalid()
    }

    #[allow(clippy::unnecessary_fallible_conversions)]
    fn validity_mask(array: &SparseArray) -> Mask {
        let fill_is_valid = array.fill_scalar().is_valid();
        let values_validity = array.patches().values().validity_mask();
        let len = array.len();

        if matches!(values_validity, Mask::AllTrue(_)) && fill_is_valid {
            return Mask::AllTrue(len);
        }
        if matches!(values_validity, Mask::AllFalse(_)) && !fill_is_valid {
            return Mask::AllFalse(len);
        }

        // TODO(ngates): use vortex-buffer::BitBufferMut when it exists.
        let mut is_valid_buffer = BooleanBufferBuilder::new(len);
        is_valid_buffer.append_n(len, fill_is_valid);

        let indices = array.patches().indices().to_primitive();
        let index_offset = array.patches().offset();

        match_each_integer_ptype!(indices.ptype(), |I| {
            let indices = indices.as_slice::<I>();
            patch_validity(&mut is_valid_buffer, indices, index_offset, values_validity);
        });

        Mask::from_buffer(is_valid_buffer.finish())
    }
}

fn patch_validity<I: NativePType>(
    is_valid_buffer: &mut BooleanBufferBuilder,
    indices: &[I],
    index_offset: usize,
    values_validity: Mask,
) {
    let indices = indices.iter().map(|index| {
        let index = <usize as NumCast>::from(*index).vortex_expect("Failed to cast to usize");
        index - index_offset
    });
    match values_validity {
        Mask::AllTrue(_) => {
            for index in indices {
                is_valid_buffer.set_bit(index, true);
            }
        }
        Mask::AllFalse(_) => {
            for index in indices {
                is_valid_buffer.set_bit(index, false);
            }
        }
        Mask::Values(mask_values) => {
            let is_valid = mask_values.boolean_buffer().iter();
            for (index, is_valid) in indices.zip_eq(is_valid) {
                is_valid_buffer.set_bit(index, is_valid);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::IntoArray;
    use vortex_array::arrays::{ConstantArray, PrimitiveArray};
    use vortex_array::compute::cast;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexUnwrap;
    use vortex_scalar::{PrimitiveScalar, Scalar};

    use super::*;

    fn nullable_fill() -> Scalar {
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
    }

    fn non_nullable_fill() -> Scalar {
        Scalar::from(42i32)
    }

    fn sparse_array(fill_value: Scalar) -> ArrayRef {
        // merged array: [null, null, 100, null, null, 200, null, null, 300, null]
        let mut values = buffer![100i32, 200, 300].into_array();
        values = cast(&values, fill_value.dtype()).unwrap();

        SparseArray::try_new(buffer![2u64, 5, 8].into_array(), values, 10, fill_value)
            .unwrap()
            .into_array()
    }

    #[test]
    pub fn test_scalar_at() {
        let array = sparse_array(nullable_fill());

        assert_eq!(array.scalar_at(0), nullable_fill());
        assert_eq!(array.scalar_at(2), Scalar::from(Some(100_i32)));
        assert_eq!(array.scalar_at(5), Scalar::from(Some(200_i32)));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_at_oob() {
        let array = sparse_array(nullable_fill());
        let _ = array.scalar_at(10);
    }

    #[test]
    pub fn test_scalar_at_again() {
        let arr = SparseArray::try_new(
            ConstantArray::new(10u32, 1).into_array(),
            ConstantArray::new(Scalar::primitive(1234u32, Nullability::Nullable), 1).into_array(),
            100,
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
        )
        .unwrap();

        assert_eq!(
            PrimitiveScalar::try_from(&arr.scalar_at(10))
                .unwrap()
                .typed_value::<u32>(),
            Some(1234)
        );
        assert!(arr.scalar_at(0).is_null());
        assert!(arr.scalar_at(99).is_null());
    }

    #[test]
    pub fn scalar_at_sliced() {
        let sliced = sparse_array(nullable_fill()).slice(2..7);
        assert_eq!(usize::try_from(&sliced.scalar_at(0)).unwrap(), 100);
    }

    #[test]
    pub fn validity_mask_sliced_null_fill() {
        let sliced = sparse_array(nullable_fill()).slice(2..7);
        assert_eq!(
            sliced.validity_mask(),
            Mask::from_iter(vec![true, false, false, true, false])
        );
    }

    #[test]
    pub fn validity_mask_sliced_nonnull_fill() {
        let sliced = SparseArray::try_new(
            buffer![2u64, 5, 8].into_array(),
            ConstantArray::new(
                Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
                3,
            )
            .into_array(),
            10,
            Scalar::primitive(1.0f32, Nullability::Nullable),
        )
        .unwrap()
        .slice(2..7);

        assert_eq!(
            sliced.validity_mask(),
            Mask::from_iter(vec![false, true, true, false, true])
        );
    }

    #[test]
    pub fn scalar_at_sliced_twice() {
        let sliced_once = sparse_array(nullable_fill()).slice(1..8);
        assert_eq!(usize::try_from(&sliced_once.scalar_at(1)).unwrap(), 100);

        let sliced_twice = sliced_once.slice(1..6);
        assert_eq!(usize::try_from(&sliced_twice.scalar_at(3)).unwrap(), 200);
    }

    #[test]
    pub fn sparse_validity_mask() {
        let array = sparse_array(nullable_fill());
        assert_eq!(
            array
                .validity_mask()
                .to_boolean_buffer()
                .iter()
                .collect_vec(),
            [
                false, false, true, false, false, true, false, false, true, false
            ]
        );
    }

    #[test]
    fn sparse_validity_mask_non_null_fill() {
        let array = sparse_array(non_nullable_fill());
        assert!(array.validity_mask().all_true());
    }

    #[test]
    #[should_panic]
    fn test_invalid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 100, 0_u32.into()).unwrap();
    }

    #[test]
    fn test_valid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 101, 0_u32.into()).unwrap();
    }

    #[test]
    fn encode_with_nulls() {
        let sparse = SparseArray::encode(
            &PrimitiveArray::new(
                buffer![0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4],
                Validity::from_iter(vec![
                    true, true, false, true, false, true, false, true, true, false, true, false,
                ]),
            )
            .into_array(),
            None,
        )
        .vortex_unwrap();
        let canonical = sparse.to_primitive();
        assert_eq!(
            sparse.validity_mask(),
            Mask::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ])
        );
        assert_eq!(
            canonical.as_slice::<i32>(),
            vec![0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4]
        );
    }

    #[test]
    fn validity_mask_includes_null_values_when_fill_is_null() {
        let indices = buffer![0u8, 2, 4, 6, 8].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i16), Some(1), None, None, Some(4)])
            .into_array();
        let array = SparseArray::try_new(indices, values, 10, Scalar::null_typed::<i16>()).unwrap();
        let actual = array.validity_mask();
        let expected = Mask::from_iter([
            true, false, true, false, false, false, false, false, true, false,
        ]);

        assert_eq!(actual, expected);
    }
}
