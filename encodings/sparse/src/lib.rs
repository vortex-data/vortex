use std::fmt::Debug;

use vortex_array::arrays::{BooleanBufferBuilder, ConstantArray};
use vortex_array::compute::{Operator, compare, fill_null, filter, scalar_at, sub_scalar};
use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl, Encoding, IntoArray,
    ProstMetadata, ToCanonical, try_from_array_ref,
};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::serde::SparseMetadata;

mod canonical;
mod compute;
mod ops;
mod serde;
mod variants;

#[derive(Clone, Debug)]
pub struct SparseArray {
    patches: Patches,
    fill_value: Scalar,
    stats_set: ArrayStats,
}

try_from_array_ref!(SparseArray);

#[derive(Debug)]
pub struct SparseEncoding;
impl Encoding for SparseEncoding {
    type Array = SparseArray;
    type Metadata = ProstMetadata<SparseMetadata>;
}

impl SparseArray {
    pub fn try_new(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        Self::try_new_with_offset(indices, values, len, 0, fill_value)
    }

    pub(crate) fn try_new_with_offset(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        indices_offset: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        if indices.len() != values.len() {
            vortex_bail!(
                "Mismatched indices {} and values {} length",
                indices.len(),
                values.len()
            );
        }

        if !indices.is_empty() {
            let last_index = usize::try_from(&scalar_at(&indices, indices.len() - 1)?)?;

            if last_index - indices_offset >= len {
                vortex_bail!("Array length was set to {len} but the last index is {last_index}");
            }
        }

        let patches = Patches::new(len, indices_offset, indices, values);

        Self::try_new_from_patches(patches, fill_value)
    }

    pub fn try_new_from_patches(patches: Patches, fill_value: Scalar) -> VortexResult<Self> {
        if fill_value.dtype() != patches.values().dtype() {
            vortex_bail!(
                "fill value, {:?}, should be instance of values dtype, {}",
                fill_value,
                patches.values().dtype(),
            );
        }
        Ok(Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        })
    }

    #[inline]
    pub fn patches(&self) -> &Patches {
        &self.patches
    }

    #[inline]
    pub fn resolved_patches(&self) -> VortexResult<Patches> {
        let (len, offset, indices, values) = self.patches().clone().into_parts();
        let indices_offset = Scalar::from(offset).cast(indices.dtype())?;
        let indices = sub_scalar(&indices, indices_offset)?;
        Ok(Patches::new(len, 0, indices, values))
    }

    #[inline]
    pub fn fill_scalar(&self) -> &Scalar {
        &self.fill_value
    }

    /// Encode given array as a SparseArray.
    ///
    /// Optionally provided fill value will be respected if the array is less than 90% null.
    pub fn encode(array: &dyn Array, fill_value: Option<Scalar>) -> VortexResult<ArrayRef> {
        if let Some(fill_value) = fill_value.as_ref() {
            if array.dtype() != fill_value.dtype() {
                vortex_bail!(
                    "Array and fill value types must match. got {} and {}",
                    array.dtype(),
                    fill_value.dtype()
                )
            }
        }
        let mask = array.validity_mask()?;

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
                .to_primitive()?
                .top_value()?
                .vortex_expect("Non empty or all null array");

            Scalar::primitive_value(top_pvalue, top_pvalue.ptype(), array.dtype().nullability())
        };

        let fill_array = ConstantArray::new(fill.clone(), array.len()).into_array();
        let non_top_mask = Mask::from_buffer(
            fill_null(
                &compare(array, &fill_array, Operator::NotEq)?,
                Scalar::bool(true, Nullability::NonNullable),
            )?
            .to_bool()?
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

        SparseArray::try_new(
            indices.into_array(),
            non_top_values.into_array(),
            array.len(),
            fill,
        )
        .map(|a| a.into_array())
    }
}

impl ArrayImpl for SparseArray {
    type Encoding = SparseEncoding;

    fn _len(&self) -> usize {
        self.patches.array_len()
    }

    fn _dtype(&self) -> &DType {
        self.fill_value.dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&SparseEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let patch_indices = children[0].clone();
        let patch_values = children[1].clone();

        let patches = Patches::new(
            self.patches().array_len(),
            self.patches().offset(),
            patch_indices,
            patch_values,
        );

        Self::try_new_from_patches(patches, self.fill_value.clone())
    }
}

impl ArrayStatisticsImpl for SparseArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for SparseArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        Ok(match self.patches().get_patched(index)? {
            None => self.fill_scalar().is_valid(),
            Some(patch_value) => patch_value.is_valid(),
        })
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        if self.fill_scalar().is_null() {
            // We need _all_ values to be patched, and all patches to be valid
            return Ok(self.patches().values().len() == self.len()
                && self.patches().values().all_valid()?);
        }

        self.patches().values().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        if !self.fill_scalar().is_null() {
            // We need _all_ values to be patched, and all patches to be invalid
            return Ok(self.patches().values().len() == self.len()
                && self.patches().values().all_invalid()?);
        }

        self.patches().values().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        let indices = self.patches().indices().to_primitive()?;

        if self.fill_scalar().is_null() {
            // If we have a null fill value, then we set each patch value to true.
            let mut buffer = BooleanBufferBuilder::new(self.len());
            // TODO(ngates): use vortex-buffer::BitBufferMut when it exists.
            buffer.append_n(self.len(), false);

            match_each_integer_ptype!(indices.ptype(), |$I| {
                indices.as_slice::<$I>().into_iter().for_each(|&index| {
                    buffer.set_bit(usize::try_from(index).vortex_expect("Failed to cast to usize") - self.patches().offset(), true);
                });
            });

            return Ok(Mask::from_buffer(buffer.finish()));
        }

        // If the fill_value is non-null, then the validity is based on the validity of the
        // patch values.
        let mut buffer = BooleanBufferBuilder::new(self.len());
        buffer.append_n(self.len(), true);

        let values_validity = self.patches().values().validity_mask()?;
        match_each_integer_ptype!(indices.ptype(), |$I| {
            indices.as_slice::<$I>()
                .into_iter()
                .enumerate()
                .for_each(|(patch_idx, &index)| {
                    buffer.set_bit(usize::try_from(index).vortex_expect("Failed to cast to usize") - self.patches().offset(), values_validity.value(patch_idx));
                })
        });

        Ok(Mask::from_buffer(buffer.finish()))
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
    use vortex_error::{VortexError, VortexUnwrap};
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

        assert_eq!(scalar_at(&array, 0).unwrap(), nullable_fill());
        assert_eq!(scalar_at(&array, 2).unwrap(), Scalar::from(Some(100_i32)));
        assert_eq!(scalar_at(&array, 5).unwrap(), Scalar::from(Some(200_i32)));

        let error = scalar_at(&array, 10).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 10);
        assert_eq!(start, 0);
        assert_eq!(stop, 10);
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
            PrimitiveScalar::try_from(&scalar_at(&arr, 10).unwrap())
                .unwrap()
                .typed_value::<u32>(),
            Some(1234)
        );
        assert!(scalar_at(&arr, 0).unwrap().is_null());
        assert!(scalar_at(&arr, 99).unwrap().is_null());
    }

    #[test]
    pub fn scalar_at_sliced() {
        let sliced = sparse_array(nullable_fill()).slice(2, 7).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced, 0).unwrap()).unwrap(),
            100
        );
        let error = scalar_at(&sliced, 5).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 5);
        assert_eq!(start, 0);
        assert_eq!(stop, 5);
    }

    #[test]
    pub fn validity_mask_sliced_null_fill() {
        let sliced = sparse_array(nullable_fill()).slice(2, 7).unwrap();
        assert_eq!(
            sliced.validity_mask().unwrap(),
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
        .slice(2, 7)
        .unwrap();

        assert_eq!(
            sliced.validity_mask().unwrap(),
            Mask::from_iter(vec![false, true, true, false, true])
        );
    }

    #[test]
    pub fn scalar_at_sliced_twice() {
        let sliced_once = sparse_array(nullable_fill()).slice(1, 8).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced_once, 1).unwrap()).unwrap(),
            100
        );
        let error = scalar_at(&sliced_once, 7).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error else {
            unreachable!()
        };
        assert_eq!(i, 7);
        assert_eq!(start, 0);
        assert_eq!(stop, 7);

        let sliced_twice = sliced_once.slice(1, 6).unwrap();
        assert_eq!(
            usize::try_from(&scalar_at(&sliced_twice, 3).unwrap()).unwrap(),
            200
        );
        let error2 = scalar_at(&sliced_twice, 5).err().unwrap();
        let VortexError::OutOfBounds(i, start, stop, _) = error2 else {
            unreachable!()
        };
        assert_eq!(i, 5);
        assert_eq!(start, 0);
        assert_eq!(stop, 5);
    }

    #[test]
    pub fn sparse_validity_mask() {
        let array = sparse_array(nullable_fill());
        assert_eq!(
            array
                .validity_mask()
                .unwrap()
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
        assert!(array.validity_mask().unwrap().all_true());
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
        let canonical = sparse.to_primitive().vortex_unwrap();
        assert_eq!(
            sparse.validity_mask().unwrap(),
            Mask::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ])
        );
        assert_eq!(
            canonical.as_slice::<i32>(),
            vec![0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4]
        );
    }
}
