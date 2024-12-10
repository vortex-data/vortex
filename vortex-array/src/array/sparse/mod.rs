use std::fmt::{Debug, Display};

use ::serde::{Deserialize, Serialize};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::array::constant::ConstantArray;
use crate::compute::{scalar_at, subtract_scalar};
use crate::encoding::ids;
use crate::patches::{Patches, PatchesMetadata};
use crate::stats::{ArrayStatistics, Stat, StatisticsVTable, StatsSet};
use crate::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, IntoArrayData};

mod canonical;
mod compute;
mod variants;

impl_encoding!("vortex.sparse", ids::SPARSE, Sparse);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMetadata {
    // Offset value for patch indices as a result of slicing
    indices_offset: usize,
    patches: PatchesMetadata,
    fill_value: ScalarValue,
}

impl Display for SparseMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl SparseArray {
    pub fn try_new(
        indices: ArrayData,
        values: ArrayData,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        Self::try_new_with_offset(indices, values, len, 0, fill_value)
    }

    pub(crate) fn try_new_with_offset(
        indices: ArrayData,
        values: ArrayData,
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

        let patches = Patches::new(len, indices, values);

        Self::try_new_from_patches(patches, len, indices_offset, fill_value)
    }

    pub fn try_new_from_patches(
        patches: Patches,
        len: usize,
        indices_offset: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        if fill_value.dtype() != patches.values().dtype() {
            vortex_bail!(
                "fill value, {:?}, should be instance of values dtype, {}",
                fill_value,
                patches.values().dtype(),
            );
        }

        let patches_metadata = patches.to_metadata(len, patches.dtype())?;

        Self::try_from_parts(
            patches.dtype().clone(),
            len,
            SparseMetadata {
                indices_offset,
                patches: patches_metadata,
                fill_value: fill_value.into_value(),
            },
            [patches.indices().clone(), patches.values().clone()].into(),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn indices_offset(&self) -> usize {
        self.metadata().indices_offset
    }

    #[inline]
    pub fn patches(&self) -> Patches {
        let indices = self
            .as_ref()
            .child(
                0,
                &self.metadata().patches.indices_dtype(),
                self.metadata().patches.len(),
            )
            .vortex_expect("Missing indices array in SparseArray");
        let values = self
            .as_ref()
            .child(1, self.dtype(), self.metadata().patches.len())
            .vortex_expect("Missing values array in SparseArray");
        Patches::new(self.len(), indices, values)
    }

    #[inline]
    pub fn resolved_patches(&self) -> VortexResult<Patches> {
        let (len, indices, values) = self.patches().into_parts();
        let indices = subtract_scalar(indices, &Scalar::from(self.indices_offset()))?;
        Ok(Patches::new(len, indices, values))
    }

    #[inline]
    pub fn fill_scalar(&self) -> Scalar {
        Scalar::new(self.dtype().clone(), self.metadata().fill_value.clone())
    }
}

impl ArrayTrait for SparseArray {}

impl VisitorVTable<SparseArray> for SparseEncoding {
    fn accept(&self, array: &SparseArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_patches(&array.patches())
    }
}

impl StatisticsVTable<SparseArray> for SparseEncoding {
    fn compute_statistics(&self, array: &SparseArray, stat: Stat) -> VortexResult<StatsSet> {
        let values = array.patches().into_values();
        let mut stats = values.statistics().compute_all(&[stat])?;
        if array.len() == values.len() {
            return Ok(stats);
        }

        let fill_len = array.len() - values.len();
        let fill_stats = if array.fill_scalar().is_null() {
            StatsSet::nulls(fill_len, array.dtype())
        } else {
            StatsSet::constant(&array.fill_scalar(), fill_len)
        };

        if values.is_empty() {
            return Ok(fill_stats);
        }

        stats.merge_unordered(&fill_stats);
        Ok(stats)
    }
}

impl ValidityVTable<SparseArray> for SparseEncoding {
    fn is_valid(&self, array: &SparseArray, index: usize) -> bool {
        match array.patches().get_patched(index) {
            Ok(None) => array.fill_scalar().is_valid(),
            Ok(Some(patch_value)) => patch_value.is_valid(),
            Err(e) => vortex_panic!(e, "Error while finding index {} in sparse array", index),
        }
    }

    fn logical_validity(&self, array: &SparseArray) -> LogicalValidity {
        let validity = if array.fill_scalar().is_null() {
            // If we have a null fill value, then the result is a Sparse array with a fill_value
            // of true, and patch values of false.
            SparseArray::try_new_from_patches(
                array
                    .patches()
                    .map_values(|values| Ok(ConstantArray::new(true, values.len()).into_array()))
                    .vortex_expect("constant array has same length as values array"),
                array.len(),
                array.indices_offset(),
                false.into(),
            )
        } else {
            // If the fill_value is non-null, then the validity is based on the validity of the
            // existing values.
            SparseArray::try_new_from_patches(
                array
                    .patches()
                    .map_values(|values| Ok(values.logical_validity().into_array()))
                    .vortex_expect("logical validity preserves length"),
                array.len(),
                array.indices_offset(),
                true.into(),
            )
        }
        .vortex_expect("Error determining logical validity for sparse array");
        LogicalValidity::Array(validity.into_array())
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, PType};
    use vortex_error::VortexError;
    use vortex_scalar::{PrimitiveScalar, Scalar};

    use crate::array::sparse::SparseArray;
    use crate::array::ConstantArray;
    use crate::compute::{scalar_at, slice, try_cast};
    use crate::validity::ArrayValidity;
    use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

    fn nullable_fill() -> Scalar {
        Scalar::null(DType::Primitive(PType::I32, Nullable))
    }

    fn non_nullable_fill() -> Scalar {
        Scalar::from(42i32)
    }

    fn sparse_array(fill_value: Scalar) -> ArrayData {
        // merged array: [null, null, 100, null, null, 200, null, null, 300, null]
        let mut values = vec![100i32, 200, 300].into_array();
        values = try_cast(&values, fill_value.dtype()).unwrap();

        SparseArray::try_new(vec![2u64, 5, 8].into_array(), values, 10, fill_value)
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
            ConstantArray::new(Scalar::primitive(1234u32, Nullable), 1).into_array(),
            100,
            Scalar::null(DType::Primitive(PType::U32, Nullable)),
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
        let sliced = slice(sparse_array(nullable_fill()), 2, 7).unwrap();
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
    pub fn scalar_at_sliced_twice() {
        let sliced_once = slice(sparse_array(nullable_fill()), 1, 8).unwrap();
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

        let sliced_twice = slice(&sliced_once, 1, 6).unwrap();
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
    pub fn sparse_logical_validity() {
        let array = sparse_array(nullable_fill());
        let validity = array.logical_validity().into_array().into_bool().unwrap();
        assert_eq!(
            validity.boolean_buffer().iter().collect_vec(),
            [false, false, true, false, false, true, false, false, true, false]
        );
    }

    #[test]
    fn sparse_logical_validity_non_null_fill() {
        let array = sparse_array(non_nullable_fill());

        assert_eq!(
            array
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true; 10]
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_length() {
        let values = vec![15_u32, 135, 13531, 42].into_array();
        let indices = vec![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 100, 0_u32.into()).unwrap();
    }

    #[test]
    fn test_valid_length() {
        let values = vec![15_u32, 135, 13531, 42].into_array();
        let indices = vec![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 101, 0_u32.into()).unwrap();
    }
}
