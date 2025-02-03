use std::fmt::{Debug, Display};

use vortex_array::array::BooleanBufferBuilder;
use vortex_array::compute::{scalar_at, sub_scalar};
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{StatisticsVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{encoding_ids, impl_encoding, Array, IntoArray, IntoArrayVariant, RkyvMetadata};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarValue};

mod canonical;
mod compute;
mod variants;

impl_encoding!(
    "vortex.sparse",
    encoding_ids::SPARSE,
    Sparse,
    RkyvMetadata<SparseMetadata>
);

#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[repr(C)]
pub struct SparseMetadata {
    // Offset value for patch indices as a result of slicing
    indices_offset: usize,
    patches: PatchesMetadata,
}

impl Display for SparseMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl SparseArray {
    pub fn try_new(
        indices: Array,
        values: Array,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        Self::try_new_with_offset(indices, values, len, 0, fill_value)
    }

    pub(crate) fn try_new_with_offset(
        indices: Array,
        values: Array,
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

        let fill_value_buffer = fill_value.into_value().to_flexbytes();

        Self::try_from_parts(
            patches.dtype().clone(),
            len,
            RkyvMetadata(SparseMetadata {
                indices_offset,
                patches: patches_metadata,
            }),
            Some([fill_value_buffer.into_inner()].into()),
            Some([patches.indices().clone(), patches.values().clone()].into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn indices_offset(&self) -> usize {
        self.metadata().indices_offset
    }

    #[inline]
    pub fn patches(&self) -> Patches {
        let patches = self.metadata().patches;
        let indices = self
            .as_ref()
            .child(0, &patches.indices_dtype(), patches.len())
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
        let indices_offset = Scalar::from(self.indices_offset()).cast(indices.dtype())?;
        let indices = sub_scalar(indices, indices_offset)?;
        Ok(Patches::new(len, indices, values))
    }

    #[inline]
    pub fn fill_scalar(&self) -> Scalar {
        let sv = ScalarValue::from_flexbytes(
            self.as_ref()
                .byte_buffer(0)
                .vortex_expect("Missing fill value buffer"),
        )
        .vortex_expect("Failed to deserialize fill value");
        Scalar::new(self.dtype().clone(), sv)
    }
}

impl ValidateVTable<SparseArray> for SparseEncoding {}

impl VisitorVTable<SparseArray> for SparseEncoding {
    fn accept(&self, array: &SparseArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_patches(&array.patches())
    }
}

impl StatisticsVTable<SparseArray> for SparseEncoding {
    fn compute_statistics(&self, array: &SparseArray, stat: Stat) -> VortexResult<StatsSet> {
        let values = array.patches().into_values();
        let stats = values.statistics().compute_all(&[stat])?;
        if array.len() == values.len() {
            return Ok(stats);
        }

        let fill_len = array.len() - values.len();
        let fill_stats = if array.fill_scalar().is_null() {
            StatsSet::nulls(fill_len, array.dtype())
        } else {
            StatsSet::constant(array.fill_scalar(), fill_len)
        };

        if values.is_empty() {
            return Ok(fill_stats);
        }

        Ok(stats.merge_unordered(&fill_stats, array.dtype()))
    }
}

impl ValidityVTable<SparseArray> for SparseEncoding {
    fn is_valid(&self, array: &SparseArray, index: usize) -> VortexResult<bool> {
        Ok(match array.patches().get_patched(index)? {
            None => array.fill_scalar().is_valid(),
            Some(patch_value) => patch_value.is_valid(),
        })
    }

    fn all_valid(&self, array: &SparseArray) -> VortexResult<bool> {
        if array.fill_scalar().is_null() {
            // We need _all_ values to be patched, and all patches to be valid
            return Ok(array.patches().values().len() == array.len()
                && array.patches().values().all_valid()?);
        }

        array.patches().values().all_valid()
    }

    fn validity_mask(&self, array: &SparseArray) -> VortexResult<Mask> {
        let indices = array.patches().indices().clone().into_primitive()?;

        if array.fill_scalar().is_null() {
            // If we have a null fill value, then we set each patch value to true.
            let mut buffer = BooleanBufferBuilder::new(array.len());
            // TODO(ngates): use vortex-buffer::BitBufferMut when it exists.
            buffer.append_n(array.len(), false);

            match_each_integer_ptype!(indices.ptype(), |$I| {
                indices.as_slice::<$I>().into_iter().for_each(|&index| {
                    buffer.set_bit(index.try_into().vortex_expect("Failed to cast to usize"), true);
                });
            });

            return Ok(Mask::from_buffer(buffer.finish()));
        }

        // If the fill_value is non-null, then the validity is based on the validity of the
        // patch values.
        let mut buffer = BooleanBufferBuilder::new(array.len());
        buffer.append_n(array.len(), true);

        let values_validity = array.patches().values().validity_mask()?;
        match_each_integer_ptype!(indices.ptype(), |$I| {
            indices.as_slice::<$I>()
                .into_iter()
                .enumerate()
                .for_each(|(patch_idx, &index)| {
                    buffer.set_bit(index.try_into().vortex_expect("failed to cast to usize"), values_validity.value(patch_idx));
                })
        });

        Ok(Mask::from_buffer(buffer.finish()))
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::array::ConstantArray;
    use vortex_array::compute::{slice, try_cast};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, PType};
    use vortex_error::VortexError;
    use vortex_scalar::{PrimitiveScalar, Scalar};

    use super::*;

    fn nullable_fill() -> Scalar {
        Scalar::null(DType::Primitive(PType::I32, Nullable))
    }

    fn non_nullable_fill() -> Scalar {
        Scalar::from(42i32)
    }

    fn sparse_array(fill_value: Scalar) -> Array {
        // merged array: [null, null, 100, null, null, 200, null, null, 300, null]
        let mut values = buffer![100i32, 200, 300].into_array();
        values = try_cast(&values, fill_value.dtype()).unwrap();

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
    pub fn sparse_validity_mask() {
        let array = sparse_array(nullable_fill());
        assert_eq!(
            array
                .validity_mask()
                .unwrap()
                .to_boolean_buffer()
                .iter()
                .collect_vec(),
            [false, false, true, false, false, true, false, false, true, false]
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
}
