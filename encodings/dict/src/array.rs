use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::compute::{scalar_at, take, try_cast};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{CanonicalVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{
    encoding_ids, impl_encoding, Array, Canonical, IntoArray, IntoArrayVariant, IntoCanonical,
    SerdeMetadata,
};
use vortex_dtype::{match_each_integer_ptype, DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::{AllOr, Mask};

impl_encoding!(
    "vortex.dict",
    encoding_ids::DICT,
    Dict,
    SerdeMetadata<DictMetadata>
);

#[derive(
    Copy,
    Clone,
    Debug,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Portable,
    rkyv::Serialize,
    rkyv::Deserialize,
    rkyv::bytecheck::CheckBytes,
)]
#[rkyv(as = DictNullability)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(u8)]
enum DictNullability {
    NonNullable,
    NullableCodes,
    NullableValues,
    BothNullable,
}

impl DictNullability {
    fn from_dtypes(codes_dtype: &DType, values_dtype: &DType) -> Self {
        match (codes_dtype.is_nullable(), values_dtype.is_nullable()) {
            (true, true) => Self::BothNullable,
            (true, false) => Self::NullableCodes,
            (false, true) => Self::NullableValues,
            (false, false) => Self::NonNullable,
        }
    }

    fn codes_nullability(&self) -> Nullability {
        match self {
            DictNullability::NonNullable => Nullability::NonNullable,
            DictNullability::NullableCodes => Nullability::Nullable,
            DictNullability::NullableValues => Nullability::NonNullable,
            DictNullability::BothNullable => Nullability::Nullable,
        }
    }

    fn values_nullability(&self) -> Nullability {
        match self {
            DictNullability::NonNullable => Nullability::NonNullable,
            DictNullability::NullableCodes => Nullability::NonNullable,
            DictNullability::NullableValues => Nullability::Nullable,
            DictNullability::BothNullable => Nullability::Nullable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: usize, // TODO(ngates): make this a u32
    dict_nullability: DictNullability,
}

impl DictArray {
    pub fn try_new(codes: Array, values: Array) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", codes.dtype());
        }

        let dtype = if codes.dtype().is_nullable() {
            values.dtype().as_nullable()
        } else {
            values.dtype().clone()
        };
        let dict_nullability = DictNullability::from_dtypes(codes.dtype(), values.dtype());

        Self::try_from_parts(
            dtype,
            codes.len(),
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::try_from(codes.dtype())
                    .vortex_expect("codes dtype must be uint"),
                values_len: values.len(),
                dict_nullability,
            }),
            None,
            Some([codes, values].into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn codes(&self) -> Array {
        self.as_ref()
            .child(
                0,
                &DType::Primitive(
                    self.metadata().codes_ptype,
                    self.metadata().dict_nullability.codes_nullability(),
                ),
                self.len(),
            )
            .vortex_expect("DictArray is missing its codes child array")
    }

    #[inline]
    pub fn values(&self) -> Array {
        self.as_ref()
            .child(
                1,
                &self
                    .dtype()
                    .with_nullability(self.metadata().dict_nullability.values_nullability()),
                self.metadata().values_len,
            )
            .vortex_expect("DictArray is missing its values child array")
    }
}

impl ValidateVTable<DictArray> for DictEncoding {}

impl CanonicalVTable<DictArray> for DictEncoding {
    fn into_canonical(&self, array: DictArray) -> VortexResult<Canonical> {
        match array.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: Array = array.values().into_canonical()?.into_array();
                try_cast(take(canonical_values, array.codes())?, array.dtype())?.into_canonical()
            }
            // Non-string case: take and then canonicalize
            _ => try_cast(take(array.values(), array.codes())?, array.dtype())?.into_canonical(),
        }
    }
}

impl ValidityVTable<DictArray> for DictEncoding {
    fn is_valid(&self, array: &DictArray, index: usize) -> VortexResult<bool> {
        let scalar = scalar_at(array.codes(), index).map_err(|err| {
            err.with_context(format!(
                "Failed to get index {} from DictArray codes",
                index
            ))
        })?;

        if scalar.is_null() {
            return Ok(false);
        };
        let values_index: usize = scalar
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert dictionary code to usize");
        array.values().is_valid(values_index)
    }

    fn all_valid(&self, array: &DictArray) -> VortexResult<bool> {
        if !array.dtype().is_nullable() {
            return Ok(true);
        }

        Ok(array.codes().all_valid()? && array.values().all_valid()?)
    }

    fn validity_mask(&self, array: &DictArray) -> VortexResult<Mask> {
        let codes_validity = array.codes().validity_mask()?;
        match codes_validity.boolean_buffer() {
            AllOr::All => {
                let primitive_codes = array.codes().into_primitive()?;
                let values_mask = array.values().validity_mask()?;
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                    let codes_slice = primitive_codes.as_slice::<$P>();
                    BooleanBuffer::collect_bool(array.len(), |idx| {
                       values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            }
            AllOr::None => Ok(Mask::AllFalse(array.len())),
            AllOr::Some(validity_buff) => {
                let primitive_codes = array.codes().into_primitive()?;
                let values_mask = array.values().validity_mask()?;
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                    let codes_slice = primitive_codes.as_slice::<$P>();
                    BooleanBuffer::collect_bool(array.len(), |idx| {
                       validity_buff.value(idx) && values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            }
        }
    }
}

impl VisitorVTable<DictArray> for DictEncoding {
    fn accept(&self, array: &DictArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("values", &array.values())?;
        visitor.visit_child("codes", &array.codes())
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, SerdeMetadata};
    use vortex_buffer::buffer;
    use vortex_dtype::PType;
    use vortex_error::vortex_panic;
    use vortex_mask::AllOr;

    use crate::array::DictNullability::BothNullable;
    use crate::{DictArray, DictMetadata};

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::U64,
                values_len: usize::MAX,
                dict_nullability: BothNullable,
            }),
        );
    }

    #[test]
    fn nullable_codes_validity() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2, 2, 1],
                Validity::from(BooleanBuffer::from(vec![true, false, true, false, true])),
            )
            .into_array(),
            buffer![3, 6, 9].into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask().unwrap();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [0, 2, 4]);
    }

    #[test]
    fn nullable_values_validity() {
        let dict = DictArray::try_new(
            buffer![0u32, 1, 2, 2, 1].into_array(),
            PrimitiveArray::new(
                buffer![3, 6, 9],
                Validity::from(BooleanBuffer::from(vec![true, false, false])),
            )
            .into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask().unwrap();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [0]);
    }

    #[test]
    fn nullable_codes_and_values() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2, 2, 1],
                Validity::from(BooleanBuffer::from(vec![true, false, true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(
                buffer![3, 6, 9],
                Validity::from(BooleanBuffer::from(vec![false, true, true])),
            )
            .into_array(),
        )
        .unwrap();
        let mask = dict.validity_mask().unwrap();
        let AllOr::Some(indices) = mask.indices() else {
            vortex_panic!("Expected indices from mask")
        };
        assert_eq!(indices, [2, 4]);
    }
}
