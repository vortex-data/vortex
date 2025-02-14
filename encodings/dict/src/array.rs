use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::builders::ArrayBuilder;
use vortex_array::compute::{scalar_at, take, take_into, try_cast};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{CanonicalVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{
    encoding_ids, impl_encoding, Array, Canonical, IntoArray, IntoArrayVariant, IntoCanonical,
    SerdeMetadata,
};
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::{AllOr, Mask};

impl_encoding!(
    "vortex.dict",
    encoding_ids::DICT,
    Dict,
    SerdeMetadata<DictMetadata>
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: usize, // TODO(ngates): make this a u32
}

impl DictArray {
    pub fn try_new(mut codes: Array, values: Array) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", codes.dtype());
        }

        let dtype = values.dtype();
        if dtype.is_nullable() {
            // If the values are nullable, we force codes to be nullable as well.
            codes = try_cast(&codes, &codes.dtype().as_nullable())?;
        } else {
            // If the values are non-nullable, we assert the codes are non-nullable as well.
            if codes.dtype().is_nullable() {
                vortex_bail!("Cannot have nullable codes for non-nullable dict array");
            }
        }
        assert_eq!(
            codes.dtype().nullability(),
            values.dtype().nullability(),
            "Mismatched nullability between codes and values"
        );

        Self::try_from_parts(
            dtype.clone(),
            codes.len(),
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::try_from(codes.dtype())
                    .vortex_expect("codes dtype must be uint"),
                values_len: values.len(),
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
                &DType::Primitive(self.metadata().codes_ptype, self.dtype().nullability()),
                self.len(),
            )
            .vortex_expect("DictArray is missing its codes child array")
    }

    #[inline]
    pub fn values(&self) -> Array {
        self.as_ref()
            .child(1, self.dtype(), self.metadata().values_len)
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
                take(canonical_values, array.codes())?.into_canonical()
            }
            // Non-string case: take and then canonicalize
            _ => take(array.values(), array.codes())?.into_canonical(),
        }
    }

    fn canonicalize_into(
        &self,
        array: DictArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        match array.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            // TODO(joe): is the above still true?, investigate this.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: Array = array.values().into_canonical()?.into_array();
                take_into(canonical_values, array.codes(), builder)
            }
            // Non-string case: take and then canonicalize
            _ => take_into(array.values(), array.codes(), builder),
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

    fn all_invalid(&self, array: &DictArray) -> VortexResult<bool> {
        if !array.dtype().is_nullable() {
            return Ok(false);
        }

        Ok(array.codes().all_invalid()? || array.values().all_invalid()?)
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
    use rand::distributions::{Distribution, Standard};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::array::{ChunkedArray, PrimitiveArray};
    use vortex_array::builders::builder_with_capacity;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, IntoArrayVariant, IntoCanonical, SerdeMetadata};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, NativePType, PType};
    use vortex_error::{vortex_panic, VortexExpect, VortexUnwrap};
    use vortex_mask::AllOr;

    use crate::{DictArray, DictMetadata};

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            SerdeMetadata(DictMetadata {
                codes_ptype: PType::U64,
                values_len: usize::MAX,
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
            PrimitiveArray::new(buffer![3, 6, 9], Validity::AllValid).into_array(),
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

    fn make_dict_primitive_chunks<T: NativePType, U: NativePType>(
        len: usize,
        unique_values: usize,
        chunk_count: usize,
    ) -> Array
    where
        Standard: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(0);

        (0..chunk_count)
            .map(|_| {
                let values = (0..unique_values)
                    .map(|_| rng.gen::<T>())
                    .collect::<PrimitiveArray>();
                let codes = (0..len)
                    .map(|_| U::from(rng.gen_range(0..unique_values)).vortex_expect("valid value"))
                    .collect::<PrimitiveArray>();

                DictArray::try_new(codes.into_array(), values.into_array())
                    .vortex_unwrap()
                    .into_array()
            })
            .collect::<ChunkedArray>()
            .into_array()
    }

    #[test]
    fn test_dict_array_from_primitive_chunks() {
        let len = 2;
        let chunk_count = 2;
        let array = make_dict_primitive_chunks::<u64, u64>(len, 2, chunk_count);

        let mut builder = builder_with_capacity(
            &DType::Primitive(PType::U64, NonNullable),
            len * chunk_count,
        );
        array
            .clone()
            .canonicalize_into(builder.as_mut())
            .vortex_unwrap();

        let into_prim = array.into_primitive().unwrap();
        let prim_into = builder.finish().unwrap().into_primitive().unwrap();

        assert_eq!(into_prim.as_slice::<u64>(), prim_into.as_slice::<u64>());
        assert_eq!(
            into_prim.validity_mask().unwrap().boolean_buffer(),
            prim_into.validity_mask().unwrap().boolean_buffer()
        )
    }
}
