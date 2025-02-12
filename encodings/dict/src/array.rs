use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use serde::{Deserialize, Serialize};
use vortex_array::builders::ArrayBuilder;
use vortex_array::compute::{scalar_at, take, take_into};
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{CanonicalVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use vortex_array::{
    encoding_ids, impl_encoding, Array, Canonical, IntoArray, IntoArrayVariant, IntoCanonical,
    SerdeMetadata,
};
use vortex_dtype::{match_each_integer_ptype, DType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

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
    pub fn try_new(codes: Array, values: Array) -> VortexResult<Self> {
        if !codes.dtype().is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", codes.dtype());
        }
        Self::try_from_parts(
            values.dtype().clone(),
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
            // TODO(joe): is the above still true?
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
        // If the values are all valid, then the dictionary must be all valid
        if array.values().all_valid()? {
            return Ok(true);
        }

        // Otherwise, find the null code.
        // For now, we assume this must be code zero.
        assert!(scalar_at(array.values(), 0)?.is_null());

        // Attempt to short-circuit with a min statistic
        if let Some(min) = array.codes().statistics().compute_as::<u64>(Stat::Min) {
            return Ok(min > 0);
        }

        // Otherwise, check each code
        let primitive_codes = array.codes().into_primitive()?;
        match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
            for code in primitive_codes.as_slice::<$P>() {
                if *code == 0 {
                    return Ok(false);
                }
            }
        });

        Ok(true)
    }

    fn validity_mask(&self, array: &DictArray) -> VortexResult<Mask> {
        if array.dtype().is_nullable() {
            let primitive_codes = array.codes().into_primitive()?;
            match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                // This is correct since the code will be 0 if the value is null.
                let is_valid = primitive_codes
                    .as_slice::<$P>();
                let is_valid_buffer = BooleanBuffer::collect_bool(is_valid.len(), |idx| {
                    is_valid[idx] != 0
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            })
        } else {
            Ok(Mask::AllTrue(array.len()))
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
    use rand::distributions::{Distribution, Standard};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::array::{ChunkedArray, PrimitiveArray};
    use vortex_array::builders::builder_with_capacity;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::{Array, IntoArray, IntoArrayVariant, IntoCanonical, SerdeMetadata};
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, NativePType, PType};
    use vortex_error::{VortexExpect, VortexUnwrap};

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
        println!(
            "{:?}",
            array.clone().into_primitive().unwrap().as_slice::<u64>()
        );

        let mut builder = builder_with_capacity(
            &DType::Primitive(PType::U64, NonNullable),
            len * chunk_count,
        );
        array.canonicalize_into(builder.as_mut()).vortex_unwrap();
        println!(
            "{:?}",
            builder
                .finish()
                .vortex_unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>()
        );
    }
}
