use std::fmt::Debug;

use arrow_buffer::BooleanBuffer;
use vortex_array::builders::ArrayBuilder;
use vortex_array::compute::{scalar_at, take, take_into, try_cast};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    Canonical, Encoding, IntoArray, RkyvMetadata, ToCanonical,
};
use vortex_dtype::{DType, match_each_integer_ptype};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_mask::{AllOr, Mask};

use crate::serde::DictMetadata;

#[derive(Debug, Clone)]
pub struct DictArray {
    codes: ArrayRef,
    values: ArrayRef,
    stats_set: ArrayStats,
}

pub struct DictEncoding;
impl Encoding for DictEncoding {
    type Array = DictArray;
    type Metadata = RkyvMetadata<DictMetadata>;
}

impl DictArray {
    pub fn try_new(mut codes: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
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

        Ok(Self {
            codes,
            values,
            stats_set: Default::default(),
        })
    }

    #[inline]
    pub fn codes(&self) -> &ArrayRef {
        &self.codes
    }

    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }
}

impl ArrayImpl for DictArray {
    type Encoding = DictEncoding;

    fn _len(&self) -> usize {
        self.codes.len()
    }

    fn _dtype(&self) -> &DType {
        self.values.dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DictEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let codes = children[0].clone();
        let values = children[1].clone();

        Self::try_new(codes, values)
    }
}

impl ArrayCanonicalImpl for DictArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        match self.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: ArrayRef = self.values().to_canonical()?.into_array();
                take(&canonical_values, self.codes())?.to_canonical()
            }
            _ => take(self.values(), self.codes())?.to_canonical(),
        }
    }

    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        match self.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            // TODO(joe): is the above still true?, investigate this.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: ArrayRef = self.values().to_canonical()?.into_array();
                take_into(&canonical_values, self.codes(), builder)
            }
            // Non-string case: take and then canonicalize
            _ => take_into(self.values(), self.codes(), builder),
        }
    }
}

impl ArrayValidityImpl for DictArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        let scalar = scalar_at(self.codes(), index).map_err(|err| {
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
        self.values().is_valid(values_index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        if !self.dtype().is_nullable() {
            return Ok(true);
        }

        Ok(self.codes().all_valid()? && self.values().all_valid()?)
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        if !self.dtype().is_nullable() {
            return Ok(false);
        }

        Ok(self.codes().all_invalid()? || self.values().all_invalid()?)
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        let codes_validity = self.codes().validity_mask()?;
        match codes_validity.boolean_buffer() {
            AllOr::All => {
                let primitive_codes = self.codes().to_primitive()?;
                let values_mask = self.values().validity_mask()?;
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                    let codes_slice = primitive_codes.as_slice::<$P>();
                    BooleanBuffer::collect_bool(self.len(), |idx| {
                       values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            }
            AllOr::None => Ok(Mask::AllFalse(self.len())),
            AllOr::Some(validity_buff) => {
                let primitive_codes = self.codes().to_primitive()?;
                let values_mask = self.values().validity_mask()?;
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |$P| {
                    let codes_slice = primitive_codes.as_slice::<$P>();
                    BooleanBuffer::collect_bool(self.len(), |idx| {
                       validity_buff.value(idx) && values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Ok(Mask::from_buffer(is_valid_buffer))
            }
        }
    }
}

impl ArrayStatisticsImpl for DictArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use rand::distr::{Distribution, StandardUniform};
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::arrays::{ChunkedArray, PrimitiveArray};
    use vortex_array::builders::builder_with_capacity;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, NativePType, PType};
    use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
    use vortex_mask::AllOr;

    use crate::DictArray;

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
    ) -> ArrayRef
    where
        StandardUniform: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(0);

        (0..chunk_count)
            .map(|_| {
                let values = (0..unique_values)
                    .map(|_| rng.random::<T>())
                    .collect::<PrimitiveArray>();
                let codes = (0..len)
                    .map(|_| {
                        U::from(rng.random_range(0..unique_values)).vortex_expect("valid value")
                    })
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
            .append_to_builder(builder.as_mut())
            .vortex_unwrap();

        let into_prim = array.to_primitive().unwrap();
        let prim_into = builder.finish().to_primitive().unwrap();

        assert_eq!(into_prim.as_slice::<u64>(), prim_into.as_slice::<u64>());
        assert_eq!(
            into_prim.validity_mask().unwrap().boolean_buffer(),
            prim_into.validity_mask().unwrap().boolean_buffer()
        )
    }
}
