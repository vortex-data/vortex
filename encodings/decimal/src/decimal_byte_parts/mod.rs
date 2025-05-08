mod compute;
mod serde;

use itertools::Itertools;
use vortex_array::arrays::DecimalArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::{DecimalArrayTrait, PrimitiveArrayTrait};
use vortex_array::vtable::{ComputeVTable, VTableRef};
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, Canonical, Encoding, ProstMetadata, try_from_array_ref,
};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, DecimalDType, PType, match_each_signed_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::decimal_byte_parts::serde::DecimalBytesPartsMetadata;

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// These are stored as big endian.
///  e.g. for a decimal i128 [ 64..127 | 0..64] child[0] = 64..127 and child[1] = 0..64
/// The child[0] sorting the most significant decimal bits must be signed and is nullable iff the
/// decimal is nullable.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsArray {
    parts: Vec<ArrayRef>,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(parts: Vec<ArrayRef>, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        if parts.is_empty() || parts.len() > 4 {
            vortex_bail!(
                "DecimalBytePartsArray must have between 1..=4 children, instead given: {}",
                parts.len()
            )
        }

        // For now only signed integer types are supported, this can be relaxed in the future.
        if !parts[0].dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        if parts
            .iter()
            .skip(1)
            .any(|a| a.dtype() != &DType::Primitive(PType::U64, NonNullable))
        {
            vortex_bail!("decimal bytes parts 2nd to 4th must be non-nullable u64 primitive typed")
        }

        let primitive_bit_width = parts
            .iter()
            .map(|a| {
                PType::try_from(a.dtype())
                    .vortex_expect("already checked")
                    .bit_width()
            })
            .sum();

        if decimal_dtype.bit_width() > primitive_bit_width {
            vortex_bail!(
                "cannot represent a decimal {decimal_dtype} as primitive parts {:?}",
                parts.iter().map(|a| a.dtype()).collect_vec()
            )
        }

        let nullable = parts[0].dtype().nullability();
        Ok(Self {
            parts,
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        })
    }

    pub fn decimal_dtype(&self) -> &DecimalDType {
        self.dtype
            .as_decimal()
            .vortex_expect("must be a decimal dtype")
    }
}

#[derive(Debug)]
pub struct DecimalBytePartsEncoding;

impl ComputeVTable for DecimalBytePartsEncoding {}

impl Encoding for DecimalBytePartsEncoding {
    type Array = DecimalBytePartsArray;
    type Metadata = ProstMetadata<DecimalBytesPartsMetadata>;
}

impl ArrayImpl for DecimalBytePartsArray {
    type Encoding = DecimalBytePartsEncoding;

    fn _len(&self) -> usize {
        self.parts[0].len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DecimalBytePartsEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        DecimalBytePartsArray::try_new(children.to_vec(), *self.decimal_dtype())
    }
}

impl DecimalArrayTrait for DecimalBytePartsArray {}

impl ArrayVariantsImpl for DecimalBytePartsArray {
    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }
}

impl ArrayCanonicalImpl for DecimalBytePartsArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        // TODO(joe): support parts len != 1
        assert!(self.parts.len() == 1);
        let prim = self.parts[0].to_canonical()?.into_primitive()?;
        // Depending on the decimal type and the min/max of the primitive array we can choose
        // the correct buffer size

        let res = match_each_signed_integer_ptype!(prim.ptype(), |$P| {
           Canonical::Decimal(DecimalArray::new(
                prim.buffer::<$P>(),
                self.decimal_dtype().clone(),
                prim.validity().clone(),
            ))
        });

        Ok(res)
    }
}

impl ArrayOperationsImpl for DecimalBytePartsArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(
            self.parts
                .iter()
                .map(|p| p.slice(start, stop))
                .try_collect()?,
            *self.decimal_dtype(),
        )
        .map(|d| d.to_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        // TODO(joe): suppor parts len != 1
        assert!(self.parts.len() == 1);
        let scalar = self.parts[0].scalar_at(index)?;

        scalar.cast(self.dtype())
        // scalar
        //     .as_primitive()
        //     .typed_value::<i32>()
        //     .map(|v| Scalar::new(self.dtype().clone(), DecimalValue::from(v).into()))
        //     .unwrap_or_else(|| Scalar::null(self.dtype().clone()))
    }
}

impl ArrayStatisticsImpl for DecimalBytePartsArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for DecimalBytePartsArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        // validity stored in 0th child
        self.parts[0].is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.parts[0].all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.parts[0].all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.parts[0].validity_mask()
    }
}

try_from_array_ref!(DecimalBytePartsArray);
