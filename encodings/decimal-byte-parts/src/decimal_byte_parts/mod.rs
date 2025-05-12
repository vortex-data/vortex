mod compute;
mod serde;

use std::iter;

use itertools::Itertools;
use vortex_array::arrays::DecimalArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayOperationsImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, Canonical, Encoding, ProstMetadata, try_from_array_ref,
};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, DecimalDType, PType, match_each_signed_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::decimal_byte_parts::serde::DecimalBytesPartsMetadata;

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// The most significant part (msp) sorting the most significant decimal bits.
/// This array must be signed and is nullable iff the decimal is nullable.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower_part\[0\] = 64..0
#[derive(Clone, Debug)]
pub struct DecimalBytePartsArray {
    msp: ArrayRef,
    lower_parts: Vec<ArrayRef>,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<Self> {
        if !lower_parts.is_empty() {
            // TODO(joe): remove this constraint.
            vortex_bail!("DecimalBytePartsArray doesn't support lower parts arrays")
        }

        if !(0usize..=3).contains(&lower_parts.iter().len()) {
            vortex_bail!(
                "DecimalBytePartsArray lower_parts must have between 0..=3 children, instead given: {}",
                lower_parts.len()
            )
        }

        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        if lower_parts
            .iter()
            .any(|a| a.dtype() != &DType::Primitive(PType::U64, NonNullable))
        {
            vortex_bail!("decimal bytes parts 2nd to 4th must be non-nullable u64 primitive typed")
        }

        let primitive_bit_width = iter::once(&msp)
            .chain(&lower_parts)
            .map(|a| {
                PType::try_from(a.dtype())
                    .vortex_expect("already checked")
                    .bit_width()
            })
            .sum();

        if decimal_dtype.bit_width() > primitive_bit_width {
            vortex_bail!(
                "cannot represent a decimal {decimal_dtype} as primitive parts {:?}",
                iter::once(&msp)
                    .chain(&lower_parts)
                    .map(|a| a.dtype())
                    .collect_vec()
            )
        }

        let nullable = msp.dtype().nullability();
        Ok(Self {
            msp,
            lower_parts,
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

impl Encoding for DecimalBytePartsEncoding {
    type Array = DecimalBytePartsArray;
    type Metadata = ProstMetadata<DecimalBytesPartsMetadata>;
}

impl ArrayImpl for DecimalBytePartsArray {
    type Encoding = DecimalBytePartsEncoding;

    fn _len(&self) -> usize {
        self.msp.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DecimalBytePartsEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let msp = children[0].clone();
        let lower_parts = children.iter().skip(1).cloned().collect_vec();
        DecimalBytePartsArray::try_new(msp, lower_parts, *self.decimal_dtype())
    }
}

impl ArrayCanonicalImpl for DecimalBytePartsArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        // TODO(joe): support parts len != 1
        assert!(self.lower_parts.is_empty());
        let prim = self.msp.to_canonical()?.into_primitive()?;
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
            self.msp.slice(start, stop)?,
            self.lower_parts
                .iter()
                .map(|p| p.slice(start, stop))
                .try_collect()?,
            *self.decimal_dtype(),
        )
        .map(|d| d.to_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        // TODO(joe): suppor parts len != 1
        assert!(self.lower_parts.is_empty());
        let scalar = self.msp.scalar_at(index)?;

        scalar.cast(self.dtype())
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
        self.msp.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.msp.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.msp.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.msp.validity_mask()
    }
}

try_from_array_ref!(DecimalBytePartsArray);
