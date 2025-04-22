mod serde;

use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;

use crate::array::{Array, ArrayCanonicalImpl, ArrayValidityImpl, ArrayVariantsImpl};
use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::variants::DecimalArrayTrait;
use crate::vtable::{ComputeVTable, VTableRef};
use crate::{
    ArrayComputeImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayVisitorImpl, Canonical,
    EmptyMetadata, Encoding, try_from_array_ref,
};

pub struct DecimalEncoding;

impl ComputeVTable for DecimalEncoding {}

impl Encoding for DecimalEncoding {
    type Array = DecimalArray;
    type Metadata = EmptyMetadata;
}

/// Array for decimal-typed real numbers
#[derive(Clone, Debug)]
pub struct DecimalArray {
    dtype: DType,
    values: ByteBuffer,
    validity: Validity,
    stats_set: ArrayStats,
}

try_from_array_ref!(DecimalArray);

impl DecimalArray {
    /// Creates a new [`DecimalArray`] from a [`ByteBuffer`] and [`Validity`], without checking
    /// any invariants.
    pub fn new(buffer: ByteBuffer, decimal_dtype: DecimalDType, validity: Validity) -> Self {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() / 16 != len {
                // Assuming 128-bit (16-byte) decimal representation
                vortex_panic!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len() / 16,
                    len
                );
            }
        }

        Self {
            dtype: DType::Decimal(decimal_dtype, validity.nullability()),
            values: buffer,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns the underlying [`ByteBuffer`] of the array.
    pub fn byte_buffer(&self) -> &ByteBuffer {
        &self.values
    }

    /// Returns the underlying [`Validity`] of the array.
    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    /// Returns the decimal type information
    pub fn decimal_dtype(&self) -> DecimalDType {
        match &self.dtype {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => vortex_panic!("Expected Decimal dtype, got {:?}", self.dtype),
        }
    }

    pub fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    pub fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }
}

impl ArrayComputeImpl for DecimalArray {}

impl ArrayVisitorImpl<EmptyMetadata> for DecimalArray {
    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl ArrayImpl for DecimalArray {
    type Encoding = DecimalEncoding;

    #[inline]
    fn _len(&self) -> usize {
        self.values.len() / 16 // Assuming 128-bit (16-byte) decimal representation
    }

    #[inline]
    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DecimalEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let validity = if self.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            self.validity().clone()
        };

        Ok(Self::new(
            self.byte_buffer().clone(),
            self.decimal_dtype(),
            validity,
        ))
    }
}

impl ArrayStatisticsImpl for DecimalArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayVariantsImpl for DecimalArray {
    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        Some(self)
    }
}

impl DecimalArrayTrait for DecimalArray {}

impl ArrayCanonicalImpl for DecimalArray {
    #[inline]
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        Ok(Canonical::Decimal(self.clone()))
    }

    #[inline]
    fn _append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(self)
    }
}

impl ArrayValidityImpl for DecimalArray {
    #[inline]
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    #[inline]
    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    #[inline]
    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    #[inline]
    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_mask(self.len())
    }
}

#[cfg(test)]
mod test {
    use arrow_array::Decimal128Array;

    #[test]
    fn test_decimal() {
        // They pass it b/c the DType carries the information. No other way to carry a
        // dtype except via the array.
        let value = Decimal128Array::new_null(100);
        let numeric = value.value(10);
        assert_eq!(numeric, 0i128);
    }
}
