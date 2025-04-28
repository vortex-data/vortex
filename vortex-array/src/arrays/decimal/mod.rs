mod compute;
mod serde;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::i256;

use crate::array::{Array, ArrayCanonicalImpl, ArrayValidityImpl, ArrayVariantsImpl};
use crate::arrays::decimal::serde::{DecimalMetadata, DecimalValueType};
use crate::builders::ArrayBuilder;
use crate::compute::{ScalarAtFn, SliceFn};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::variants::DecimalArrayTrait;
use crate::vtable::{ComputeVTable, VTableRef};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayVisitorImpl, Canonical, Encoding, ProstMetadata, try_from_array_ref,
};

#[derive(Debug)]
pub struct DecimalEncoding;

impl ComputeVTable for DecimalEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    // TODO(aduffy): SumFn
    // TODO(aduffy): BetweenFn
    // TODO(aduffy): IsSortedFn
    // TODO(aduffy): SearchSortedFn
    // TODO(aduffy): CompareFn
    // TODO(aduffy): IsConstant
    // TODO(aduffy): BetweenFn
    // TODO(aduffy): BinaryNumericFn
    // TODO(aduffy): TakeFn
}

impl Encoding for DecimalEncoding {
    type Array = DecimalArray;
    type Metadata = ProstMetadata<DecimalMetadata>;
}

/// Type of decimal scalar values.
pub trait NativeDecimalType {
    const VALUES_TYPE: DecimalValueType;
}

impl NativeDecimalType for i128 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I128;
}

impl NativeDecimalType for i256 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I256;
}

/// Array for decimal-typed real numbers
#[derive(Clone, Debug)]
pub struct DecimalArray {
    dtype: DType,
    values: ByteBuffer,
    values_type: DecimalValueType,
    validity: Validity,
    stats_set: ArrayStats,
}

try_from_array_ref!(DecimalArray);

impl DecimalArray {
    /// Creates a new [`DecimalArray`] from a [`Buffer`] and [`Validity`], without checking
    /// any invariants.
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        if let Some(len) = validity.maybe_len() {
            if buffer.len() != len {
                vortex_panic!(
                    "Buffer and validity length mismatch: buffer={}, validity={}",
                    buffer.len(),
                    len
                );
            }
        }

        Self {
            dtype: DType::Decimal(decimal_dtype, validity.nullability()),
            values: buffer.into_byte_buffer(),
            values_type: T::VALUES_TYPE,
            validity,
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns the underlying [`ByteBuffer`] of the array.
    pub fn byte_buffer(&self) -> ByteBuffer {
        self.values.clone()
    }

    pub fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        if self.values_type != T::VALUES_TYPE {
            vortex_panic!(
                "Cannot extract Buffer<{:?}> for DecimalArray with values_type {:?}",
                T::VALUES_TYPE,
                self.values_type
            );
        }
        Buffer::<T>::from_byte_buffer(self.values.clone())
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

impl ArrayVisitorImpl<ProstMetadata<DecimalMetadata>> for DecimalArray {
    fn _metadata(&self) -> ProstMetadata<DecimalMetadata> {
        ProstMetadata(DecimalMetadata {
            values_type: self.values_type as i32,
        })
    }

    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&self.values);
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len())
    }
}

impl ArrayImpl for DecimalArray {
    type Encoding = DecimalEncoding;

    #[inline]
    fn _len(&self) -> usize {
        let divisor = match self.values_type {
            DecimalValueType::I128 => 16,
            DecimalValueType::I256 => 32,
        };
        self.values.len() / divisor
    }

    #[inline]
    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DecimalEncoding)
    }

    fn _with_children(&self, _children: &[ArrayRef]) -> VortexResult<Self> {
        // No non-validity child arrays to replace.
        Ok(self.clone())
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
