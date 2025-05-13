mod compute;
mod macros;
mod ops;
mod serde;

use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_panic};
use vortex_scalar::{DecimalValueType, NativeDecimalType};

use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper, VisitorVTable,
};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingId, EncodingRef, vtable,
};

vtable!(Decimal);

impl VTable for DecimalVTable {
    type Array = DecimalArray;
    type Encoding = DecimalEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.decimal")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DecimalEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct DecimalEncoding;

/// Maps a decimal precision into the small type that can represent it.
pub fn precision_to_storage_size(decimal_dtype: &DecimalDType) -> DecimalValueType {
    match decimal_dtype.precision() {
        1..=2 => DecimalValueType::I8,
        3..=4 => DecimalValueType::I16,
        5..=9 => DecimalValueType::I32,
        10..=18 => DecimalValueType::I64,
        19..=38 => DecimalValueType::I128,
        39..=76 => DecimalValueType::I256,
        0 => unreachable!("precision must be greater than 0"),
        p => unreachable!("precision larger than 76 is invalid found precision {p}"),
    }
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
                    len,
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
                self.values_type,
            );
        }
        Buffer::<T>::from_byte_buffer(self.values.clone())
    }

    /// Returns the decimal type information
    pub fn decimal_dtype(&self) -> DecimalDType {
        match &self.dtype {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => vortex_panic!("Expected Decimal dtype, got {:?}", self.dtype),
        }
    }

    pub fn values_type(&self) -> DecimalValueType {
        self.values_type
    }

    pub fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    pub fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }
}

impl ArrayVTable<DecimalVTable> for DecimalVTable {
    fn len(array: &DecimalArray) -> usize {
        let divisor = match array.values_type {
            DecimalValueType::I8 => 1,
            DecimalValueType::I16 => 2,
            DecimalValueType::I32 => 4,
            DecimalValueType::I64 => 8,
            DecimalValueType::I128 => 16,
            DecimalValueType::I256 => 32,
            ty => vortex_panic!("unknown decimal value type {:?}", ty),
        };
        array.values.len() / divisor
    }

    fn dtype(array: &DecimalArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DecimalArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.values);
    }

    fn visit_children(array: &DecimalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }

    fn with_children(array: &DecimalArray, _children: &[ArrayRef]) -> VortexResult<DecimalArray> {
        // FIXME(ngates): ported this logic from old code, but it needs to handle replacing
        //  any validity child.
        // No non-validity child arrays to replace.
        Ok(array.clone())
    }
}

impl CanonicalVTable<DecimalVTable> for DecimalVTable {
    fn canonicalize(array: &DecimalArray) -> VortexResult<Canonical> {
        Ok(Canonical::Decimal(array.clone()))
    }

    fn append_to_builder(array: &DecimalArray, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref())
    }
}

impl ValidityHelper for DecimalArray {
    fn validity(&self) -> &Validity {
        &self.validity
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
