// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod ops;
mod patch;
mod serde;

use arrow_buffer::BooleanBufferBuilder;
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_scalar::{DecimalValueType, NativeDecimalType};

use crate::builders::ArrayBuilder;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper, VisitorVTable,
};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, Canonical, EncodingId, EncodingRef, vtable};

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

/// Maps a decimal precision into the smallest type that can represent it.
pub fn smallest_storage_type(decimal_dtype: &DecimalDType) -> DecimalValueType {
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

/// True if `value_type` can represent every value of the type `dtype`.
pub fn compatible_storage_type(value_type: DecimalValueType, dtype: DecimalDType) -> bool {
    value_type >= smallest_storage_type(&dtype)
}

/// A decimal array that stores fixed-precision decimal numbers with configurable scale.
///
/// This mirrors the Apache Arrow Decimal encoding and provides exact arithmetic for
/// financial and scientific computations where floating-point precision loss is unacceptable.
///
/// ## Storage Format
///
/// Decimals are stored as scaled integers in a supported scalar value type.
///
/// The precisions supported for each scalar type are:
/// - **i8**: precision 1-2 digits
/// - **i16**: precision 3-4 digits  
/// - **i32**: precision 5-9 digits
/// - **i64**: precision 10-18 digits
/// - **i128**: precision 19-38 digits
/// - **i256**: precision 39-76 digits
///
/// These are just the maximal ranges for each scalar type, but it is perfectly legal to store
/// values with precision that does not match this exactly. For example, a valid DecimalArray with
/// precision=39 may store its values in an `i8` if all of the actual values fit into it.
///
/// Similarly, a `DecimalArray` can be built that stores a set of precision=2 values in a
/// `Buffer<i256>`.
///
/// ## Precision and Scale
///
/// - **Precision**: Total number of significant digits (1-76, u8 range)
/// - **Scale**: Number of digits after the decimal point (-128 to 127, i8 range)
/// - **Value**: `stored_integer / 10^scale`
///
/// For example, with precision=5 and scale=2:
/// - Stored value 12345 represents 123.45
/// - Range: -999.99 to 999.99
///
/// ## Valid Scalar Types
///
/// The underlying storage uses these native types based on precision:
/// - `DecimalValueType::I8`, `I16`, `I32`, `I64`, `I128`, `I256`
/// - Type selection is automatic based on the required precision
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::DecimalArray;
/// use vortex_dtype::DecimalDType;
/// use vortex_buffer::{buffer, Buffer};
/// use vortex_array::validity::Validity;
///
/// // Create a decimal array with precision=5, scale=2 (e.g., 123.45)
/// let decimal_dtype = DecimalDType::new(5, 2);
/// let values = buffer![12345i32, 67890i32, -12300i32]; // 123.45, 678.90, -123.00
/// let array = DecimalArray::new(values, decimal_dtype, Validity::NonNullable);
///
/// assert_eq!(array.precision(), 5);
/// assert_eq!(array.scale(), 2);
/// assert_eq!(array.len(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct DecimalArray {
    dtype: DType,
    values: ByteBuffer,
    values_type: DecimalValueType,
    validity: Validity,
    stats_set: ArrayStats,
}

impl DecimalArray {
    fn validate<T: NativeDecimalType>(buffer: &Buffer<T>, validity: &Validity) -> VortexResult<()> {
        if let Some(len) = validity.maybe_len() {
            vortex_ensure!(
                buffer.len() == len,
                "Buffer and validity length mismatch: buffer={}, validity={}",
                buffer.len(),
                len,
            );
        }

        Ok(())
    }
}

impl DecimalArray {
    /// Creates a new [`DecimalArray`] from a [`Buffer`] and [`Validity`], without checking
    /// any invariants.
    ///
    /// # Panics
    ///
    /// Panics if the provided buffer and validity differ in length.
    ///
    /// See also [`DecimalArray::try_new`].
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new(buffer, decimal_dtype, validity).vortex_expect("DecimalArray new")
    }

    /// Build a new `DecimalArray` from a component `buffer`, decimal_dtype` and `validity`.
    ///
    /// This constructor validates the length of the buffer and validity are equal, returning
    /// an error otherwise.
    ///
    /// See [`DecimalArray::new`] for an infallible constructor that panics on validation errors.
    pub fn try_new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&buffer, &validity)?;

        Ok(Self {
            values: buffer.into_byte_buffer(),
            values_type: T::VALUES_TYPE,
            dtype: DType::Decimal(decimal_dtype, validity.nullability()),
            validity,
            stats_set: Default::default(),
        })
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

    pub fn from_option_iter<T: NativeDecimalType, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut values = BufferMut::with_capacity(iter.size_hint().0);
        let mut validity = BooleanBufferBuilder::new(values.capacity());

        for i in iter {
            match i {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(e) => {
                    validity.append(true);
                    values.push(e);
                }
            }
        }
        Self::new(
            values.freeze(),
            decimal_dtype,
            Validity::from(validity.finish()),
        )
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
