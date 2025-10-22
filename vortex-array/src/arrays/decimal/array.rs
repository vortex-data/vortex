// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::{BitBufferMut, Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{DType, DecimalDType, IntegerPType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_panic};
use vortex_scalar::{BigCast, DecimalValueType, NativeDecimalType, match_each_decimal_value_type};

use crate::ToCanonical;
use crate::arrays::is_compatible_decimal_value_type;
use crate::patches::Patches;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

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
    pub(super) dtype: DType,
    pub(super) values: ByteBuffer,
    pub(super) values_type: DecimalValueType,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

impl DecimalArray {
    /// Creates a new [`DecimalArray`].
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented in
    /// [`DecimalArray::new_unchecked`].
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new(buffer, decimal_dtype, validity)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Constructs a new `DecimalArray`.
    ///
    /// See [`DecimalArray::new_unchecked`] for more information.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided components do not satisfy the invariants documented in
    /// [`DecimalArray::new_unchecked`].
    pub fn try_new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&buffer, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked(buffer, decimal_dtype, validity) })
    }

    /// Creates a new [`DecimalArray`] without validation from these components:
    ///
    /// * `buffer` is a typed buffer containing the decimal values.
    /// * `decimal_dtype` specifies the decimal precision and scale.
    /// * `validity` holds the null values.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - All non-null values in `buffer` must be representable within the specified precision.
    /// - For example, with precision=5 and scale=2, all values must be in range [-999.99, 999.99].
    /// - If `validity` is [`Validity::Array`], its length must exactly equal `buffer.len()`.
    pub unsafe fn new_unchecked<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        #[cfg(debug_assertions)]
        Self::validate(&buffer, &validity)
            .vortex_expect("[Debug Assertion]: Invalid `DecimalArray` parameters");

        Self {
            values: buffer.into_byte_buffer(),
            values_type: T::VALUES_TYPE,
            dtype: DType::Decimal(decimal_dtype, validity.nullability()),
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`DecimalArray`].
    ///
    /// This function checks all the invariants required by [`DecimalArray::new_unchecked`].
    pub fn validate<T: NativeDecimalType>(
        buffer: &Buffer<T>,
        validity: &Validity,
    ) -> VortexResult<()> {
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
        if let DType::Decimal(decimal_dtype, _) = self.dtype {
            decimal_dtype
        } else {
            vortex_panic!("Expected Decimal dtype, got {:?}", self.dtype)
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
        let mut validity = BitBufferMut::with_capacity(values.capacity());

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
            Validity::from(validity.freeze()),
        )
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn patch(self, patches: &Patches) -> Self {
        let offset = patches.offset();
        let patch_indices = patches.indices().to_primitive();
        let patch_values = patches.values().to_decimal();

        let patched_validity = self.validity().clone().patch(
            self.len(),
            offset,
            patch_indices.as_ref(),
            patch_values.validity(),
        );
        assert_eq!(self.decimal_dtype(), patch_values.decimal_dtype());

        match_each_integer_ptype!(patch_indices.ptype(), |I| {
            let patch_indices = patch_indices.as_slice::<I>();
            match_each_decimal_value_type!(patch_values.values_type(), |PatchDVT| {
                let patch_values = patch_values.buffer::<PatchDVT>();
                match_each_decimal_value_type!(self.values_type(), |ValuesDVT| {
                    let buffer = self.buffer::<ValuesDVT>().into_mut();
                    patch_typed(
                        buffer,
                        self.decimal_dtype(),
                        patch_indices,
                        offset,
                        patch_values,
                        patched_validity,
                    )
                })
            })
        })
    }
}

fn patch_typed<I, ValuesDVT, PatchDVT>(
    mut buffer: BufferMut<ValuesDVT>,
    decimal_dtype: DecimalDType,
    patch_indices: &[I],
    patch_indices_offset: usize,
    patch_values: Buffer<PatchDVT>,
    patched_validity: Validity,
) -> DecimalArray
where
    I: IntegerPType,
    PatchDVT: NativeDecimalType,
    ValuesDVT: NativeDecimalType,
{
    if !is_compatible_decimal_value_type(ValuesDVT::VALUES_TYPE, decimal_dtype) {
        vortex_panic!(
            "patch_typed: {:?} cannot represent every value in {}.",
            ValuesDVT::VALUES_TYPE,
            decimal_dtype
        )
    }

    for (idx, value) in patch_indices.iter().zip_eq(patch_values.into_iter()) {
        buffer[idx.as_() - patch_indices_offset] = <ValuesDVT as BigCast>::from(value).vortex_expect(
            "values of a given DecimalDType are representable in all compatible NativeDecimalType",
        );
    }

    DecimalArray::new(buffer.freeze(), decimal_dtype, patched_validity)
}
