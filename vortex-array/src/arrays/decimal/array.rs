// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::patches::Patches;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::vtable::validity_to_child;

pub(super) const VALIDITY_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

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
/// - `DecimalType::I8`, `I16`, `I32`, `I64`, `I128`, `I256`
/// - Type selection is automatic based on the required precision
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::DecimalArray;
/// use vortex_array::dtype::DecimalDType;
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
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) dtype: DType,
    pub(super) values: BufferHandle,
    pub(super) values_type: DecimalType,
    pub(super) validity: Validity,
    pub(super) stats_set: ArrayStats,
}

pub struct DecimalArrayParts {
    pub decimal_dtype: DecimalDType,
    pub values: BufferHandle,
    pub values_type: DecimalType,
    pub validity: Validity,
}

impl DecimalArray {
    fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        vec![validity_to_child(validity, len)]
    }

    /// Creates a new [`DecimalArray`] using a host-native buffer.
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

    /// Creates a new [`DecimalArray`] from a [`BufferHandle`] of values that may live in
    /// host or device memory.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented in
    /// [`DecimalArray::new_unchecked`].
    pub fn new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new_handle(values, values_type, decimal_dtype, validity)
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
        let values = BufferHandle::new_host(buffer.into_byte_buffer());
        let values_type = T::DECIMAL_TYPE;

        Self::try_new_handle(values, values_type, decimal_dtype, validity)
    }

    /// Constructs a new `DecimalArray` with validation from a [`BufferHandle`].
    ///
    /// This pathway allows building new decimal arrays that may come from host or device memory.
    ///
    /// # Errors
    ///
    /// See [`DecimalArray::new_unchecked`] for invariants that are checked.
    pub fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::validate(&values, values_type, &validity)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked_handle(values, values_type, decimal_dtype, validity) })
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
        // SAFETY: new_unchecked_handle inherits the safety guarantees of new_unchecked
        unsafe {
            Self::new_unchecked_handle(
                BufferHandle::new_host(buffer.into_byte_buffer()),
                T::DECIMAL_TYPE,
                decimal_dtype,
                validity,
            )
        }
    }

    /// Create a new array with decimal values backed by the given buffer handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure all of the following invariants are satisfied:
    ///
    /// - All non-null values in `values` must be representable within the specified precision.
    /// - For example, with precision=5 and scale=2, all values must be in range [-999.99, 999.99].
    /// - If `validity` is [`Validity::Array`], its length must exactly equal `buffer.len()`.
    pub unsafe fn new_unchecked_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            Self::validate(&values, values_type, &validity)
                .vortex_expect("[Debug Assertion]: Invalid `DecimalArray` parameters");
        }

        let len = values.len() / values_type.byte_width();
        Self {
            slots: Self::make_slots(&validity, len),
            values,
            values_type,
            dtype: DType::Decimal(decimal_dtype, validity.nullability()),
            validity,
            stats_set: Default::default(),
        }
    }

    /// Validates the components that would be used to create a [`DecimalArray`] from a byte buffer.
    ///
    /// This function checks all the invariants required by [`DecimalArray::new_unchecked`].
    fn validate(
        buffer: &BufferHandle,
        values_type: DecimalType,
        validity: &Validity,
    ) -> VortexResult<()> {
        if let Some(validity_len) = validity.maybe_len() {
            let expected_len = values_type.byte_width() * validity_len;
            vortex_ensure!(
                buffer.len() == expected_len,
                InvalidArgument: "expected buffer of size {} bytes, was {} bytes",
                expected_len,
                buffer.len(),
            );
        }

        Ok(())
    }

    /// Creates a new [`DecimalArray`] from a raw byte buffer without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The `byte_buffer` contains valid data for the specified `values_type`
    /// - The buffer length is compatible with the `values_type` (i.e., divisible by the type size)
    /// - All non-null values are representable within the specified precision
    /// - If `validity` is [`Validity::Array`], its length must equal the number of elements
    pub unsafe fn new_unchecked_from_byte_buffer(
        byte_buffer: ByteBuffer,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        // SAFETY: inherits the same safety contract as `new_unchecked_from_byte_buffer`
        unsafe {
            Self::new_unchecked_handle(
                BufferHandle::new_host(byte_buffer),
                values_type,
                decimal_dtype,
                validity,
            )
        }
    }

    pub fn into_parts(self) -> DecimalArrayParts {
        let decimal_dtype = self.dtype.into_decimal_opt().vortex_expect("cannot fail");

        DecimalArrayParts {
            decimal_dtype,
            values: self.values,
            values_type: self.values_type,
            validity: self.validity,
        }
    }

    /// Returns the underlying [`ByteBuffer`] of the array.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }

    pub fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        if self.values_type != T::DECIMAL_TYPE {
            vortex_panic!(
                "Cannot extract Buffer<{:?}> for DecimalArray with values_type {:?}",
                T::DECIMAL_TYPE,
                self.values_type,
            );
        }
        Buffer::<T>::from_byte_buffer(self.values.as_host().clone())
    }

    /// Returns the decimal type information
    pub fn decimal_dtype(&self) -> DecimalDType {
        if let DType::Decimal(decimal_dtype, _) = self.dtype {
            decimal_dtype
        } else {
            vortex_panic!("Expected Decimal dtype, got {:?}", self.dtype)
        }
    }

    /// Return the `DecimalType` used to represent the values in the array.
    pub fn values_type(&self) -> DecimalType {
        self.values_type
    }

    pub fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    pub fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }

    pub fn from_iter<T: NativeDecimalType, I: IntoIterator<Item = T>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let iter = iter.into_iter();

        Self::new(
            BufferMut::from_iter(iter).freeze(),
            decimal_dtype,
            Validity::NonNullable,
        )
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

    #[expect(
        clippy::cognitive_complexity,
        reason = "complexity from nested match_each_* macros"
    )]
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let offset = patches.offset();
        let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<DecimalArray>(ctx)?;

        let patched_validity = self.validity().clone().patch(
            self.len(),
            offset,
            &patch_indices.clone().into_array(),
            patch_values.validity(),
            ctx,
        )?;
        assert_eq!(self.decimal_dtype(), patch_values.decimal_dtype());

        Ok(match_each_integer_ptype!(patch_indices.ptype(), |I| {
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
        }))
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
    if !ValuesDVT::DECIMAL_TYPE.is_compatible_decimal_value_type(decimal_dtype) {
        vortex_panic!(
            "patch_typed: {:?} cannot represent every value in {}.",
            ValuesDVT::DECIMAL_TYPE,
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
