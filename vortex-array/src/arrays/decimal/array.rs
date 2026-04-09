// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_buffer::Alignment;
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
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

/// The validity bitmap indicating which elements are non-null.
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
pub struct DecimalData {
    pub(super) decimal_dtype: DecimalDType,
    pub(super) values: BufferHandle,
    pub(super) values_type: DecimalType,
}

impl Display for DecimalData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decimal_dtype: {}, values_type: {}",
            self.decimal_dtype, self.values_type
        )
    }
}

pub struct DecimalDataParts {
    pub decimal_dtype: DecimalDType,
    pub values: BufferHandle,
    pub values_type: DecimalType,
    pub validity: Validity,
}

pub trait DecimalArrayExt: TypedArrayRef<Decimal> {
    fn decimal_dtype(&self) -> DecimalDType {
        match self.as_ref().dtype() {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => unreachable!("DecimalArrayExt requires a decimal dtype"),
        }
    }

    fn nullability(&self) -> Nullability {
        match self.as_ref().dtype() {
            DType::Decimal(_, nullability) => *nullability,
            _ => unreachable!("DecimalArrayExt requires a decimal dtype"),
        }
    }

    fn validity_child(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[VALIDITY_SLOT].as_ref()
    }

    fn validity(&self) -> Validity {
        child_to_validity(&self.as_ref().slots()[VALIDITY_SLOT], self.nullability())
    }

    fn values_type(&self) -> DecimalType {
        self.values_type
    }

    fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }

    fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }

    fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        DecimalData::buffer::<T>(self)
    }
}
impl<T: TypedArrayRef<Decimal>> DecimalArrayExt for T {}

impl DecimalData {
    /// Build the slots vector for this array.
    pub(super) fn make_slots(validity: &Validity, len: usize) -> Vec<Option<ArrayRef>> {
        vec![validity_to_child(validity, len)]
    }

    /// Creates a new [`DecimalArray`] using a host-native buffer.
    ///
    /// # Panics
    ///
    /// Panics if the provided components do not satisfy the invariants documented in
    /// [`DecimalArray::new_unchecked`].
    pub fn new<T: NativeDecimalType>(buffer: Buffer<T>, decimal_dtype: DecimalDType) -> Self {
        Self::try_new(buffer, decimal_dtype).vortex_expect("DecimalArray construction failed")
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
    ) -> Self {
        Self::try_new_handle(values, values_type, decimal_dtype)
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
    ) -> VortexResult<Self> {
        let values = BufferHandle::new_host(buffer.into_byte_buffer());
        let values_type = T::DECIMAL_TYPE;

        Self::try_new_handle(values, values_type, decimal_dtype)
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
    ) -> VortexResult<Self> {
        Self::validate(&values, values_type)?;

        // SAFETY: validate ensures all invariants are met.
        Ok(unsafe { Self::new_unchecked_handle(values, values_type, decimal_dtype) })
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
    ) -> Self {
        // SAFETY: new_unchecked_handle inherits the safety guarantees of new_unchecked
        unsafe {
            Self::new_unchecked_handle(
                BufferHandle::new_host(buffer.into_byte_buffer()),
                T::DECIMAL_TYPE,
                decimal_dtype,
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
    ) -> Self {
        Self {
            decimal_dtype,
            values,
            values_type,
        }
    }

    /// Validates the components that would be used to create a [`DecimalArray`] from a byte buffer.
    ///
    /// This function checks all the invariants required by [`DecimalArray::new_unchecked`].
    fn validate(buffer: &BufferHandle, values_type: DecimalType) -> VortexResult<()> {
        let byte_width = values_type.byte_width();
        vortex_ensure!(
            buffer.len().is_multiple_of(byte_width),
            InvalidArgument: "decimal buffer size {} is not divisible by element width {}",
            buffer.len(),
            byte_width,
        );
        match_each_decimal_value_type!(values_type, |D| {
            vortex_ensure!(
                buffer.is_aligned_to(Alignment::of::<D>()),
                InvalidArgument: "decimal buffer alignment {:?} is invalid for values type {:?}",
                buffer.alignment(),
                D::DECIMAL_TYPE,
            );
            Ok::<(), vortex_error::VortexError>(())
        })?;
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
    ) -> Self {
        // SAFETY: inherits the same safety contract as `new_unchecked_from_byte_buffer`
        unsafe {
            Self::new_unchecked_handle(
                BufferHandle::new_host(byte_buffer),
                values_type,
                decimal_dtype,
            )
        }
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.values.len() / self.values_type.byte_width()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Return the `DecimalType` used to represent the values in the array.
    pub fn values_type(&self) -> DecimalType {
        self.values_type
    }

    /// Returns the decimal type information.
    pub fn decimal_dtype(&self) -> DecimalDType {
        self.decimal_dtype
    }

    pub fn precision(&self) -> u8 {
        self.decimal_dtype.precision()
    }

    pub fn scale(&self) -> i8 {
        self.decimal_dtype.scale()
    }
}

impl Array<Decimal> {
    pub fn into_data_parts(self) -> DecimalDataParts {
        let validity = DecimalArrayExt::validity(&self);
        let decimal_dtype = DecimalArrayExt::decimal_dtype(&self);
        let data = self.into_data();
        DecimalDataParts {
            decimal_dtype,
            values: data.values,
            values_type: data.values_type,
            validity,
        }
    }
}

impl Array<Decimal> {
    /// Creates a new [`DecimalArray`] using a host-native buffer.
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new(buffer, decimal_dtype, validity)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a new [`DecimalArray`] without validation.
    ///
    /// # Safety
    ///
    /// See [`DecimalData::new_unchecked`].
    pub unsafe fn new_unchecked<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        let len = buffer.len();
        let slots = DecimalData::make_slots(&validity, len);
        let data = unsafe { DecimalData::new_unchecked(buffer, decimal_dtype) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Decimal, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Creates a new [`DecimalArray`] from a host-native buffer with validation.
    pub fn try_new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        let len = buffer.len();
        let slots = DecimalData::make_slots(&validity, len);
        let data = DecimalData::try_new(buffer, decimal_dtype)?;
        Array::try_from_parts(ArrayParts::new(Decimal, dtype, len, data).with_slots(slots))
    }

    /// Creates a new [`DecimalArray`] from an iterator of values.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<T: NativeDecimalType, I: IntoIterator<Item = T>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        Self::new(
            BufferMut::from_iter(iter).freeze(),
            decimal_dtype,
            Validity::NonNullable,
        )
    }

    /// Creates a new [`DecimalArray`] from an iterator of optional values.
    pub fn from_option_iter<T: NativeDecimalType, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut values = BufferMut::with_capacity(iter.size_hint().0);
        let mut validity = BitBufferMut::with_capacity(values.capacity());

        for value in iter {
            match value {
                Some(value) => {
                    values.push(value);
                    validity.append(true);
                }
                None => {
                    values.push(T::default());
                    validity.append(false);
                }
            }
        }

        Self::new(
            values.freeze(),
            decimal_dtype,
            Validity::from(validity.freeze()),
        )
    }

    /// Creates a new [`DecimalArray`] from a [`BufferHandle`].
    pub fn new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new_handle(values, values_type, decimal_dtype, validity)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a new [`DecimalArray`] from a [`BufferHandle`] with validation.
    pub fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        let len = values.len() / values_type.byte_width();
        let slots = DecimalData::make_slots(&validity, len);
        let data = DecimalData::try_new_handle(values, values_type, decimal_dtype)?;
        Array::try_from_parts(ArrayParts::new(Decimal, dtype, len, data).with_slots(slots))
    }

    /// Creates a new [`DecimalArray`] without validation from a [`BufferHandle`].
    ///
    /// # Safety
    ///
    /// See [`DecimalData::new_unchecked_handle`].
    pub unsafe fn new_unchecked_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        let len = values.len() / values_type.byte_width();
        let slots = DecimalData::make_slots(&validity, len);
        let data = unsafe { DecimalData::new_unchecked_handle(values, values_type, decimal_dtype) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Decimal, dtype, len, data).with_slots(slots),
            )
        }
    }

    #[expect(
        clippy::cognitive_complexity,
        reason = "patching depends on both patch and value physical types"
    )]
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let offset = patches.offset();
        let dtype = self.dtype().clone();
        let len = self.len();
        let patch_indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<DecimalArray>(ctx)?;

        let patch_validity = patch_values.validity()?;
        let patched_validity = self.validity()?.patch(
            self.len(),
            offset,
            &patch_indices.clone().into_array(),
            &patch_validity,
            ctx,
        )?;
        assert_eq!(self.decimal_dtype(), patch_values.decimal_dtype());

        let data = self.into_data();
        let data = match_each_integer_ptype!(patch_indices.ptype(), |I| {
            let patch_indices = patch_indices.as_slice::<I>();
            match_each_decimal_value_type!(patch_values.values_type(), |PatchDVT| {
                let patch_values = patch_values.buffer::<PatchDVT>();
                match_each_decimal_value_type!(data.values_type(), |ValuesDVT| {
                    let buffer = data.buffer::<ValuesDVT>().into_mut();
                    patch_typed(
                        buffer,
                        data.decimal_dtype(),
                        patch_indices,
                        offset,
                        patch_values,
                    )
                })
            })
        });
        let slots = DecimalData::make_slots(&patched_validity, len);
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Decimal, dtype, len, data).with_slots(slots),
            )
        })
    }
}

fn patch_typed<I, ValuesDVT, PatchDVT>(
    mut buffer: BufferMut<ValuesDVT>,
    decimal_dtype: DecimalDType,
    patch_indices: &[I],
    patch_indices_offset: usize,
    patch_values: Buffer<PatchDVT>,
) -> DecimalData
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

    DecimalData::new(buffer.freeze(), decimal_dtype)
}
