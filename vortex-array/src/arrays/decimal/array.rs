// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use smallvec::smallvec;
use vortex_buffer::Alignment;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::child_to_validity;
use crate::array::validity_to_child;
use crate::array_slots;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::cast_value_in_precision;
use crate::arrays::decimal::check_precision;
use crate::arrays::decimal::precision_bounds;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value_type;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[array_slots(Decimal)]
pub struct DecimalSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
}

/// The decimal dtype and values buffer of a [`DecimalArray`].
///
/// A valid row stores a signed integer whose logical value is `integer / 10^scale`. The
/// precision determines the integer type through [`DecimalDType::canonical_values_type`].
///
/// | Precision | Integer type |
/// |-----------|--------------|
/// | 1 to 2    | `i8`         |
/// | 3 to 4    | `i16`        |
/// | 5 to 9    | `i32`        |
/// | 10 to 18  | `i64`        |
/// | 19 to 38  | `i128`       |
/// | 39 to 76  | `i256`       |
///
/// The `vortex.decimal` wire format records the stored width separately. The reader uses
/// [`DecimalDataParts`] to convert a different width to a canonical leaf or to a
/// [`Widen`](crate::arrays::Widen) over a lower-precision leaf.
///
/// # Examples
///
/// ```
/// use vortex_array::VortexSessionExecute;
/// use vortex_array::arrays::DecimalArray;
/// use vortex_array::dtype::DecimalDType;
/// use vortex_array::validity::Validity;
/// use vortex_buffer::buffer;
/// use vortex_session::VortexSession;
///
/// let decimal_dtype = DecimalDType::new(5, 2);
/// let values = buffer![12345i32, 67890i32, -12300i32];
/// let mut ctx = VortexSession::empty().create_execution_ctx();
/// let array = DecimalArray::try_new(values, decimal_dtype, Validity::NonNullable, &mut ctx)?;
///
/// assert_eq!(array.precision(), 5);
/// assert_eq!(array.scale(), 2);
/// assert_eq!(array.len(), 3);
/// # Ok::<(), vortex_error::VortexError>(())
/// ```
#[derive(Clone, Debug)]
pub struct DecimalData {
    pub(super) decimal_dtype: DecimalDType,
    pub(super) values: BufferHandle,
}

impl Display for DecimalData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "decimal_dtype: {}, values_type: {}",
            self.decimal_dtype,
            self.values_type()
        )
    }
}

/// Decimal storage and validity, as returned by [`DecimalArray::into_data_parts`].
pub struct DecimalDataParts {
    /// The precision and scale of the values.
    pub decimal_dtype: DecimalDType,
    /// Unscaled integers of type `values_type` in host or device memory.
    pub values: BufferHandle,
    /// Actual storage type, retained to interpret input before canonical conversion.
    ///
    /// Legacy `vortex.decimal` nodes can use a different width from
    /// [`DecimalDType::canonical_values_type`]. Comparing the widths identifies noncanonical
    /// storage, not the age of the data. Current [`WidenPlugin`] writes store noncanonical widths
    /// too.
    ///
    /// [`WidenPlugin`]: crate::arrays::widen::WidenPlugin
    pub values_type: DecimalType,
    /// The validity of the values, which also gives the nullability of the array.
    pub validity: Validity,
}

pub trait DecimalArrayExt: TypedArrayRef<Decimal> {
    fn decimal_dtype(&self) -> DecimalDType {
        match self.dtype() {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => unreachable!("DecimalArrayExt requires a decimal dtype"),
        }
    }

    fn nullability(&self) -> Nullability {
        match self.dtype() {
            DType::Decimal(_, nullability) => *nullability,
            _ => unreachable!("DecimalArrayExt requires a decimal dtype"),
        }
    }

    fn validity_child(&self) -> Option<&ArrayRef> {
        self.slots()[DecimalSlots::VALIDITY_CHILD].as_ref()
    }

    fn validity(&self) -> Validity {
        child_to_validity(
            self.slots()[DecimalSlots::VALIDITY_CHILD].as_ref(),
            self.nullability(),
        )
    }

    fn values_type(&self) -> DecimalType {
        DecimalData::values_type(self)
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
    fn make_slots(validity: &Validity, len: usize) -> ArraySlots {
        smallvec![validity_to_child(validity, len)]
    }

    pub(super) fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<Self> {
        Self::validate_width(values_type, decimal_dtype)?;
        Self::validate(&values, values_type)?;

        Ok(Self {
            decimal_dtype,
            values,
        })
    }

    pub(super) fn validate_width(
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<()> {
        let canonical = decimal_dtype.canonical_values_type();
        vortex_ensure_eq!(
            values_type,
            canonical,
            "{decimal_dtype} requires {canonical} values, got {values_type}",
        );

        Ok(())
    }

    /// Validates the layout of a decimal values buffer.
    ///
    /// Checks byte length and alignment for `values_type`. Does not read values or check validity.
    pub(crate) fn validate(buffer: &BufferHandle, values_type: DecimalType) -> VortexResult<()> {
        let byte_width = values_type.byte_width();
        vortex_ensure!(
            buffer.len().is_multiple_of(byte_width),
            "decimal buffer size {} is not divisible by element width {}",
            buffer.len(),
            byte_width,
        );
        match_each_decimal_value_type!(values_type, |D| {
            vortex_ensure!(
                buffer.is_aligned_to(Alignment::of::<D>()),
                "decimal buffer alignment {:?} is invalid for values type {:?}",
                buffer.alignment(),
                D::DECIMAL_TYPE,
            );
            Ok::<(), vortex_error::VortexError>(())
        })?;
        Ok(())
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.values.len() / self.values_type().byte_width()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the host or device buffer that stores the unscaled integers.
    pub fn buffer_handle(&self) -> &BufferHandle {
        &self.values
    }

    pub fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        if self.values_type() != T::DECIMAL_TYPE {
            vortex_panic!(
                "Cannot extract Buffer<{:?}> for DecimalArray with values_type {:?}",
                T::DECIMAL_TYPE,
                self.values_type(),
            );
        }
        Buffer::<T>::from_byte_buffer(self.values.as_host().clone())
    }

    /// Returns the integer type of the values, which is the canonical type for the precision.
    pub fn values_type(&self) -> DecimalType {
        self.decimal_dtype.canonical_values_type()
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
            values_type: decimal_dtype.canonical_values_type(),
            validity,
        }
    }
}

// Keep array construction in the non-generic handle constructors so downstream crates do not
// compile the decimal vtable for each typed constructor.
impl Array<Decimal> {
    /// Creates a decimal array from a buffer of the canonical type for `decimal_dtype`.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::try_new`] rejects the buffer or validity.
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::try_new(buffer, decimal_dtype, validity, ctx)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a decimal array without checking its values or validity.
    ///
    /// A valid row outside the declared precision can yield incorrect results in later operations,
    /// including comparisons that use the declared range.
    ///
    /// # Panics
    ///
    /// Panics if `T` differs from the canonical type for `decimal_dtype`.
    ///
    /// # Safety
    ///
    /// An array-backed validity must contain one non-nullable boolean per value.
    pub unsafe fn new_unchecked<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        let values = BufferHandle::new_host(buffer.into_byte_buffer());

        // SAFETY: The typed buffer supplies the required length and alignment. The caller
        // guarantees the validity matches it.
        unsafe { Self::new_unchecked_handle(values, T::DECIMAL_TYPE, decimal_dtype, validity) }
    }

    /// Creates a decimal array, reusing the typed buffer. When every validity buffer is on the
    /// host, checks that each valid row fits the precision. Null slots may hold any value.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Self::try_new_handle`] rejects the width, validity or values.
    pub fn try_new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let values = BufferHandle::new_host(buffer.into_byte_buffer());
        Self::try_new_handle(values, T::DECIMAL_TYPE, decimal_dtype, validity, ctx)
    }

    /// Collects values into a buffer of the canonical type for `decimal_dtype`.
    ///
    /// # Panics
    ///
    /// Panics if any value falls outside the declared precision.
    #[expect(
        clippy::same_name_method,
        reason = "intentionally named from_iter like Iterator::from_iter"
    )]
    pub fn from_iter<T: NativeDecimalType, I: IntoIterator<Item = T>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let values_type = decimal_dtype.canonical_values_type();
        let values = collect_decimal_values(iter.into_iter(), decimal_dtype);

        // SAFETY: Collection produced an aligned buffer at the canonical width, checking each value.
        unsafe {
            Self::new_unchecked_handle(values, values_type, decimal_dtype, Validity::NonNullable)
        }
    }

    /// Collects optional values into a nullable array of the canonical type for `decimal_dtype`.
    /// Null slots contain zero.
    ///
    /// # Panics
    ///
    /// Panics if any present value falls outside the declared precision.
    pub fn from_option_iter<T: NativeDecimalType, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut validity = BitBufferMut::with_capacity(iter.size_hint().0);
        let values_type = decimal_dtype.canonical_values_type();
        let values = collect_decimal_values(
            iter.map(|value| {
                validity.append(value.is_some());
                value.unwrap_or_default()
            }),
            decimal_dtype,
        );
        let validity = Validity::from(validity.freeze());

        // SAFETY: Collection produced one validity bit for each value in the typed, canonical buffer.
        unsafe { Self::new_unchecked_handle(values, values_type, decimal_dtype, validity) }
    }

    /// Creates a decimal array from a host or device buffer.
    ///
    /// # Panics
    ///
    /// Panics on any error from [`Self::try_new_handle`].
    pub fn new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::try_new_handle(values, values_type, decimal_dtype, validity, ctx)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a decimal array from a host or device buffer.
    ///
    /// If the values and every buffer of the validity are host-resident, checks each valid
    /// value against the declared precision. Null slots may hold any value. If either input
    /// contains a device buffer, checks only the layout and leaves the buffers in place.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout is invalid, host validity execution fails, or a valid
    /// host row exceeds the declared precision. The precision error names the first such row.
    pub fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let array =
            Self::try_new_handle_from_trusted_values(values, values_type, decimal_dtype, validity)?;
        let validity = array.validity()?;
        if array.buffer_handle().is_on_host() && validity.as_array().is_none_or(ArrayRef::is_host) {
            let mask = validity.execute_mask(array.len(), ctx)?;
            match_each_decimal_value_type!(values_type, |T| {
                check_precision(array.buffer::<T>().as_slice(), &mask, decimal_dtype)
            })?;
        }
        Ok(array)
    }

    /// Reconstructs stored decimal values after checking their layout.
    ///
    /// Use this when reading a file that declares its precision, or when a kernel preserves
    /// or has already checked the valid values. The values buffer and validity remain in place.
    /// If the declared precision is wrong, later operations may return incorrect results or a
    /// precision error. Use [`Self::try_new_handle`] to check host values.
    ///
    /// # Errors
    ///
    /// Returns an error if the width differs from the canonical width of `decimal_dtype`, the
    /// buffer has invalid alignment or byte length, or the validity has the wrong length or dtype.
    pub fn try_new_handle_from_trusted_values(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        DecimalData::validate_width(values_type, decimal_dtype)?;
        let parts = Self::decimal_parts(values, decimal_dtype, validity);

        Array::try_from_parts(parts)
    }

    /// Creates a decimal array without checking its buffer layout, validity or values.
    ///
    /// Valid rows must fit the declared precision for correct results. Trusted file values
    /// can exceed that precision without affecting memory safety.
    ///
    /// # Panics
    ///
    /// Panics if `values_type` differs from the canonical type for `decimal_dtype`.
    ///
    /// # Safety
    ///
    /// The buffer must be aligned for `values_type` and contain a whole number of values. An
    /// array-backed validity must contain one non-nullable boolean per value. A malformed layout
    /// can panic when the array is read.
    pub unsafe fn new_unchecked_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        DecimalData::validate_width(values_type, decimal_dtype)
            .unwrap_or_else(|error| vortex_panic!(error));
        let parts = Self::decimal_parts(values, decimal_dtype, validity);

        // SAFETY: The width matches the dtype, and the caller guarantees the layout and validity.
        unsafe { Array::from_parts_unchecked(parts) }
    }

    /// Assembles the parts of a leaf whose buffer holds values of the canonical type for
    /// `decimal_dtype`. [`Array::try_from_parts`] checks the buffer layout and validity before
    /// exposing the array. A caller that skips validation must establish them.
    pub(super) fn decimal_parts(
        values: BufferHandle,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> ArrayParts<Decimal> {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        let len = values.len() / decimal_dtype.canonical_values_type().byte_width();
        let slots = DecimalData::make_slots(&validity, len);
        let data = DecimalData {
            decimal_dtype,
            values,
        };

        ArrayParts::new(Decimal, dtype, len, data, slots)
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
        // Patch indices are non-negative; reinterpret to unsigned so this dispatches over 4 widths
        // instead of 8 (the decimal value-type dimensions are unaffected).
        let patch_indices_unsigned =
            patch_indices.reinterpret_cast(patch_indices.ptype().to_unsigned());
        let data = match_each_unsigned_integer_ptype!(patch_indices_unsigned.ptype(), |I| {
            let patch_indices = patch_indices_unsigned.as_slice::<I>();
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
            Array::from_parts_unchecked(ArrayParts::new(Decimal, dtype, len, data, slots))
        })
    }
}

fn collect_decimal_values<T: NativeDecimalType>(
    iter: impl Iterator<Item = T>,
    decimal_dtype: DecimalDType,
) -> BufferHandle {
    match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |D| {
        let bounds = precision_bounds::<D>(decimal_dtype);
        let values = Buffer::<D>::from_iter(
            iter.map(|value| cast_value_in_precision(value, decimal_dtype, &bounds)),
        );

        BufferHandle::new_host(values.into_byte_buffer())
    })
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

    for (idx, value) in patch_indices.iter().zip_eq(patch_values) {
        buffer[idx.as_() - patch_indices_offset] = <ValuesDVT as BigCast>::from(value).vortex_expect(
            "values of a given DecimalDType are representable in all compatible NativeDecimalType",
        );
    }

    let values = BufferHandle::new_host(buffer.freeze().into_byte_buffer());
    DecimalData::try_new_handle(values, ValuesDVT::DECIMAL_TYPE, decimal_dtype)
        .vortex_expect("patching preserves the canonical width and layout of the input buffer")
}
