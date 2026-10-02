// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_buffer::Alignment;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::NarrowArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::IntegerPType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::integer::integer_dtype;
use crate::dtype::integer::signed_integer_type;
use crate::integer;
use crate::integer::IntegerBuffer;
use crate::match_each_decimal_value_type;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::Patches;
use crate::validity::Validity;

#[array_slots(Decimal)]
pub struct DecimalSlots {
    /// Signed integer values, with logical width determined by the decimal precision.
    /// This child also owns the decimal's validity.
    #[slot(0)]
    pub values: ArrayRef,
}

/// Decimal has no buffers or metadata beyond its dtype and integer child.
pub type DecimalData = EmptyArrayData;

/// Materialized decimal storage at the child's stored integer width.
///
/// Call [`DecimalArrayExt::materialize_values`] before extracting these components from an array
/// whose integer child uses a compressed encoding.
pub struct DecimalDataParts {
    /// Decimal precision and scale.
    pub decimal_dtype: DecimalDType,
    /// Native integer values at their stored width.
    pub values: BufferHandle,
    /// The signed native type of `values`.
    pub values_type: DecimalType,
    /// The validity of the integer values.
    pub validity: Validity,
}

/// Accessors for a canonical decimal and its signed integer child.
pub trait DecimalArrayExt: TypedArrayRef<Decimal> + DecimalArraySlotsExt {
    /// Returns the decimal precision and scale.
    fn decimal_dtype(&self) -> DecimalDType {
        *self
            .as_ref()
            .dtype()
            .as_decimal_opt()
            .vortex_expect("Decimal dtype")
    }

    /// Returns the nullability shared by the decimal and its integer child.
    fn nullability(&self) -> Nullability {
        self.as_ref().dtype().nullability()
    }

    /// Logical integer dtype, fixed by the decimal precision and nullability.
    fn values_dtype(&self) -> &DType {
        self.values().dtype()
    }

    /// Integer width of the stored child, which may be narrower than its logical dtype.
    fn values_type(&self) -> DecimalType {
        signed_integer_type(integer::storage_child(self.values()).dtype())
            .vortex_expect("Decimal values have a signed integer dtype")
    }

    /// Returns the decimal precision.
    fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    /// Returns the decimal scale.
    fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }

    /// Borrows a native buffer after the stored child has been materialized.
    fn buffer_handle(&self) -> &BufferHandle {
        integer::buffer_handle(self.values())
            .vortex_expect("Materialize decimal values before borrowing their buffer")
    }

    /// Borrows typed native storage after calling [`Self::materialize_values`].
    fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        assert_eq!(self.values_type(), T::DECIMAL_TYPE);
        Buffer::<T>::from_byte_buffer(self.buffer_handle().as_host().clone())
    }

    /// Decodes the stored child without expanding a Narrow child to its logical width.
    fn materialize_values(&self, ctx: &mut ExecutionCtx) -> VortexResult<DecimalArray> {
        let values = integer::from_buffer(integer::materialize(self.values(), ctx)?)?;
        DecimalArray::from_integer_values(values, self.decimal_dtype())
    }
}

impl<T: TypedArrayRef<Decimal> + DecimalArraySlotsExt> DecimalArrayExt for T {}

impl Array<Decimal> {
    /// Returns the signed integer child with precision-derived logical dtype.
    pub fn values(&self) -> &ArrayRef {
        DecimalArraySlotsExt::values(self)
    }

    /// Returns the logical integer dtype of the child.
    pub fn values_dtype(&self) -> &DType {
        DecimalArrayExt::values_dtype(self)
    }

    /// Returns the stored integer width, without materializing the child.
    pub fn values_type(&self) -> DecimalType {
        DecimalArrayExt::values_type(self)
    }

    /// Returns the decimal precision and scale.
    pub fn decimal_dtype(&self) -> DecimalDType {
        DecimalArrayExt::decimal_dtype(self)
    }

    /// Returns the decimal precision.
    pub fn precision(&self) -> u8 {
        self.decimal_dtype().precision()
    }

    /// Returns the decimal scale.
    pub fn scale(&self) -> i8 {
        self.decimal_dtype().scale()
    }

    /// Borrows native storage after calling [`Self::materialize_values`].
    pub fn buffer_handle(&self) -> &BufferHandle {
        DecimalArrayExt::buffer_handle(self)
    }

    /// Borrows typed storage after calling [`Self::materialize_values`].
    pub fn buffer<T: NativeDecimalType>(&self) -> Buffer<T> {
        DecimalArrayExt::buffer(self)
    }

    /// Decodes stored values without expanding Narrow to its logical width.
    pub fn materialize_values(&self, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        DecimalArrayExt::materialize_values(self, ctx)
    }

    /// Constructs a canonical decimal from a signed integer child.
    ///
    /// The child dtype must equal the integer dtype selected by the decimal precision. Use a
    /// [`NarrowArray`] to retain smaller physical values without changing this logical dtype.
    pub fn try_new_values(values: ArrayRef, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        let dtype = DType::Decimal(decimal_dtype, values.dtype().nullability());
        let len = values.len();
        Array::try_from_parts(
            ArrayParts::new(Decimal, dtype, len, EmptyArrayData)
                .with_slots(DecimalSlots { values }.into_slots()),
        )
    }

    pub(crate) fn from_integer_values(
        values: ArrayRef,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<Self> {
        let required = DecimalType::smallest_decimal_value_type(&decimal_dtype);
        let dtype = integer_dtype(required, values.dtype().nullability());
        let stored = signed_integer_type(values.dtype())
            .vortex_expect("Decimal constructors use signed integer values");
        let values = if stored < required {
            NarrowArray::try_new(values, dtype)?.into_array()
        } else {
            values.cast(dtype)?
        };
        Self::try_new_values(values, decimal_dtype)
    }

    /// Creates a decimal from a native buffer, retaining its stored width when narrower.
    pub fn new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new(buffer, decimal_dtype, validity)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a decimal from a native buffer with structural validation.
    pub fn try_new<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        Self::try_new_handle(
            BufferHandle::new_host(buffer.into_byte_buffer()),
            T::DECIMAL_TYPE,
            decimal_dtype,
            validity,
        )
    }

    /// Creates a decimal from native storage in host or device memory.
    pub fn new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::try_new_handle(values, values_type, decimal_dtype, validity)
            .vortex_expect("DecimalArray construction failed")
    }

    /// Creates a decimal from native storage with structural validation.
    pub fn try_new_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            values.len().is_multiple_of(values_type.byte_width()),
            "Decimal buffer size {} is not divisible by {}",
            values.len(),
            values_type.byte_width(),
        );
        match_each_decimal_value_type!(values_type, |T| {
            vortex_ensure!(
                values.is_aligned_to(Alignment::of::<T>()),
                "Decimal buffer is not aligned for {values_type}"
            );
            Ok::<_, vortex_error::VortexError>(())
        })?;
        let len = values.len() / values_type.byte_width();
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                InvalidArgument: "Decimal validity must have {len} entries, got {validity_len}"
            );
        }
        let values = integer::from_buffer(IntegerBuffer {
            values,
            values_type,
            validity,
        })?;
        Self::from_integer_values(values, decimal_dtype)
    }

    /// Creates a decimal from a native buffer.
    ///
    /// # Safety
    ///
    /// All non-null values must fit the declared decimal precision. The validity must match the
    /// number of values. Structural validation is retained at the integer-child boundary.
    pub unsafe fn new_unchecked<T: NativeDecimalType>(
        buffer: Buffer<T>,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::new(buffer, decimal_dtype, validity)
    }

    /// Creates a decimal from native storage.
    ///
    /// # Safety
    ///
    /// The values must be aligned, contain whole elements, and fit the declared decimal precision.
    /// The validity must match the number of values.
    pub unsafe fn new_unchecked_handle(
        values: BufferHandle,
        values_type: DecimalType,
        decimal_dtype: DecimalDType,
        validity: Validity,
    ) -> Self {
        Self::new_handle(values, values_type, decimal_dtype, validity)
    }

    /// Creates a decimal from non-null values.
    #[expect(
        clippy::same_name_method,
        reason = "matches the iterator constructor convention"
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

    /// Creates a decimal from optional values.
    pub fn from_option_iter<T: NativeDecimalType, I: IntoIterator<Item = Option<T>>>(
        iter: I,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let iter = iter.into_iter();
        let mut values = BufferMut::with_capacity(iter.size_hint().0);
        let mut validity = BitBufferMut::with_capacity(values.capacity());
        for value in iter {
            values.push(value.unwrap_or_default());
            validity.append(value.is_some());
        }
        Self::new(
            values.freeze(),
            decimal_dtype,
            Validity::from(validity.freeze()),
        )
    }

    /// Extracts native storage after calling [`DecimalArrayExt::materialize_values`].
    pub fn into_data_parts(self) -> DecimalDataParts {
        DecimalDataParts {
            decimal_dtype: self.decimal_dtype(),
            values: self.buffer_handle().clone(),
            values_type: self.values_type(),
            validity: self.validity().vortex_expect("Decimal child validity"),
        }
    }

    /// Applies patches, widening stored values when a patch requires a larger integer width.
    pub fn patch(self, patches: &Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array = self.materialize_values(ctx)?;
        let patch_values = patches
            .values()
            .clone()
            .execute::<DecimalArray>(ctx)?
            .materialize_values(ctx)?;
        vortex_ensure!(
            array.decimal_dtype() == patch_values.decimal_dtype(),
            "Decimal patch dtype does not match the array"
        );
        let target = array.values_type().max(patch_values.values_type());
        let values = integer::cast_array(
            array.values(),
            &integer_dtype(target, array.dtype().nullability()),
            ctx,
        )?;
        let patch_values = integer::cast_array(
            patch_values.values(),
            &integer_dtype(target, patch_values.dtype().nullability()),
            ctx,
        )?;
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?.patch(
            array.len(),
            patches.offset(),
            &indices.clone().into_array(),
            &patch_values.validity()?,
            ctx,
        )?;
        let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());
        let values = integer::buffer_handle(&values)?.to_host_sync();
        let patch_values = integer::buffer_handle(&patch_values)?.to_host_sync();
        let values = match_each_decimal_value_type!(target, |T| {
            let values = Buffer::<T>::from_byte_buffer(values).into_mut();
            let patch_values = Buffer::<T>::from_byte_buffer(patch_values);
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                patch_typed(values, indices.as_slice::<I>(), patches.offset(), patch_values)
            })
        });
        Self::try_new_handle(values, target, array.decimal_dtype(), validity)
    }
}

fn patch_typed<T: NativeDecimalType, I: IntegerPType>(
    mut values: BufferMut<T>,
    indices: &[I],
    offset: usize,
    patches: Buffer<T>,
) -> BufferHandle {
    for (&index, value) in indices.iter().zip_eq(patches) {
        let index: usize = index.as_();
        values[index - offset] = value;
    }
    BufferHandle::new_host(values.freeze().into_byte_buffer())
}
