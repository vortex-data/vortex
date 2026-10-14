// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lookup arrays of integers: primitives by their bit patterns, and decimals by their unscaled
//! values.
//!
//! A float is an element exactly when the compare kernel would call it equal to an element,
//! which is when their bit patterns match. That distinguishes `-0.0` from `0.0` and one NaN
//! payload from another, so floats are looked up by their bits, as integers. A decimal is looked
//! up by its unscaled value, as the narrowest type that holds every value of its precision. A
//! needle has the precision of the elements, so a valid needle converts to that type whatever its
//! own storage width, and a needle stored at that type converts without a copy.

use num_traits::ToPrimitive;
use num_traits::WrappingSub;
use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::collect_bits;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use super::ArrayHaystack;
use super::HaystackVTable;
use super::HaystackValidity;
use super::haystack_vtable_common;
use super::slot;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::decimal::converted_buffer;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// A set whose span of values needs at most this many bits per element is looked up through a
/// bitmap over the span, bounding the bitmap to a few words per element.
const BITMAP_BITS_PER_ELEMENT: u128 = 64;
/// A span this narrow is looked up through a bitmap whatever the size of the set.
const BITMAP_MIN_BITS: u128 = 1 << 12;

/// Binds `$T` to the integer type of the primitive or decimal array `$values`, `$values` to its
/// integers, and `$needles` to the integers of the canonical needles of its dtype, then evaluates
/// `$body` to the membership bits. Gives the bits and the validity of the needles.
macro_rules! with_integers {
    ($values:expr, $needles:expr, $ctx:expr, | $T:ident, $vals:ident, $nds:ident | $body:expr) => {{
        match ($values.dtype(), $needles) {
            (DType::Primitive(ptype, _), CanonicalView::Primitive(needles)) => {
                let ptype = bit_pattern_ptype(*ptype);
                let values = $values.as_::<Primitive>().reinterpret_cast(ptype);
                let needles = needles.reinterpret_cast(ptype);
                let validity = needles.validity()?;

                match_each_integer_ptype!(ptype, |$T| {
                    let $vals: &[$T] = values.as_slice::<$T>();
                    let $nds: &[$T] = needles.as_slice::<$T>();
                    Ok(($body, validity))
                })
            }
            (DType::Decimal(decimal, _), CanonicalView::Decimal(needles)) => {
                let values_type = DecimalType::smallest_decimal_value_type(decimal);
                let values = $values.as_::<Decimal>().into_owned();
                let needles = needles.into_owned();
                let validity = needles.validity()?;
                let valid = validity.execute_mask(needles.len(), $ctx)?;

                match_each_decimal_value_type!(values_type, |$T| {
                    let values = converted_buffer::<$T>(&values, &Mask::new_true(values.len()))?;
                    let needles = converted_buffer::<$T>(&needles, &valid)?;
                    let $vals: &[$T] = &values;
                    let $nds: &[$T] = &needles;
                    Ok(($body, validity))
                })
            }
            (dtype, needles) => vortex_bail!(
                "An integer haystack of {dtype} cannot look up needles of {}",
                needles.to_array_ref().dtype()
            ),
        }
    }};
}

/// Builds the bitmap haystack of the non-null `members` when their span is dense, and the
/// sorted haystack otherwise.
pub(super) fn try_new(members: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayHaystack> {
    match members.dtype().clone() {
        DType::Primitive(ptype, _) => {
            let bits_ptype = bit_pattern_ptype(ptype);
            let members = members
                .execute::<PrimitiveArray>(ctx)?
                .reinterpret_cast(bits_ptype);

            match_each_integer_ptype!(bits_ptype, |T| {
                let values = members.into_buffer::<T>();
                let primitive = |values: Buffer<T>| {
                    PrimitiveArray::new(values, Validity::NonNullable)
                        .reinterpret_cast(ptype)
                        .into_array()
                };

                Ok(match integer_bitmap(&values, ctx.allocator()) {
                    Some((min, bitmap)) => ArrayHaystack::new(BitmapHaystack::new(
                        primitive(Buffer::from(vec![min])),
                        bitmap,
                    )),
                    None => {
                        ArrayHaystack::new(SortedHaystack::new(primitive(sorted_values(values))))
                    }
                })
            })
        }
        DType::Decimal(decimal, _) => {
            let values_type = DecimalType::smallest_decimal_value_type(&decimal);
            let members = members.execute::<DecimalArray>(ctx)?;
            let all_valid = Mask::new_true(members.len());

            match_each_decimal_value_type!(values_type, |T| {
                let values = converted_buffer::<T>(&members, &all_valid)?;
                let decimals = |values: Buffer<T>| {
                    DecimalArray::new(values, decimal, Validity::NonNullable).into_array()
                };

                Ok(match integer_bitmap(&values, ctx.allocator()) {
                    Some((min, bitmap)) => ArrayHaystack::new(BitmapHaystack::new(
                        decimals(Buffer::from(vec![min])),
                        bitmap,
                    )),
                    None => {
                        ArrayHaystack::new(SortedHaystack::new(decimals(sorted_values(values))))
                    }
                })
            })
        }
        dtype => vortex_bail!("An integer haystack holds primitives or decimals, not {dtype}"),
    }
}

/// A lookup array of integers sorted without duplicates, found by binary search.
///
/// Its one child holds the values, a primitive array sorted by bit pattern or a decimal array
/// sorted by unscaled value. A hash set and, for a handful of elements, a linear scan both lost
/// to the binary search at every set size measured by the `list_contains_set` benchmark, up to
/// 16 384 elements.
#[derive(Clone, Debug)]
pub(crate) struct SortedHaystack;

impl SortedHaystack {
    const VALUES: usize = 0;

    fn new(values: ArrayRef) -> Array<Self> {
        let dtype = values.dtype().clone();
        let len = values.len();
        Array::try_from_parts(ArrayParts::new(
            Self,
            dtype,
            len,
            EmptyArrayData,
            smallvec![Some(values)],
        ))
        .vortex_expect("A sorted haystack is valid by construction")
    }
}

impl VTable for SortedHaystack {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = HaystackValidity;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.haystack.sorted");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 1, "A sorted haystack has one child");
        let values = slots[Self::VALUES]
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("A sorted haystack needs its values"))?;
        vortex_ensure_eq!(
            values.dtype(),
            dtype,
            "A sorted haystack has its values' dtype"
        );
        vortex_ensure!(
            is_integer_array(values),
            "A sorted haystack holds a primitive or decimal array, got {}",
            values.encoding_id()
        );
        vortex_ensure_eq!(values.len(), len, "A sorted haystack has a row per value");
        Ok(())
    }

    haystack_vtable_common!("SortedHaystack");

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "values".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            slot(array.slots(), Self::VALUES).clone(),
        ))
    }
}

impl OperationsVTable<SortedHaystack> for SortedHaystack {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        slot(array.slots(), Self::VALUES).execute_scalar(index, ctx)
    }
}

impl HaystackVTable for SortedHaystack {
    fn contains(
        array: ArrayView<'_, Self>,
        needles: CanonicalView<'_>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let allocator = ctx.allocator().clone();
        with_integers!(
            slot(array.slots(), Self::VALUES),
            needles,
            ctx,
            |T, sorted, needles| collect_bits(
                needles,
                |needle| sorted.binary_search(&needle).is_ok(),
                &allocator
            )
        )
    }
}

/// A lookup array of integers spanning a dense range: one bit per integer of the span.
///
/// Its children are the smallest integer, a primitive or decimal array of one row, and a bitmap
/// over the span above it. Its rows are the integers of the set bits, in increasing order of
/// offset from the smallest.
#[derive(Clone, Debug)]
pub(crate) struct BitmapHaystack;

impl BitmapHaystack {
    const MIN: usize = 0;
    const BITMAP: usize = 1;
    const NAMES: [&'static str; 2] = ["min", "bitmap"];

    fn new(min: ArrayRef, bitmap: BitBuffer) -> Array<Self> {
        let dtype = min.dtype().clone();
        let len = bitmap.true_count();
        let bitmap = BoolArray::new(bitmap, Validity::NonNullable).into_array();
        Array::try_from_parts(ArrayParts::new(
            Self,
            dtype,
            len,
            EmptyArrayData,
            smallvec![Some(min), Some(bitmap)],
        ))
        .vortex_expect("A bitmap haystack is valid by construction")
    }

    fn bitmap(slots: &[Option<ArrayRef>]) -> BitBuffer {
        slot(slots, Self::BITMAP).as_::<Bool>().to_bit_buffer()
    }
}

impl VTable for BitmapHaystack {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = HaystackValidity;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.haystack.bitmap");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        _len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 2, "A bitmap haystack has two children");
        let (Some(min), Some(bitmap)) = (&slots[Self::MIN], &slots[Self::BITMAP]) else {
            vortex_bail!("A bitmap haystack needs its smallest value and its bitmap");
        };
        vortex_ensure_eq!(
            min.dtype(),
            dtype,
            "A bitmap haystack has its values' dtype"
        );
        vortex_ensure_eq!(min.len(), 1, "A bitmap haystack has one smallest value");
        vortex_ensure!(
            is_integer_array(min),
            "A bitmap haystack holds its smallest value as a primitive or decimal array, got {}",
            min.encoding_id()
        );
        vortex_ensure!(
            bitmap.is::<Bool>(),
            "A bitmap haystack needs a bool array bitmap, got {}",
            bitmap.encoding_id()
        );
        Ok(())
    }

    haystack_vtable_common!("BitmapHaystack");

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        Self::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let bitmap = Self::bitmap(array.slots());
        let values = offset_values(slot(array.slots(), Self::MIN), bitmap.set_indices())?;
        Ok(ExecutionResult::done(values))
    }
}

impl OperationsVTable<BitmapHaystack> for BitmapHaystack {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let offset = Self::bitmap(array.slots())
            .select(index)
            .ok_or_else(|| vortex_err!("A bitmap haystack has no row {index}"))?;
        offset_values(slot(array.slots(), Self::MIN), [offset].into_iter())?.execute_scalar(0, ctx)
    }
}

impl HaystackVTable for BitmapHaystack {
    fn contains(
        array: ArrayView<'_, Self>,
        needles: CanonicalView<'_>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let allocator = ctx.allocator().clone();
        let bitmap = Self::bitmap(array.slots());
        with_integers!(
            slot(array.slots(), Self::MIN),
            needles,
            ctx,
            |T, min, needles| {
                let min = min[0];
                collect_bits(
                    needles,
                    // A needle below the smallest element wraps past the bitmap, so one comparison
                    // checks both bounds.
                    |needle| {
                        needle
                            .offset_from(min)
                            .is_some_and(|offset| offset < bitmap.len() && bitmap.value(offset))
                    },
                    &allocator,
                )
            }
        )
    }
}

/// Whether `array` holds integers in a form the integer haystacks read without executing it.
fn is_integer_array(array: &ArrayRef) -> bool {
    array.is::<Primitive>() || array.is::<Decimal>()
}

/// The integers `offsets` above the one integer of `min`, a primitive or decimal array.
fn offset_values(min: &ArrayRef, offsets: impl Iterator<Item = usize>) -> VortexResult<ArrayRef> {
    Ok(match min.dtype() {
        DType::Primitive(ptype, _) => {
            let ptype = *ptype;
            let bits_ptype = bit_pattern_ptype(ptype);
            let min = min.as_::<Primitive>().reinterpret_cast(bits_ptype);
            match_each_integer_ptype!(bits_ptype, |T| {
                let min = min.as_slice::<T>()[0];
                let values: Buffer<T> = offsets.map(|offset| min.add_offset(offset)).collect();
                PrimitiveArray::new(values, Validity::NonNullable)
                    .reinterpret_cast(ptype)
                    .into_array()
            })
        }
        DType::Decimal(decimal, _) => {
            let decimal = *decimal;
            let values_type = DecimalType::smallest_decimal_value_type(&decimal);
            let min = min.as_::<Decimal>().into_owned();
            match_each_decimal_value_type!(values_type, |T| {
                let min = converted_buffer::<T>(&min, &Mask::new_true(1))?[0];
                let values: Buffer<T> = offsets.map(|offset| min.add_offset(offset)).collect();
                DecimalArray::new(values, decimal, Validity::NonNullable).into_array()
            })
        }
        dtype => vortex_bail!("An integer haystack holds primitives or decimals, not {dtype}"),
    })
}

/// The integer type with a float's bit pattern, or the type itself for an integer.
fn bit_pattern_ptype(ptype: PType) -> PType {
    match ptype {
        PType::F16 => PType::U16,
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
        _ => ptype,
    }
}

/// An unsigned modular distance, rejecting offsets too wide to address a bitmap.
/// Keeping the arithmetic at the physical width also handles signed ranges spanning zero.
trait SetInteger: Copy + Ord + Send + Sync + 'static {
    fn offset_from(self, min: Self) -> Option<usize>;

    /// The integer `offset` above `self`, for an offset within a bitmap's span.
    fn add_offset(self, offset: usize) -> Self;
}

macro_rules! impl_set_integer {
    ($($signed:ty => $unsigned:ty),* $(,)?) => {
        $(impl SetInteger for $signed {
            fn offset_from(self, min: Self) -> Option<usize> {
                usize::try_from((self as $unsigned).wrapping_sub(min as $unsigned)).ok()
            }

            fn add_offset(self, offset: usize) -> Self {
                let offset = <$unsigned>::try_from(offset)
                    .vortex_expect("A bitmap offset fits the integer type");
                (self as $unsigned).wrapping_add(offset) as $signed
            }
        })*
    };
}

impl_set_integer!(
    u8 => u8, u16 => u16, u32 => u32, u64 => u64,
    i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128,
);

impl SetInteger for i256 {
    fn offset_from(self, min: Self) -> Option<usize> {
        self.wrapping_sub(&min).to_usize()
    }

    fn add_offset(self, offset: usize) -> Self {
        let offset = u64::try_from(offset).vortex_expect("A bitmap offset fits in 64 bits");
        i256::wrapping_add(&self, i256::from(offset))
    }
}

/// The smallest integer and a bitmap over the span of `values` above it, when the span is dense.
fn integer_bitmap<T: SetInteger>(
    values: &[T],
    allocator: &BufferAllocatorRef,
) -> Option<(T, BitBuffer)> {
    let min = *values.iter().min()?;
    let max = *values.iter().max()?;
    let span = max.offset_from(min)?;
    if span == usize::MAX
        || span as u128 >= (values.len() as u128 * BITMAP_BITS_PER_ELEMENT).max(BITMAP_MIN_BITS)
    {
        return None;
    }
    let mut bitmap = BitBufferMut::from_buffer(
        BufferMut::zeroed_in((span + 1).div_ceil(8), allocator.clone()),
        0,
        span + 1,
    );
    for &value in values {
        bitmap.set(
            value
                .offset_from(min)
                .vortex_expect("value within bitmap span"),
        );
    }
    Some((min, bitmap.freeze()))
}

fn sorted_values<T: Copy + Ord + Send + Sync + 'static>(values: Buffer<T>) -> Buffer<T> {
    if values.is_sorted_by(|a, b| a < b) {
        return values;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    Buffer::from(sorted)
}
