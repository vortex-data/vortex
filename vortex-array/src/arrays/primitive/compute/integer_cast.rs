// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Casts between integer values: primitive integers and decimal storage, optionally rescaled by a
//! power of ten.
//!
//! Every primitive-to-decimal, decimal-to-decimal and decimal-to-integer cast is one of these: the
//! source integers are multiplied or divided by `10^k` and must land in the target's range. Rather
//! than compiling a fused, checked loop for every (source, target) pair, an [`IntegerCast`] splits
//! the cast in two:
//!
//! - [`IntegerCast::first_misfit`] translates the target range back into the source domain, so
//!   validation is a range check (plus a divisibility check for exact division), compiled once
//!   per source type.
//! - [`IntegerCast::convert`] then cannot fail. It divides in the source type, resizes from the
//!   source type to the target type, and multiplies in the target type. Resizing only depends on
//!   the widths (and the source's signedness when widening), and each arithmetic step on one type.
//!
//! A multiply between types of equal width skips the resize: one kernel per source type checks
//! the range and writes the products in the same pass.

use std::mem::MaybeUninit;
use std::slice;

use num_traits::CheckedMul;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::PType;
use crate::dtype::i256;

/// An integer type that holds primitive values or decimal storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntegerType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
}

impl IntegerType {
    /// The integer type of the primitive `ptype`, which must be an integer.
    pub(crate) fn of_ptype(ptype: PType) -> Self {
        match ptype {
            PType::U8 => Self::U8,
            PType::U16 => Self::U16,
            PType::U32 => Self::U32,
            PType::U64 => Self::U64,
            PType::I8 => Self::I8,
            PType::I16 => Self::I16,
            PType::I32 => Self::I32,
            PType::I64 => Self::I64,
            PType::F16 | PType::F32 | PType::F64 => unreachable!("{ptype} is not an integer"),
        }
    }

    /// The integer type that stores decimals of `decimal_type`.
    pub(crate) fn of_decimal(decimal_type: DecimalType) -> Self {
        match decimal_type {
            DecimalType::I8 => Self::I8,
            DecimalType::I16 => Self::I16,
            DecimalType::I32 => Self::I32,
            DecimalType::I64 => Self::I64,
            DecimalType::I128 => Self::I128,
            DecimalType::I256 => Self::I256,
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 => 4,
            Self::U64 | Self::I64 => 8,
            Self::I128 => 16,
            Self::I256 => 32,
        }
    }

    fn alignment(self) -> Alignment {
        match_each_integer_type!(self, |T| { Alignment::of::<T>() })
    }

    /// The inclusive range of the type.
    pub(crate) fn range(self) -> (i256, i256) {
        match_each_integer_type!(self, |T| { (T::MIN.to_i256(), T::MAX.to_i256()) })
    }

    /// The value at `index` of `values`, which hold this type.
    pub(crate) fn value_at(self, values: &ByteBuffer, index: usize) -> i256 {
        match_each_integer_type!(self, |T| { values_as::<T>(values)[index].to_i256() })
    }
}

/// Expands `$body` with `$T` bound to the Rust type of the [`IntegerType`] `$ty`.
macro_rules! match_each_integer_type {
    ($ty:expr, | $T:ident | $body:block) => {{
        match $ty {
            IntegerType::U8 => {
                type $T = u8;
                $body
            }
            IntegerType::U16 => {
                type $T = u16;
                $body
            }
            IntegerType::U32 => {
                type $T = u32;
                $body
            }
            IntegerType::U64 => {
                type $T = u64;
                $body
            }
            IntegerType::I8 => {
                type $T = i8;
                $body
            }
            IntegerType::I16 => {
                type $T = i16;
                $body
            }
            IntegerType::I32 => {
                type $T = i32;
                $body
            }
            IntegerType::I64 => {
                type $T = i64;
                $body
            }
            IntegerType::I128 => {
                type $T = i128;
                $body
            }
            IntegerType::I256 => {
                type $T = i256;
                $body
            }
        }
    }};
}
use match_each_integer_type;

/// How a division by the rescale factor treats a remainder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Remainder {
    /// The value must divide exactly, as when rescaling decimals.
    Reject,
    /// The quotient truncates toward zero, as when casting decimals to integers.
    Truncate,
}

/// A cast of integers from one [`IntegerType`] to another, rescaling by `10^scale_delta`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IntegerCast {
    from: IntegerType,
    to: IntegerType,
    scale_delta: i16,
    remainder: Remainder,
    /// The inclusive range that rescaled values must land in.
    min: i256,
    max: i256,
}

/// What validating the source values of an [`IntegerCast`] takes.
#[derive(Clone, Copy, Debug)]
enum SourceCheck {
    /// Every source value casts.
    AllFit,
    /// No source value casts.
    NoneFit,
    /// Source values cast when they lie in `[min, max]` and are multiples of `divisor`.
    Range {
        min: i256,
        max: i256,
        divisor: Option<i256>,
    },
}

/// How the source values change before resizing (`Divide`) or after it (`Multiply`).
#[derive(Clone, Copy, Debug)]
enum Rescale {
    None,
    Multiply(i256),
    Divide(i256),
    /// The factor overflows i256, so every value that fits rescales to zero.
    Zero,
}

impl IntegerCast {
    /// A cast that rescales by `10^scale_delta`, into the range of `to`.
    pub(crate) fn new(
        from: IntegerType,
        to: IntegerType,
        scale_delta: i16,
        remainder: Remainder,
    ) -> Self {
        let (min, max) = to.range();
        Self {
            from,
            to,
            scale_delta,
            remainder,
            min,
            max,
        }
    }

    /// A cast into decimals of `decimal_dtype`, stored as its canonical values type.
    pub(crate) fn to_decimal(
        from: IntegerType,
        from_scale: i8,
        decimal_dtype: DecimalDType,
    ) -> Self {
        let precision = usize::from(decimal_dtype.precision());
        Self {
            min: i256::MIN_BY_PRECISION[precision],
            max: i256::MAX_BY_PRECISION[precision],
            ..Self::new(
                from,
                IntegerType::of_decimal(decimal_dtype.canonical_values_type()),
                i16::from(decimal_dtype.scale()) - i16::from(from_scale),
                Remainder::Reject,
            )
        }
    }

    pub(crate) fn source_type(&self) -> IntegerType {
        self.from
    }

    fn rescale(&self) -> Rescale {
        let factor = i256::from_i128(10).checked_pow(u32::from(self.scale_delta.unsigned_abs()));
        match (self.scale_delta.signum(), factor) {
            (0, _) => Rescale::None,
            (_, None) => Rescale::Zero,
            (1, Some(factor)) => Rescale::Multiply(factor),
            (_, Some(factor)) => Rescale::Divide(factor),
        }
    }

    /// The inclusive range of source values that rescale into `[min, max]`, and the divisor they
    /// must be multiples of, if any.
    fn source_bounds(&self) -> (i256, i256, Option<i256>) {
        // Both target ranges contain zero, so truncating division rounds `min` up and `max` down.
        match (self.rescale(), self.remainder) {
            (Rescale::None, _) => (self.min, self.max, None),
            (Rescale::Multiply(factor), _) => (self.min / factor, self.max / factor, None),
            (Rescale::Divide(factor), remainder) => (
                self.min.checked_mul(&factor).unwrap_or(i256::MIN),
                self.max.checked_mul(&factor).unwrap_or(i256::MAX),
                (remainder == Remainder::Reject).then_some(factor),
            ),
            // Only zero survives multiplying by, or dividing by, a factor that overflows i256. The
            // scalar cast rejects every other value too, even where the quotient would truncate
            // to zero.
            (Rescale::Zero, _) => (i256::ZERO, i256::ZERO, None),
        }
    }

    /// What validating the source values takes, with the bounds clamped to the source range.
    fn source_check(&self) -> SourceCheck {
        let (mut min, mut max, mut divisor) = self.source_bounds();
        let (from_min, from_max) = self.from.range();
        if divisor.is_some_and(|d| d > from_max) {
            // No nonzero source value is a multiple of the divisor.
            (min, max, divisor) = (min.max(i256::ZERO), max.min(i256::ZERO), None);
        }
        if divisor.is_none() && min <= from_min && max >= from_max {
            return SourceCheck::AllFit;
        }
        if min > max || max < from_min || min > from_max {
            return SourceCheck::NoneFit;
        }
        SourceCheck::Range {
            min: min.max(from_min),
            max: max.min(from_max),
            divisor,
        }
    }

    /// Returns the index of the first valid value in `values` that does not cast, if any.
    ///
    /// `values` hold the source type, and `mask` marks which of them are valid.
    pub(crate) fn first_misfit(&self, values: &ByteBuffer, mask: &Mask) -> Option<usize> {
        let (min, max, divisor) = match self.source_check() {
            SourceCheck::AllFit => return None,
            SourceCheck::NoneFit => return mask.first(),
            SourceCheck::Range { min, max, divisor } => (min, max, divisor),
        };
        match_each_integer_type!(self.from, |T| {
            first_outside::<T>(values_as(values), mask, min, max, divisor)
        })
    }

    /// Casts `values` of the source type, returning values of the target type, or the index of
    /// the first valid value that does not cast.
    ///
    /// Null lanes hold unspecified values afterwards.
    pub(crate) fn cast(&self, values: &ByteBuffer, mask: &Mask) -> Result<ByteBuffer, usize> {
        if let SourceCheck::Range {
            min,
            max,
            divisor: Some(divisor),
        } = self.source_check()
        {
            // Exact division needs the quotients, so divide once and check that the quotient
            // multiplies back to the value, rather than dividing again for the remainder.
            let quotients = match_each_integer_type!(self.from, |T| {
                divide_exactly::<T>(values_as(values), mask, min, max, divisor)
            })?;
            return Ok(resize(&quotients, self.from, self.to));
        }

        if let Rescale::Multiply(factor) = self.rescale()
            && self.from.byte_width() == self.to.byte_width()
        {
            // At equal widths the products have the same bits in either type, so multiply in the
            // source type while checking the range, in one pass into a new buffer.
            let (min, max) = match self.source_check() {
                SourceCheck::AllFit => self.from.range(),
                SourceCheck::NoneFit => {
                    return mask
                        .first()
                        .map_or_else(|| Ok(resize(values, self.from, self.to)), Err);
                }
                SourceCheck::Range { min, max, .. } => (min, max),
            };
            return match_each_integer_type!(self.from, |T| {
                multiply_within::<T>(values_as(values), mask, min, max, factor)
            });
        }

        if let Some(index) = self.first_misfit(values, mask) {
            return Err(index);
        }
        Ok(self.convert(values))
    }

    /// Casts `values` of the source type, every valid one of which has passed
    /// [`Self::first_misfit`], returning values of the target type.
    ///
    /// Null lanes hold unspecified values afterwards.
    fn convert(&self, values: &ByteBuffer) -> ByteBuffer {
        let len = values.len() / self.from.byte_width();
        match self.rescale() {
            Rescale::None => resize(values, self.from, self.to),
            // `cast` multiplies between types of equal width, so `resize` returns a new buffer.
            Rescale::Multiply(factor) => {
                let mut values = resize(values, self.from, self.to).into_mut();
                match_each_integer_type!(self.to, |T| {
                    multiply_in_place::<T>(values_as_mut(&mut values), factor);
                });
                values.freeze()
            }
            // A factor beyond the source range leaves only zero quotients.
            Rescale::Divide(factor) if factor <= self.from.range().1 => {
                let quotients = match_each_integer_type!(self.from, |T| {
                    divide::<T>(values_as(values), T::from_i256(factor))
                });
                resize(&quotients, self.from, self.to)
            }
            Rescale::Divide(_) | Rescale::Zero => match_each_integer_type!(self.to, |T| {
                BufferMut::<T>::zeroed(len).freeze().into_byte_buffer()
            }),
        }
    }
}

/// Returns the index of the first valid value that is outside `[min, max]` or, given a divisor,
/// is not a multiple of it. The bounds and divisor fit `T`.
fn first_outside<T: IntegerValue>(
    values: &[T],
    mask: &Mask,
    min: i256,
    max: i256,
    divisor: Option<i256>,
) -> Option<usize> {
    let (min, max) = (T::from_i256(min), T::from_i256(max));
    match divisor {
        None => first_misfit(values, mask, |v| (min <= v) & (v <= max)),
        Some(divisor) => {
            let divisor = T::from_i256(divisor);
            first_misfit(values, mask, |v| {
                (min <= v) & (v <= max) & (v.rem(divisor) == T::default())
            })
        }
    }
}

/// Returns the index of the first value that `mask` marks valid and that `fits` rejects.
fn first_misfit<T: Copy>(values: &[T], mask: &Mask, fits: impl Fn(T) -> bool) -> Option<usize> {
    if all_valid_lanes(values, mask, &fits) {
        return None;
    }
    first_misfit_index(values, mask, &fits)
}

#[cold]
#[inline(never)]
fn first_misfit_index<T: Copy>(
    values: &[T],
    mask: &Mask,
    fits: &impl Fn(T) -> bool,
) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .position(|(index, &v)| mask.value(index) && !fits(v))
}

/// Returns `true` if `fits` holds for every value that `mask` marks valid.
pub(crate) fn all_valid_lanes<T: Copy>(
    values: &[T],
    mask: &Mask,
    fits: impl Fn(T) -> bool,
) -> bool {
    let all_lanes = || {
        values
            .chunks(64)
            .all(|chunk| chunk.iter().fold(true, |all, &v| all & fits(v)))
    };
    match mask {
        Mask::AllFalse(_) => true,
        Mask::AllTrue(_) => all_lanes(),
        Mask::Values(mask) => {
            all_lanes()
                || values
                    .chunks(64)
                    .zip(mask.bit_buffer().chunks().iter_padded())
                    .all(|(chunk, valid)| {
                        let misfits = chunk.iter().enumerate().fold(0u64, |acc, (i, &v)| {
                            acc | (<u64 as From<bool>>::from(!fits(v)) << i)
                        });
                        misfits & valid == 0
                    })
        }
    }
}

/// Returns `true` if `fits` holds for every pair of lanes of `a` and `b` that `mask` marks valid.
///
/// Lanes are tested a 64-lane chunk at a time without branching, so the test vectorizes. Null
/// lanes usually hold values that fit too, so a masked check first tests every lane and only
/// consults the mask if some lane does not fit.
fn all_valid_pairs<A: Copy, B: Copy>(
    a: &[A],
    b: &[B],
    mask: &Mask,
    fits: impl Fn(A, B) -> bool,
) -> bool {
    let chunks = || a.chunks(64).zip(b.chunks(64));
    let all_lanes =
        || chunks().all(|(a, b)| a.iter().zip(b).fold(true, |all, (&a, &b)| all & fits(a, b)));
    match mask {
        Mask::AllFalse(_) => true,
        Mask::AllTrue(_) => all_lanes(),
        Mask::Values(mask) => {
            all_lanes()
                || chunks()
                    .zip(mask.bit_buffer().chunks().iter_padded())
                    .all(|((a, b), valid)| {
                        let misfits =
                            a.iter()
                                .zip(b)
                                .enumerate()
                                .fold(0u64, |acc, (i, (&a, &b))| {
                                    acc | (<u64 as From<bool>>::from(!fits(a, b)) << i)
                                });
                        misfits & valid == 0
                    })
        }
    }
}

/// Converts `values` from `from` to `to`, keeping every value that fits both.
///
/// Narrowing keeps the low bytes, so it only depends on the two widths. Widening sign- or
/// zero-extends by the source's signedness, so it only depends on the source type and the target
/// width. The kernels write the target's width with the source's signedness (or signed, for the
/// 128- and 256-bit widths), whose bytes match the target type for every value that fits both.
pub(crate) fn resize(values: &ByteBuffer, from: IntegerType, to: IntegerType) -> ByteBuffer {
    let (from_width, to_width) = (from.byte_width(), to.byte_width());
    if from_width == to_width {
        // The source may record a weaker alignment than its pointer has, and the target's
        // constructor checks the recorded one. This copies only if the pointer is misaligned.
        return values.clone().aligned(to.alignment());
    }
    let signed = |width| match width {
        1 => IntegerType::I8,
        2 => IntegerType::I16,
        4 => IntegerType::I32,
        8 => IntegerType::I64,
        16 => IntegerType::I128,
        _ => IntegerType::I256,
    };
    let (from, to) = if from_width > to_width {
        (signed(from_width), signed(to_width))
    } else {
        let unsigned = matches!(
            from,
            IntegerType::U8 | IntegerType::U16 | IntegerType::U32 | IntegerType::U64
        );
        let to = match to_width {
            2 if unsigned => IntegerType::U16,
            4 if unsigned => IntegerType::U32,
            8 if unsigned => IntegerType::U64,
            width => signed(width),
        };
        (from, to)
    };
    match_each_integer_type!(from, |A| {
        match_each_integer_type!(to, |B| { convert_buffer::<A, B>(values_as(values)) })
    })
}

fn convert_buffer<A: IntegerValue, B: IntegerValue>(src: &[A]) -> ByteBuffer {
    let len = src.len();
    let mut out = BufferMut::<B>::with_capacity(len);
    convert::<A, B>(src, &mut out.spare_capacity_mut()[..len]);
    // SAFETY: `convert` initialized every lane.
    unsafe { out.set_len(len) };
    out.freeze().into_byte_buffer()
}

/// The conversion kernel, kept out of line so that each (`A`, `B`) pair is compiled once.
#[inline(never)]
fn convert<A: IntegerValue, B: IntegerValue>(src: &[A], dst: &mut [MaybeUninit<B>]) {
    if A::IS_I256 || B::IS_I256 {
        src.map_into(dst, |v| B::from_i256(v.to_i256()));
    } else {
        // Native integer casts through i128 fold into a single extension or truncation.
        src.map_into(dst, |v| B::from_i128(v.to_i128()));
    }
}

#[inline(never)]
fn multiply_in_place<T: IntegerValue>(values: &mut [T], factor: i256) {
    let factor = T::from_i256(factor);
    for v in values {
        *v = v.wrapping_mul(factor);
    }
}

/// Multiplies `values` by `factor`, or returns the index of the first valid value outside
/// `[min, max]`. The bounds fit `T`, and the factor wraps to `T`.
#[inline(never)]
fn multiply_within<T: IntegerValue>(
    values: &[T],
    mask: &Mask,
    min: i256,
    max: i256,
    factor: i256,
) -> Result<ByteBuffer, usize> {
    let (min, max, factor) = (T::from_i256(min), T::from_i256(max), T::from_i256(factor));
    let fits = |v: T| (min <= v) & (v <= max);

    let len = values.len();
    let mut out = BufferMut::<T>::with_capacity(len);
    let mut all_fit = true;
    for (src, dst) in values
        .chunks(64)
        .zip(out.spare_capacity_mut()[..len].chunks_mut(64))
    {
        let mut chunk_fits = true;
        for (&v, d) in src.iter().zip(dst) {
            chunk_fits &= fits(v);
            d.write(v.wrapping_mul(factor));
        }
        all_fit &= chunk_fits;
    }
    // SAFETY: the loop initialized every lane.
    unsafe { out.set_len(len) };

    if all_fit || all_valid_lanes(values, mask, fits) {
        return Ok(out.freeze().into_byte_buffer());
    }
    Err(first_misfit_index(values, mask, &fits).vortex_expect("some valid value does not fit"))
}

/// Divides `values` by `divisor`, or returns the index of the first valid value that is not a
/// multiple of `divisor` in `[min, max]`.
#[inline(never)]
fn divide_exactly<T: IntegerValue>(
    values: &[T],
    mask: &Mask,
    min: i256,
    max: i256,
    divisor: i256,
) -> Result<ByteBuffer, usize> {
    let (min, max, divisor) = (T::from_i256(min), T::from_i256(max), T::from_i256(divisor));
    let quotients = divide(values, divisor);
    let fits = |v: T, q: T| (min <= v) & (v <= max) & (q.wrapping_mul(divisor) == v);
    if all_valid_pairs(values, values_as::<T>(&quotients), mask, fits) {
        return Ok(quotients);
    }
    let fits = |v: T| (min <= v) & (v <= max) & (v.rem(divisor) == T::default());
    Err(first_misfit_index(values, mask, &fits).vortex_expect("some valid value does not fit"))
}

#[inline(never)]
fn divide<T: IntegerValue>(values: &[T], divisor: T) -> ByteBuffer {
    let len = values.len();
    let mut out = BufferMut::<T>::with_capacity(len);
    values.map_into(&mut out.spare_capacity_mut()[..len], |v| v.div(divisor));
    // SAFETY: `map_into` initialized every lane.
    unsafe { out.set_len(len) };
    out.freeze().into_byte_buffer()
}

/// `values` as a slice of `T`.
fn values_as<T: IntegerValue>(values: &ByteBuffer) -> &[T] {
    // SAFETY: integer buffers are aligned to their type when they are created, and every bit
    // pattern is a valid `T`.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len() / size_of::<T>()) }
}

fn values_as_mut<T: IntegerValue>(values: &mut ByteBufferMut) -> &mut [T] {
    // SAFETY: as for `values_as`; the buffer came from a `BufferMut<T>`.
    unsafe {
        slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<T>(),
            values.len() / size_of::<T>(),
        )
    }
}

/// The operations that integer casts need, for every [`IntegerType`].
///
/// Conversions truncate to the low bits, like `as`.
pub(crate) trait IntegerValue: Copy + Default + PartialOrd + 'static {
    const IS_I256: bool = false;

    fn to_i128(self) -> i128;
    fn from_i128(value: i128) -> Self;
    fn to_i256(self) -> i256;
    fn from_i256(value: i256) -> Self;
    fn wrapping_mul(self, factor: Self) -> Self;
    fn div(self, divisor: Self) -> Self;
    fn rem(self, divisor: Self) -> Self;
}

macro_rules! impl_native_integer_value {
    ($($T:ty),*) => {$(
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_lossless)]
        impl IntegerValue for $T {
            #[inline]
            fn to_i128(self) -> i128 {
                self as i128
            }

            #[inline]
            fn from_i128(value: i128) -> Self {
                value as $T
            }

            #[inline]
            fn to_i256(self) -> i256 {
                i256::from_parts(self as i128 as u128, if (self as i128) < 0 { -1 } else { 0 })
            }

            #[inline]
            fn from_i256(value: i256) -> Self {
                value.to_parts().0 as $T
            }

            #[inline]
            fn wrapping_mul(self, factor: Self) -> Self {
                <$T>::wrapping_mul(self, factor)
            }

            #[inline]
            fn div(self, divisor: Self) -> Self {
                self.wrapping_div(divisor)
            }

            #[inline]
            fn rem(self, divisor: Self) -> Self {
                self.wrapping_rem(divisor)
            }
        }
    )*};
}

impl_native_integer_value!(u8, u16, u32, u64, i8, i16, i32, i64, i128);

impl IntegerValue for i256 {
    const IS_I256: bool = true;

    #[inline]
    fn to_i128(self) -> i128 {
        self.to_parts().0.cast_signed()
    }

    #[inline]
    fn from_i128(value: i128) -> Self {
        i256::from_i128(value)
    }

    #[inline]
    fn to_i256(self) -> i256 {
        self
    }

    #[inline]
    fn from_i256(value: i256) -> Self {
        value
    }

    #[inline]
    fn wrapping_mul(self, factor: Self) -> Self {
        i256::wrapping_mul(&self, factor)
    }

    #[inline]
    fn div(self, divisor: Self) -> Self {
        self / divisor
    }

    #[inline]
    fn rem(self, divisor: Self) -> Self {
        self % divisor
    }
}
