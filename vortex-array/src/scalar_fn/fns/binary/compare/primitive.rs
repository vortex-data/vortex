// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of primitive arrays with packed bitmap output.

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::collect_bool_word;
use vortex_compute::lane_kernels::bit_buffer_from_words;
use vortex_compute::lane_kernels::collect_bits;
use vortex_compute::lane_kernels::collect_zip_bits;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_float_ptype;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::primitive_operand::PrimitiveOperand;
use crate::scalar_fn::fns::operators::CompareMode;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::NanOrder;
use crate::scalar_fn::fns::operators::SignedZeros;

/// Compare two primitive arrays of the same [`PType`].
///
/// In [`CompareMode::TOTAL_ORDER`], floats use a total ordering over their bit patterns, including
/// NaN signs and payloads. Equality is bitwise, so signed zeros and distinct NaN payloads remain
/// distinct. Other modes change the order of signed zeros and `NaN` values.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    mode: CompareMode,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;

    if mode.is_total_order() || !ptype.is_float() {
        return match_each_native_ptype!(ptype, |T| {
            compare_primitive_typed::<T, TotalOrderPredicates>(lhs, rhs, op, nullability, ctx)
        });
    }

    macro_rules! compare_float {
        ($zeros_equal:expr, $nan:expr) => {
            match_each_float_ptype!(ptype, |T| {
                compare_primitive_typed::<T, ModePredicates<$zeros_equal, $nan>>(
                    lhs,
                    rhs,
                    op,
                    nullability,
                    ctx,
                )
            })
        };
    }

    match (mode.signed_zeros(), mode.nan_order()) {
        (SignedZeros::Distinct, NanOrder::TotalOrder) => compare_float!(false, NAN_TOTAL_ORDER),
        (SignedZeros::Distinct, NanOrder::Canonical) => compare_float!(false, NAN_CANONICAL),
        (SignedZeros::Distinct, NanOrder::Ieee754) => compare_float!(false, NAN_IEEE754),
        (SignedZeros::Equal, NanOrder::TotalOrder) => compare_float!(true, NAN_TOTAL_ORDER),
        (SignedZeros::Equal, NanOrder::Canonical) => compare_float!(true, NAN_CANONICAL),
        (SignedZeros::Equal, NanOrder::Ieee754) => compare_float!(true, NAN_IEEE754),
    }
}

/// The six comparison predicates of a [`CompareMode`] over the native type `T`.
trait Predicates<T> {
    fn eq(lhs: T, rhs: T) -> bool;
    fn ne(lhs: T, rhs: T) -> bool;
    fn gt(lhs: T, rhs: T) -> bool;
    fn ge(lhs: T, rhs: T) -> bool;
    fn lt(lhs: T, rhs: T) -> bool;
    fn le(lhs: T, rhs: T) -> bool;
}

/// The predicates of [`CompareMode::TOTAL_ORDER`].
struct TotalOrderPredicates;

impl<T: NativePType> Predicates<T> for TotalOrderPredicates {
    #[inline]
    fn eq(lhs: T, rhs: T) -> bool {
        lhs.is_eq(rhs)
    }

    #[inline]
    fn ne(lhs: T, rhs: T) -> bool {
        !lhs.is_eq(rhs)
    }

    #[inline]
    fn gt(lhs: T, rhs: T) -> bool {
        lhs.is_gt(rhs)
    }

    #[inline]
    fn ge(lhs: T, rhs: T) -> bool {
        lhs.is_ge(rhs)
    }

    #[inline]
    fn lt(lhs: T, rhs: T) -> bool {
        lhs.is_lt(rhs)
    }

    #[inline]
    fn le(lhs: T, rhs: T) -> bool {
        lhs.is_le(rhs)
    }
}

/// [`NanOrder::TotalOrder`] as a const generic parameter.
pub(super) const NAN_TOTAL_ORDER: u8 = 0;
/// [`NanOrder::Canonical`] as a const generic parameter.
pub(super) const NAN_CANONICAL: u8 = 1;
/// [`NanOrder::Ieee754`] as a const generic parameter.
pub(super) const NAN_IEEE754: u8 = 2;

/// The predicates of a float [`CompareMode`]. `ZEROS_EQUAL` is [`SignedZeros::Equal`], and `NAN`
/// is one of the `NAN_*` constants. The const parameters let each mode compile to its own loop.
pub(super) struct ModePredicates<const ZEROS_EQUAL: bool, const NAN: u8>;

impl<const ZEROS_EQUAL: bool, const NAN: u8> ModePredicates<ZEROS_EQUAL, NAN> {
    /// Orders two values in this mode, or returns `None` when they are unordered.
    #[inline]
    pub(super) fn compare<T: NativePType>(lhs: T, rhs: T) -> Option<Ordering> {
        let (lhs, rhs) = if ZEROS_EQUAL {
            (positive_zero(lhs), positive_zero(rhs))
        } else {
            (lhs, rhs)
        };

        match NAN {
            NAN_CANONICAL => Some(match (lhs.is_nan(), rhs.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => lhs.total_compare(rhs),
            }),
            NAN_IEEE754 => (!lhs.is_nan() && !rhs.is_nan()).then(|| lhs.total_compare(rhs)),
            _ => Some(lhs.total_compare(rhs)),
        }
    }
}

/// Replaces `-0.0` with `+0.0`. Other values, including `NaN`, are unchanged.
#[inline]
fn positive_zero<T: NativePType>(value: T) -> T {
    if value == T::zero() { T::zero() } else { value }
}

impl<T: NativePType, const ZEROS_EQUAL: bool, const NAN: u8> Predicates<T>
    for ModePredicates<ZEROS_EQUAL, NAN>
{
    #[inline]
    fn eq(lhs: T, rhs: T) -> bool {
        Self::compare(lhs, rhs) == Some(Ordering::Equal)
    }

    #[inline]
    fn ne(lhs: T, rhs: T) -> bool {
        !Self::eq(lhs, rhs)
    }

    #[inline]
    fn gt(lhs: T, rhs: T) -> bool {
        Self::compare(lhs, rhs) == Some(Ordering::Greater)
    }

    #[inline]
    fn ge(lhs: T, rhs: T) -> bool {
        matches!(
            Self::compare(lhs, rhs),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }

    #[inline]
    fn lt(lhs: T, rhs: T) -> bool {
        Self::compare(lhs, rhs) == Some(Ordering::Less)
    }

    #[inline]
    fn le(lhs: T, rhs: T) -> bool {
        matches!(
            Self::compare(lhs, rhs),
            Some(Ordering::Less | Ordering::Equal)
        )
    }
}

fn compare_primitive_typed<T: NativePType, P: Predicates<T>>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "compare operator requires equal lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        );
    }

    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slices::<T, P>(lhs, rhs, op, ctx.allocator()),
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => compare_slice_constant::<T, P>(lhs, *rhs, op, ctx.allocator()),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slice_constant::<T, P>(rhs, *lhs, op.swap(), ctx.allocator()),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full_in(
                apply_op::<T, P>(*lhs, *rhs, op),
                len,
                ctx.allocator().clone(),
            )
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => {
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array(),
            );
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn apply_op<T: NativePType, P: Predicates<T>>(lhs: T, rhs: T, op: CompareOperator) -> bool {
    match op {
        CompareOperator::Eq => P::eq(lhs, rhs),
        CompareOperator::NotEq => P::ne(lhs, rhs),
        CompareOperator::Gt => P::gt(lhs, rhs),
        CompareOperator::Gte => P::ge(lhs, rhs),
        CompareOperator::Lt => P::lt(lhs, rhs),
        CompareOperator::Lte => P::le(lhs, rhs),
    }
}

fn compare_slices<T: NativePType, P: Predicates<T>>(
    lhs: &[T],
    rhs: &[T],
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // Dispatch the operator outside the lane loop so each instantiation vectorizes a single
    // branch-free predicate.
    match op {
        CompareOperator::Eq => collect_zip_bits_dispatch(lhs, rhs, P::eq, allocator),
        CompareOperator::NotEq => collect_zip_bits_dispatch(lhs, rhs, P::ne, allocator),
        CompareOperator::Gt => collect_zip_bits_dispatch(lhs, rhs, P::gt, allocator),
        CompareOperator::Gte => collect_zip_bits_dispatch(lhs, rhs, P::ge, allocator),
        CompareOperator::Lt => collect_zip_bits_dispatch(lhs, rhs, P::lt, allocator),
        CompareOperator::Lte => collect_zip_bits_dispatch(lhs, rhs, P::le, allocator),
    }
}

fn compare_slice_constant<T: NativePType, P: Predicates<T>>(
    lhs: &[T],
    rhs: T,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_bits_dispatch(lhs, |a: T| P::eq(a, rhs), allocator),
        CompareOperator::NotEq => collect_bits_dispatch(lhs, |a: T| P::ne(a, rhs), allocator),
        CompareOperator::Gt => collect_bits_dispatch(lhs, |a: T| P::gt(a, rhs), allocator),
        CompareOperator::Gte => collect_bits_dispatch(lhs, |a: T| P::ge(a, rhs), allocator),
        CompareOperator::Lt => collect_bits_dispatch(lhs, |a: T| P::lt(a, rhs), allocator),
        CompareOperator::Lte => collect_bits_dispatch(lhs, |a: T| P::le(a, rhs), allocator),
    }
}

fn collect_bits_dispatch<T: NativePType>(
    values: &[T],
    f: impl Fn(T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    // This type check folds away during monomorphization. Word packing improves float32
    // comparisons; wider types retain the lane kernel to avoid AVX2 constant-case regressions.
    if matches!(T::PTYPE, PType::I8 | PType::U8 | PType::F32) {
        collect_bits_narrow(values, f, allocator)
    } else {
        collect_bits(values, f, allocator)
    }
}

fn collect_zip_bits_dispatch<T: NativePType>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if matches!(T::PTYPE, PType::I64 | PType::U64 | PType::F64) {
        // Fill-and-pack improves wide array pairs on NEON. Constants keep their existing loop.
        return collect_zip_bits_wide(lhs, rhs, f, allocator);
    }

    if matches!(T::PTYPE, PType::I8 | PType::U8 | PType::F32) {
        collect_zip_bits_narrow(lhs, rhs, f, allocator)
    } else {
        collect_zip_bits(lhs, rhs, f, allocator)
    }
}

fn collect_bits_narrow<T: Copy>(
    values: &[T],
    f: impl Fn(T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    let (chunks, tail) = values.as_chunks::<64>();
    let mut words = BufferMut::<u64>::zeroed_in(values.len().div_ceil(64), allocator.clone());
    // Fixed-size chunks let the compiler prove the predicate's indexing stays in bounds.
    for (word, chunk) in words.iter_mut().zip(chunks) {
        *word = collect_bool_word(64, |i| f(chunk[i]));
    }
    if !tail.is_empty() {
        words[chunks.len()] = collect_bool_word(tail.len(), |i| f(tail[i]));
    }
    bit_buffer_from_words(words, values.len())
}

fn collect_zip_bits_narrow<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    assert_eq!(lhs.len(), rhs.len());
    let (left_chunks, left_tail) = lhs.as_chunks::<64>();
    let (right_chunks, right_tail) = rhs.as_chunks::<64>();
    let mut words = BufferMut::<u64>::zeroed_in(lhs.len().div_ceil(64), allocator.clone());
    for ((word, left), right) in words.iter_mut().zip(left_chunks).zip(right_chunks) {
        *word = collect_bool_word(64, |i| f(left[i], right[i]));
    }
    if !left_tail.is_empty() {
        words[left_chunks.len()] =
            collect_bool_word(left_tail.len(), |i| f(left_tail[i], right_tail[i]));
    }
    bit_buffer_from_words(words, lhs.len())
}

/// Keep the predicate free of bounds checks so NEON can fuse comparison and bitmap packing.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn collect_zip_bits_wide<T: Copy>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    assert_eq!(lhs.len(), rhs.len());
    BitBuffer::collect_bool_multiversioned_in(
        lhs.len(),
        |index| {
            // SAFETY: the collector calls with `index < lhs.len()`, and the slice lengths match.
            let (lhs, rhs) = unsafe { (*lhs.get_unchecked(index), *rhs.get_unchecked(index)) };
            f(lhs, rhs)
        },
        allocator.clone(),
    )
}
