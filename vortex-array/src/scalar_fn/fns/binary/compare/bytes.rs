// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of UTF-8 and binary arrays over canonical [`VarBinViewArray`]s.
//!
//! Equality first compares the leading 8 bytes of each view (length plus 4-byte prefix), which
//! answers most lanes without touching the data buffers. Ordering compares the inline 4-byte
//! prefixes first and only dereferences the full value on a prefix tie. UTF-8 values compare by
//! their byte representation, which matches code-point order.

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::compare::ordering_predicate;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

enum BytesOperand {
    Array {
        values: VarBinViewArray,
        validity: Validity,
    },
    Constant {
        value: Vec<u8>,
        validity: Validity,
    },
}

impl BytesOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            let value = constant_bytes(constant.scalar())?;
            return Ok(Self::Constant {
                value,
                validity: if constant.scalar().dtype().is_nullable() {
                    Validity::AllValid
                } else {
                    Validity::NonNullable
                },
            });
        }

        let values = array.clone().execute::<VarBinViewArray>(ctx)?;
        let validity = values.validity()?;
        Ok(Self::Array { values, validity })
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

fn constant_bytes(scalar: &Scalar) -> VortexResult<Vec<u8>> {
    let value = match scalar.dtype() {
        DType::Utf8(_) => scalar
            .as_utf8()
            .value()
            .map(|s| s.as_str().as_bytes().to_vec()),
        DType::Binary(_) => scalar.as_binary().value().map(|b| b.to_vec()),
        _ => vortex_bail!("expected utf8 or binary scalar, got {}", scalar.dtype()),
    };
    value.ok_or_else(|| vortex_err!("null constant handled by execute_compare"))
}

/// A resolved view over a canonical [`VarBinViewArray`]: the view structs plus borrowed slices of
/// every data buffer, supporting cheap per-lane byte access.
struct ViewsSide<'a> {
    views: &'a [BinaryView],
    buffers: Vec<&'a [u8]>,
}

impl<'a> ViewsSide<'a> {
    fn new(array: &'a VarBinViewArray) -> Self {
        Self {
            views: array.views(),
            buffers: (0..array.data_buffers().len())
                .map(|idx| array.buffer(idx).as_slice())
                .collect(),
        }
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    /// The view at `index` without a bounds check.
    ///
    /// # Safety
    ///
    /// `index` must be strictly less than `self.len()`.
    #[inline]
    unsafe fn view_unchecked(&self, index: usize) -> &'a BinaryView {
        // SAFETY: caller guarantees index < self.views.len().
        unsafe { self.views.get_unchecked(index) }
    }

    /// The full bytes of `view`, which must belong to this side.
    #[inline]
    fn view_bytes(&self, view: &'a BinaryView) -> &'a [u8] {
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let view = view.as_view();
            &self.buffers[view.buffer_index as usize][view.as_range()]
        }
    }
}

/// The leading 8 bytes of a view: the `u32` length plus the first 4 bytes of the value
/// (zero-padded for values shorter than 4 bytes).
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
fn view_head(view: &BinaryView) -> u64 {
    view.as_u128() as u64
}

/// The first 4 value bytes of a view as a big-endian `u32` (zero-padded for values shorter than
/// 4 bytes), so that numeric comparison matches lexicographic byte order.
///
/// Zero padding preserves lexicographic order: a padded position differs from a real byte only
/// when one value is a strict prefix of the other within the first 4 bytes, and the shorter
/// value orders first exactly as `0` orders before any later real byte. A padded tie falls
/// through to the tail or full byte comparison.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
fn view_prefix(view: &BinaryView) -> u32 {
    ((view.as_u128() >> 32) as u32).swap_bytes()
}

/// Value bytes 4..12 of an *inlined* view as a big-endian `u64` (zero-padded past the value's
/// length). Only meaningful for inlined views: reference views store the buffer index and offset
/// in these bytes.
#[inline]
fn view_tail(view: &BinaryView) -> u64 {
    ((view.as_u128() >> 64) as u64).swap_bytes()
}

/// Compare `lhs_view` from `lhs` against `rhs_view` from `rhs` for equality.
#[inline]
fn view_eq(
    lhs: &ViewsSide<'_>,
    lhs_view: &BinaryView,
    rhs: &ViewsSide<'_>,
    rhs_view: &BinaryView,
) -> bool {
    if view_head(lhs_view) != view_head(rhs_view) {
        return false;
    }
    if lhs_view.is_inlined() {
        // Lengths are equal and at most 12: the whole value lives in the view.
        return lhs_view.as_u128() == rhs_view.as_u128();
    }
    // Equal lengths above 12 and equal prefixes: compare the out-of-line suffixes.
    lhs.view_bytes(lhs_view)[4..] == rhs.view_bytes(rhs_view)[4..]
}

/// Compare `lhs_view` from `lhs` against `rhs_view` from `rhs`.
#[inline]
fn view_cmp(
    lhs: &ViewsSide<'_>,
    lhs_view: &BinaryView,
    rhs: &ViewsSide<'_>,
    rhs_view: &BinaryView,
) -> Ordering {
    let lhs_prefix = view_prefix(lhs_view);
    let rhs_prefix = view_prefix(rhs_view);
    if lhs_prefix != rhs_prefix {
        return lhs_prefix.cmp(&rhs_prefix);
    }
    if lhs_view.is_inlined() && rhs_view.is_inlined() {
        // Both values live entirely in their views: compare the remaining 8 (zero-padded)
        // value bytes, then lengths. A tie on padded windows means the shorter value is a
        // prefix of the longer one.
        let lhs_tail = view_tail(lhs_view);
        let rhs_tail = view_tail(rhs_view);
        if lhs_tail != rhs_tail {
            return lhs_tail.cmp(&rhs_tail);
        }
        return lhs_view.len().cmp(&rhs_view.len());
    }
    lhs.view_bytes(lhs_view).cmp(rhs.view_bytes(rhs_view))
}

/// Compare two UTF-8 or binary arrays.
pub(super) fn compare_bytes(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = BytesOperand::try_new(lhs, ctx)?;
    let rhs = BytesOperand::try_new(rhs, ctx)?;
    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (&lhs, &rhs) {
        (BytesOperand::Array { values: l, .. }, BytesOperand::Array { values: r, .. }) => {
            compare_views(&ViewsSide::new(l), &ViewsSide::new(r), op)
        }
        (BytesOperand::Array { values, .. }, BytesOperand::Constant { value, .. }) => {
            compare_views_constant(&ViewsSide::new(values), value, op)
        }
        (BytesOperand::Constant { value, .. }, BytesOperand::Array { values, .. }) => {
            compare_views_constant(&ViewsSide::new(values), value, op.swap())
        }
        (BytesOperand::Constant { value: l, .. }, BytesOperand::Constant { value: r, .. }) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full(ordering_predicate(op)(l.as_slice().cmp(r.as_slice())), len)
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_views(lhs: &ViewsSide<'_>, rhs: &ViewsSide<'_>, op: CompareOperator) -> BitBuffer {
    let len = lhs.len();
    // The unchecked view accesses below index both sides with `i < len`, so this must hold even
    // in release builds.
    assert_eq!(len, rhs.len(), "compared views must have equal lengths");
    // Dispatch the operator outside the lane loop so each predicate inlines into its own loop;
    // a shared `fn(Ordering) -> bool` pointer would cost an indirect call per lane.
    match op {
        CompareOperator::Eq => BitBuffer::collect_bool(len, |i| {
            // SAFETY: `collect_bool` yields i < len == views.len() for both sides.
            unsafe { view_eq(lhs, lhs.view_unchecked(i), rhs, rhs.view_unchecked(i)) }
        }),
        CompareOperator::NotEq => BitBuffer::collect_bool(len, |i| {
            // SAFETY: `collect_bool` yields i < len == views.len() for both sides.
            unsafe { !view_eq(lhs, lhs.view_unchecked(i), rhs, rhs.view_unchecked(i)) }
        }),
        CompareOperator::Gt => collect_ordering_bits(lhs, rhs, Ordering::is_gt),
        CompareOperator::Gte => collect_ordering_bits(lhs, rhs, Ordering::is_ge),
        CompareOperator::Lt => collect_ordering_bits(lhs, rhs, Ordering::is_lt),
        CompareOperator::Lte => collect_ordering_bits(lhs, rhs, Ordering::is_le),
    }
}

/// Bit-pack `predicate(view_cmp(lhs[i], rhs[i]))` over two equal-length view sides.
fn collect_ordering_bits(
    lhs: &ViewsSide<'_>,
    rhs: &ViewsSide<'_>,
    predicate: impl Fn(Ordering) -> bool,
) -> BitBuffer {
    let len = lhs.len();
    assert_eq!(len, rhs.len(), "compared views must have equal lengths");
    BitBuffer::collect_bool(len, |i| {
        // SAFETY: `collect_bool` yields i < len == views.len() for both sides.
        predicate(unsafe { view_cmp(lhs, lhs.view_unchecked(i), rhs, rhs.view_unchecked(i)) })
    })
}

fn compare_views_constant(lhs: &ViewsSide<'_>, constant: &[u8], op: CompareOperator) -> BitBuffer {
    let len = lhs.len();
    // The same head/prefix/tail words a view stores, precomputed once for the constant.
    let mut prefix_bytes = [0u8; 4];
    let prefix_len = constant.len().min(4);
    prefix_bytes[..prefix_len].copy_from_slice(&constant[..prefix_len]);
    let mut tail_bytes = [0u8; 8];
    if constant.len() > 4 {
        let tail_len = (constant.len() - 4).min(8);
        tail_bytes[..tail_len].copy_from_slice(&constant[4..4 + tail_len]);
    }
    let constant_head =
        (constant.len() as u64) | (u64::from(u32::from_le_bytes(prefix_bytes)) << 32);
    let constant_prefix = u32::from_be_bytes(prefix_bytes);
    let constant_tail = u64::from_be_bytes(tail_bytes);
    // The full 16-byte word an inlined view holding `constant` would carry; only meaningful when
    // the constant is short enough to inline, and only reached in that case (a longer constant
    // never head-matches an inlined view).
    let constant_inlined =
        u128::from(constant_head) | (u128::from(u64::from_le_bytes(tail_bytes)) << 64);

    match op {
        CompareOperator::Eq => BitBuffer::collect_bool(len, |i| {
            // SAFETY: `collect_bool` yields i < len == views.len().
            let view = unsafe { lhs.view_unchecked(i) };
            constant_eq(lhs, view, constant, constant_head, constant_inlined)
        }),
        CompareOperator::NotEq => BitBuffer::collect_bool(len, |i| {
            // SAFETY: `collect_bool` yields i < len == views.len().
            let view = unsafe { lhs.view_unchecked(i) };
            !constant_eq(lhs, view, constant, constant_head, constant_inlined)
        }),
        CompareOperator::Gt => collect_constant_ordering_bits(
            lhs,
            constant,
            constant_prefix,
            constant_tail,
            Ordering::is_gt,
        ),
        CompareOperator::Gte => collect_constant_ordering_bits(
            lhs,
            constant,
            constant_prefix,
            constant_tail,
            Ordering::is_ge,
        ),
        CompareOperator::Lt => collect_constant_ordering_bits(
            lhs,
            constant,
            constant_prefix,
            constant_tail,
            Ordering::is_lt,
        ),
        CompareOperator::Lte => collect_constant_ordering_bits(
            lhs,
            constant,
            constant_prefix,
            constant_tail,
            Ordering::is_le,
        ),
    }
}

/// Bit-pack `predicate(constant_cmp(lhs[i], constant))` over one view side.
fn collect_constant_ordering_bits(
    lhs: &ViewsSide<'_>,
    constant: &[u8],
    constant_prefix: u32,
    constant_tail: u64,
    predicate: impl Fn(Ordering) -> bool,
) -> BitBuffer {
    BitBuffer::collect_bool(lhs.len(), |i| {
        // SAFETY: `collect_bool` yields i < len == views.len().
        let view = unsafe { lhs.view_unchecked(i) };
        predicate(constant_cmp(
            lhs,
            view,
            constant,
            constant_prefix,
            constant_tail,
        ))
    })
}

/// Compare a view against a constant for equality using the constant's precomputed head and
/// inline words.
#[inline]
fn constant_eq(
    side: &ViewsSide<'_>,
    view: &BinaryView,
    constant: &[u8],
    constant_head: u64,
    constant_inlined: u128,
) -> bool {
    if view_head(view) != constant_head {
        return false;
    }
    if view.is_inlined() {
        // An equal head implies equal lengths, so the constant is also at most 12 bytes and
        // `constant_inlined` holds its exact inlined representation.
        return view.as_u128() == constant_inlined;
    }
    side.view_bytes(view)[4..] == constant[4..]
}

/// Compare a view against a constant using the constant's precomputed prefix and tail words.
#[inline]
fn constant_cmp(
    side: &ViewsSide<'_>,
    view: &BinaryView,
    constant: &[u8],
    constant_prefix: u32,
    constant_tail: u64,
) -> Ordering {
    let prefix = view_prefix(view);
    if prefix != constant_prefix {
        return prefix.cmp(&constant_prefix);
    }
    if view.is_inlined() {
        // The view's whole value lives in its first 12 (zero-padded) bytes, and the constant's
        // first 12 (zero-padded) bytes are precomputed: compare tails, then lengths. A padded
        // tie means one value is a prefix of the other within the first 12 bytes.
        let tail = view_tail(view);
        if tail != constant_tail {
            return tail.cmp(&constant_tail);
        }
        return (view.len() as usize).cmp(&constant.len());
    }
    side.view_bytes(view).cmp(constant)
}
