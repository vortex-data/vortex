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
use crate::arrays::varbinview::ResolvedViews;
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

/// Compare `lhs_view` from `lhs` against `rhs_view` from `rhs` for equality.
#[inline]
fn view_eq(
    lhs: &ResolvedViews<'_>,
    lhs_view: &BinaryView,
    rhs: &ResolvedViews<'_>,
    rhs_view: &BinaryView,
) -> bool {
    if lhs_view.head() != rhs_view.head() {
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
    lhs: &ResolvedViews<'_>,
    lhs_view: &BinaryView,
    rhs: &ResolvedViews<'_>,
    rhs_view: &BinaryView,
) -> Ordering {
    let lhs_prefix = lhs_view.order_prefix();
    let rhs_prefix = rhs_view.order_prefix();
    if lhs_prefix != rhs_prefix {
        return lhs_prefix.cmp(&rhs_prefix);
    }
    if lhs_view.is_inlined() && rhs_view.is_inlined() {
        // Both values live entirely in their views: compare the remaining 8 (zero-padded)
        // value bytes, then lengths. A tie on padded windows means the shorter value is a
        // prefix of the longer one.
        let lhs_tail = lhs_view.order_tail();
        let rhs_tail = rhs_view.order_tail();
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
            compare_views(&ResolvedViews::new(l), &ResolvedViews::new(r), op)
        }
        (BytesOperand::Array { values, .. }, BytesOperand::Constant { value, .. }) => {
            compare_views_constant(&ResolvedViews::new(values), value, op)
        }
        (BytesOperand::Constant { value, .. }, BytesOperand::Array { values, .. }) => {
            compare_views_constant(&ResolvedViews::new(values), value, op.swap())
        }
        (BytesOperand::Constant { value: l, .. }, BytesOperand::Constant { value: r, .. }) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full(ordering_predicate(op)(l.as_slice().cmp(r.as_slice())), len)
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_views(
    lhs: &ResolvedViews<'_>,
    rhs: &ResolvedViews<'_>,
    op: CompareOperator,
) -> BitBuffer {
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
    lhs: &ResolvedViews<'_>,
    rhs: &ResolvedViews<'_>,
    predicate: impl Fn(Ordering) -> bool,
) -> BitBuffer {
    let len = lhs.len();
    assert_eq!(len, rhs.len(), "compared views must have equal lengths");
    BitBuffer::collect_bool(len, |i| {
        // SAFETY: `collect_bool` yields i < len == views.len() for both sides.
        predicate(unsafe { view_cmp(lhs, lhs.view_unchecked(i), rhs, rhs.view_unchecked(i)) })
    })
}

fn compare_views_constant(
    lhs: &ResolvedViews<'_>,
    constant: &[u8],
    op: CompareOperator,
) -> BitBuffer {
    let len = lhs.len();
    // The same head/prefix/tail words a view stores, precomputed once for the constant.
    let constant_head = BinaryView::head_of(constant);
    let constant_prefix = BinaryView::order_prefix_of(constant);
    let constant_tail = BinaryView::order_tail_of(constant);
    // The full 16-byte word an inlined view holding `constant` would carry; only meaningful when
    // the constant is short enough to inline, and only reached in that case (a longer constant
    // never head-matches an inlined view).
    let constant_inlined = if constant.len() <= BinaryView::MAX_INLINED_SIZE {
        BinaryView::new_inlined(constant).as_u128()
    } else {
        0
    };

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
    lhs: &ResolvedViews<'_>,
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
    side: &ResolvedViews<'_>,
    view: &BinaryView,
    constant: &[u8],
    constant_head: u64,
    constant_inlined: u128,
) -> bool {
    if view.head() != constant_head {
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
    side: &ResolvedViews<'_>,
    view: &BinaryView,
    constant: &[u8],
    constant_prefix: u32,
    constant_tail: u64,
) -> Ordering {
    let prefix = view.order_prefix();
    if prefix != constant_prefix {
        return prefix.cmp(&constant_prefix);
    }
    if view.is_inlined() {
        // The view's whole value lives in its first 12 (zero-padded) bytes, and the constant's
        // first 12 (zero-padded) bytes are precomputed: compare tails, then lengths. A padded
        // tie means one value is a prefix of the other within the first 12 bytes.
        let tail = view.order_tail();
        if tail != constant_tail {
            return tail.cmp(&constant_tail);
        }
        return (view.len() as usize).cmp(&constant.len());
    }
    side.view_bytes(view).cmp(constant)
}
