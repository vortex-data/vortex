// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`].
//!
//! [`Compare`] keeps its existing scalar-function hooks while delegating primitive row execution to
//! a private [`RowFn`]. The executor owns decoding, constants, validity, and Boolean packing.

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_float_ptype;
use crate::match_each_native_ptype;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::compare::Compare;
use crate::scalar_fn::builtin::compare::CompareOptions;
use crate::scalar_fn::builtin::operators::CompareMode;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::scalar_fn::builtin::operators::FloatOrder;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowInputs;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_row_inputs;

/// Compare two primitive arrays of the same [`PType`].
///
/// The caller must supply equal-length operands with matching primitive dtypes, ignoring
/// nullability. In [`CompareMode::TOTAL_ORDER`], predicates use [`NativePType`] comparison
/// semantics, including total ordering and bitwise equality for floats. Other modes make the
/// signed zeros equal, and can change the order of `NaN` values.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    mode: CompareMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;

    // Use the total-order predicates when they give the same result: they are the fastest.
    let zero = mode
        .total_order_zero(op, rhs.as_constant())
        .map(|zero| ConstantArray::new(zero, rhs.len()).into_array());
    let total_order =
        zero.is_some() || mode.agrees_with_total_order(op, lhs.dtype(), rhs.as_constant());
    let mode = if total_order {
        CompareMode::TOTAL_ORDER
    } else {
        mode
    };
    let rhs = zero.as_ref().unwrap_or(rhs);

    execute_row_inputs(
        &PrimitiveCompare,
        &CompareOptions::new(op, mode),
        &CompareArgs::new(lhs, rhs, ptype),
        ctx,
    )
}

/// Borrowed operands and contiguous dispatch dtypes, avoiding an owned argument list.
struct CompareArgs<'a> {
    /// The operands in comparison order, without cloning their array references.
    arrays: [&'a ArrayRef; 2],
    /// Primitive dtypes built from the common physical type and each operand's nullability.
    dtypes: [DType; 2],
}

impl<'a> CompareArgs<'a> {
    /// Pair two primitive operands whose physical type is `ptype`.
    ///
    /// Building each `DType::Primitive` instead of cloning the operand dtype lets the compiler drop
    /// the dtypes without a runtime match.
    #[expect(clippy::inline_always)]
    #[inline(always)]
    fn new(lhs: &'a ArrayRef, rhs: &'a ArrayRef, ptype: PType) -> Self {
        Self {
            arrays: [lhs, rhs],
            dtypes: [
                DType::Primitive(ptype, lhs.dtype().nullability()),
                DType::Primitive(ptype, rhs.dtype().nullability()),
            ],
        }
    }
}

impl RowInputs for CompareArgs<'_> {
    fn num_inputs(&self) -> usize {
        2
    }

    fn row_count(&self) -> usize {
        self.arrays[0].len()
    }

    fn input(&self, index: usize) -> &ArrayRef {
        self.arrays[index]
    }

    fn dtypes(&self) -> Option<&[DType]> {
        Some(&self.dtypes)
    }
}

/// A private row kernel used by [`Compare`] for primitive comparisons.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        // Errors must name the caller's `Compare` function. This kernel is never registered or
        // serialized independently, so it can share that ID. Registering it would require a new ID.
        ScalarFnVTable::id(&Compare)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let [lhs_dtype, _] = args else {
            vortex_bail!(
                "a primitive comparison requires two operands, got {}",
                args.len(),
            );
        };
        let ptype = PType::try_from(lhs_dtype)?;

        let op = options.operator();
        let mode = options.compare_mode();

        if !ptype.is_float() {
            return match_each_native_ptype!(ptype, |T| { visit_compare::<T, V>(op, visitor) });
        }

        match_each_float_ptype!(ptype, |T| {
            visit_float_compare::<T, V>(op, mode.float_order(), visitor)
        })
    }
}

/// [`FloatOrder::ZerosEqual`] as a const generic parameter.
pub(super) const ZEROS_EQUAL: u8 = 0;
/// [`FloatOrder::Canonical`] as a const generic parameter.
pub(super) const CANONICAL: u8 = 1;
/// [`FloatOrder::Ieee754`] as a const generic parameter.
pub(super) const IEEE754: u8 = 2;

/// A [`FloatOrder`] where `-0.0 = +0.0`. `ORDER` is one of the constants above. The const
/// parameter lets each order compile to its own loop.
pub(super) struct ZerosEqualOrder<const ORDER: u8>;

impl<const ORDER: u8> ZerosEqualOrder<ORDER> {
    /// Orders two values, or returns `None` when they are unordered.
    #[inline]
    pub(super) fn compare<T: NativePType>(lhs: T, rhs: T) -> Option<Ordering> {
        let lhs = positive_zero(lhs);
        let rhs = positive_zero(rhs);

        match ORDER {
            CANONICAL => Some(match (lhs.is_nan(), rhs.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => lhs.total_compare(rhs),
            }),
            IEEE754 => (!lhs.is_nan() && !rhs.is_nan()).then(|| lhs.total_compare(rhs)),
            _ => Some(lhs.total_compare(rhs)),
        }
    }
}

/// Replaces `-0.0` with `+0.0`. Other values, including `NaN`, are unchanged.
#[inline]
fn positive_zero<T: NativePType>(value: T) -> T {
    if value == T::zero() { T::zero() } else { value }
}

/// Selects the monomorphic predicates of `order` before traversal.
fn visit_float_compare<T, V>(
    op: CompareOperator,
    order: FloatOrder,
    visitor: V,
) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    match order {
        FloatOrder::TotalOrder => visit_compare::<T, V>(op, visitor),
        FloatOrder::ZerosEqual => visit_default_float_compare::<T, V>(op, visitor),
        FloatOrder::Canonical => visit_zeros_equal_compare::<T, V, CANONICAL>(op, visitor),
        FloatOrder::Ieee754 => visit_zeros_equal_compare::<T, V, IEEE754>(op, visitor),
    }
}

/// Visits `op` over floats in [`FloatOrder::ZerosEqual`].
///
/// The order differs from the total order only for a `-0.0` and a `+0.0`, and IEEE 754 equality
/// is true for that pair. IEEE 754 equality is also true only for pairs that are equal in this
/// order. So each predicate is a total-order predicate and one IEEE 754 equality, without a
/// branch for each value.
fn visit_default_float_compare<T, V>(
    op: CompareOperator,
    visitor: V,
) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    match op {
        CompareOperator::Eq => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| lhs.is_eq(rhs) | (lhs == rhs))
        }
        CompareOperator::NotEq => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| !(lhs.is_eq(rhs) | (lhs == rhs)))
        }
        CompareOperator::Gt => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| !(lhs.is_le(rhs) | (lhs == rhs)))
        }
        CompareOperator::Gte => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| lhs.is_ge(rhs) | (lhs == rhs))
        }
        CompareOperator::Lt => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| !(lhs.is_ge(rhs) | (lhs == rhs)))
        }
        CompareOperator::Lte => {
            visit_compare_with::<T, V>(visitor, |lhs, rhs| lhs.is_le(rhs) | (lhs == rhs))
        }
    }
}

/// Visits `op` over floats in the order of [`ZerosEqualOrder`]. An unordered pair satisfies only
/// [`CompareOperator::NotEq`].
fn visit_zeros_equal_compare<T, V, const ORDER: u8>(
    op: CompareOperator,
    visitor: V,
) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    let compare = ZerosEqualOrder::<ORDER>::compare::<T>;

    match op {
        CompareOperator::Eq => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs) == Some(Ordering::Equal)
        }),
        CompareOperator::NotEq => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs) != Some(Ordering::Equal)
        }),
        CompareOperator::Gt => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs) == Some(Ordering::Greater)
        }),
        CompareOperator::Gte => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs).is_some_and(Ordering::is_ge)
        }),
        CompareOperator::Lt => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs) == Some(Ordering::Less)
        }),
        CompareOperator::Lte => visit_compare_with::<T, V>(visitor, move |lhs, rhs| {
            compare(lhs, rhs).is_some_and(Ordering::is_le)
        }),
    }
}

fn visit_compare<T, V>(op: CompareOperator, visitor: V) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    // Select the predicate before traversal so the row closure does not match on the operator.
    match op {
        CompareOperator::Eq => visit_compare_with::<T, V>(visitor, T::is_eq),
        CompareOperator::NotEq => visit_compare_with::<T, V>(visitor, |lhs, rhs| !lhs.is_eq(rhs)),
        CompareOperator::Gt => visit_compare_with::<T, V>(visitor, T::is_gt),
        CompareOperator::Gte => visit_compare_with::<T, V>(visitor, T::is_ge),
        CompareOperator::Lt => visit_compare_with::<T, V>(visitor, T::is_lt),
        CompareOperator::Lte => visit_compare_with::<T, V>(visitor, T::is_le),
    }
}

fn visit_compare_with<T, V>(
    visitor: V,
    compare: impl Fn(T, T) -> bool,
) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    visitor.visit_bool::<(T, T), true>(move |(lhs, rhs)| compare(lhs, rhs))
}
