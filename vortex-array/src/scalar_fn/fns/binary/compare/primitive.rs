// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`].
//!
//! The row executor handles decoding, constants, validity, and Boolean packing, so this module only
//! selects the predicate that [`CompareOperator`] names.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowInputs;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_row_inputs;

/// Compare two primitive arrays of the same [`PType`].
///
/// Floats compare with Vortex's total ordering over their bit patterns: a `NaN` with the sign bit
/// set sorts below `-inf`, any other `NaN` sorts above `+inf`, and `-0.0 < +0.0`. Equality is
/// bitwise, so distinct `NaN` payloads are not equal.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;

    execute_row_inputs(
        &PrimitiveCompare,
        &op,
        &CompareArgs::new(lhs, rhs, ptype),
        ctx,
    )
}

/// Borrowed comparison operands with their dtypes, so a call does not allocate.
struct CompareArgs<'a> {
    arrays: [&'a ArrayRef; 2],
    dtypes: [DType; 2],
}

impl<'a> CompareArgs<'a> {
    /// Pair two primitive operands of the same [`PType`] with their dtypes.
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

/// The [`RowFn`] that compares two primitive operands with the [`CompareOperator`] in its options.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        // Reusing the `Binary` ID makes errors name the function that the caller used. Nothing
        // registers or serializes `PrimitiveCompare`, which makes the reuse safe. A registered
        // version needs its own ID.
        ScalarFnVTable::id(&Binary)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        op: &Self::Options,
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

        match_each_native_ptype!(ptype, |T| { visit_compare::<T, V>(*op, visitor) })
    }
}

fn visit_compare<T, V>(op: CompareOperator, visitor: V) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
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
