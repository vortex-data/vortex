// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`].
//!
//! [`Binary`] keeps its existing scalar-function hooks while delegating primitive row execution to
//! a private [`RowFn`]. The executor owns decoding, constants, validity, and Boolean packing.

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
/// The caller must supply equal-length operands with matching primitive dtypes, ignoring
/// nullability. Predicates use [`NativePType`] comparison semantics, including total ordering and
/// bitwise equality for floats.
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

/// A private row kernel used by [`Binary`] for primitive comparisons.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        // Errors must name the caller's `Binary` function. This kernel is never registered or
        // serialized independently, so it can share that ID. Registering it would require a new ID.
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
