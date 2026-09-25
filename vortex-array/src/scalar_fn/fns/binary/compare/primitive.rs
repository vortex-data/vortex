// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`].
//!
//! [`PrimitiveCompare`] delegates decoding, constant handling, validity, and packed Boolean output
//! to the row executor. Its row kernel contains only the comparison selected by [`CompareOperator`].

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
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;

/// Compare two primitive arrays of the same [`PType`].
///
/// Floats compare with Vortex's total ordering: `NaN` is the largest value, `-0.0 < +0.0`, and
/// equality is bitwise.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    execute_rows(&PrimitiveCompare, &op, &args, ctx)
}

/// Internal row execution for primitive comparison operators.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        // `PrimitiveCompare` is a private implementation detail of `Binary`: it is never registered
        // or serialized independently. Reusing the public ID keeps execution errors attributed to
        // `Binary`. If this type becomes registrable, it needs its own ID and persistence contract.
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
