// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive comparison execution through [`RowFn`].

#[cfg(target_arch = "x86_64")]
mod columnar;
#[cfg(target_arch = "x86_64")]
mod operand;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::CompareOperator;

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
    #[cfg(target_arch = "x86_64")]
    if use_columnar_comparison(lhs, rhs, op)? {
        return columnar::compare_primitive(lhs, rhs, op, ctx);
    }

    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    ScalarFnVTable::execute(&PrimitiveCompare, &op, &args, ctx)
}

/// Internal row execution for primitive comparison operators.
#[derive(Clone)]
struct PrimitiveCompare;

impl RowFn for PrimitiveCompare {
    type Options = CompareOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        ScalarFnVTable::id(&Binary)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        op: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let ptype =
            PType::try_from(args.first().ok_or_else(|| {
                vortex_err!("a comparison operator takes two operands, got none")
            })?)?;

        match_each_native_ptype!(ptype, |T| { visit_compare::<T, V>(*op, visitor) })
    }
}

#[cfg(target_arch = "x86_64")]
fn use_columnar_comparison(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
) -> VortexResult<bool> {
    if matches!(op, CompareOperator::Eq | CompareOperator::NotEq) {
        return Ok(false);
    }

    let ptype = PType::try_from(lhs.dtype())?;
    Ok(match ptype {
        // The fused comparison and bit-packing loop produces better x86 code for signed 64-bit
        // integers and f64. The RowFn byte-output loop remains faster for narrower lanes.
        PType::I64 | PType::F64 => true,
        // LLVM vectorizes varying u64 inputs, but not the mixed-constant RowFn loop.
        PType::U64 => lhs.as_constant().is_some() || rhs.as_constant().is_some(),
        _ => false,
    })
}

fn visit_compare<T, V>(op: CompareOperator, visitor: V) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor,
{
    match op {
        CompareOperator::Eq => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_eq(rhs)),
        CompareOperator::NotEq => visitor.visit::<(T, T), bool>(|(lhs, rhs)| !lhs.is_eq(rhs)),
        CompareOperator::Gt => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_gt(rhs)),
        CompareOperator::Gte => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_ge(rhs)),
        CompareOperator::Lt => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_lt(rhs)),
        CompareOperator::Lte => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_le(rhs)),
    }
}
