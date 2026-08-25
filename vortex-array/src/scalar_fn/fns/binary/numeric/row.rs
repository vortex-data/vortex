// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive arithmetic execution through [`RowFn`].
//!
//! `Binary` keeps its registered contract; [`NumericBinary`] is only an execution helper. Decimal
//! arithmetic remains on its existing columnar path.

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::primitive::CheckedAdd;
use super::primitive::CheckedArithmetic;
use super::primitive::CheckedDiv;
use super::primitive::CheckedMul;
use super::primitive::CheckedPrimitiveOp;
use super::primitive::CheckedSub;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::unstable::row::InitializedElement;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::UninitElementSink;
use crate::scalar_fn::unstable::row::execute_rows;

pub(super) fn execute_numeric_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![lhs.clone(), rhs.clone()], lhs.len());

    execute_rows(&NumericBinary, &op, &args, ctx)
}

/// Internal row execution for the primitive arithmetic operators.
#[derive(Clone)]
struct NumericBinary;

impl RowFn for NumericBinary {
    type Options = NumericOperator;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    // Fallibility is queried without input dtypes, so this conservatively covers integer widths.
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        // `NumericBinary` is a private implementation detail of `Binary`: it is never registered or
        // serialized independently. Reusing the public ID keeps execution errors attributed to
        // `Binary`. If this type becomes registrable, it needs its own ID and persistence contract.
        ScalarFnVTable::id(&Binary)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        op: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let ptype = PType::try_from(
            args.first()
                .ok_or_else(|| vortex_err!("a numeric operator takes two operands, got none"))?,
        )?;

        match_each_native_ptype!(ptype, |T| {
            match op {
                NumericOperator::Add => visit_checked::<T, CheckedAdd, V>(visitor),
                NumericOperator::Sub => visit_checked::<T, CheckedSub, V>(visitor),
                NumericOperator::Mul => visit_checked::<T, CheckedMul, V>(visitor),
                NumericOperator::Div => visit_div::<T, V>(visitor),
            }
        })
    }
}

fn visit_checked<T, Op, V>(visitor: V) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    Op: CheckedPrimitiveOp<T>,
    V: RowVisitor,
{
    visitor.visit_deferred::<(T, T), T, Op::Fail>(
        |(lhs, rhs)| Op::apply(lhs, rhs),
        |failure| {
            if failure != <Op::Fail as Default>::default() {
                return Err(numeric_error(Op::ERROR));
            }

            Ok(())
        },
    )
}

fn visit_div<T, V>(visitor: V) -> VortexResult<V::VisitResult>
where
    T: CheckedArithmetic,
    V: RowVisitor,
{
    if T::PTYPE.is_float() {
        return visit_checked::<T, CheckedDiv, V>(visitor);
    }

    // Integer division is scalar and expensive, so deferring its cheap failure check preserves no
    // vectorization. Check each divide immediately and stop at the first failure.
    // Dense execution leaves output uninitialized. Nullable branches fill placeholders only when
    // they need to skip invalid rows.
    visitor.visit_into::<(T, T), UninitElementSink<T>, _>(|(lhs, rhs), output| {
        let (value, failed) = CheckedDiv::apply(lhs, rhs);
        if failed {
            return Err(numeric_error(<CheckedDiv as CheckedPrimitiveOp<T>>::ERROR));
        }

        // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
        Ok(unsafe { InitializedElement::write(output, value) })
    })
}

/// Keep rich error construction out of row closures so the closures remain inlineable.
#[cold]
#[inline(never)]
fn numeric_error(message: &'static str) -> VortexError {
    vortex_err!(InvalidArgument: "{message}")
}
