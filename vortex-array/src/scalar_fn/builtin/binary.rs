// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Code that the two-operand functions [`Compare`], [`Boolean`], and [`Numeric`] share.
//!
//! [`Compare`]: crate::scalar_fn::builtin::compare::Compare
//! [`Boolean`]: crate::scalar_fn::builtin::boolean::Boolean
//! [`Numeric`]: crate::scalar_fn::builtin::numeric::Numeric

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;

use crate::AnyColumnar;
use crate::ExecutionCtx;
use crate::matcher::Matcher;
use crate::require_input;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::display::ExprDisplay;

/// Returns the name of an operand of a two-operand function.
pub(super) fn binary_child_name(child_idx: usize) -> ChildName {
    match child_idx {
        0 => ChildName::from("lhs"),
        1 => ChildName::from("rhs"),
        _ => unreachable!("a binary function has only two children"),
    }
}

/// Formats a two-operand function as `(lhs <operator> rhs)`.
pub(super) fn fmt_infix(
    operator: &dyn Display,
    expr: &dyn ExprDisplay,
    f: &mut Formatter<'_>,
) -> std::fmt::Result {
    write!(f, "(")?;
    Display::fmt(expr.display_child(0), f)?;
    write!(f, " {operator} ")?;
    Display::fmt(expr.display_child(1), f)?;
    write!(f, ")")
}

/// Executes a strict two-operand function when its operands are ready.
///
/// The operands are ready when both are columnar. A null constant operand also makes them ready,
/// because it decides the result, such as `x < null`, so `execute` does not need the other
/// operand executed.
pub(super) fn execute_strict_step<V: ScalarFnVTable>(
    vtable: &V,
    options: &V::Options,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ScalarFnExecutionResult> {
    let columnar = |idx| args.input_matches(idx, AnyColumnar::matches);
    let null = |idx| {
        args.input_matches(idx, |array| {
            array.as_constant().is_some_and(|scalar| scalar.is_null())
        })
    };

    let ready = (columnar(0)? && columnar(1)?) || null(0)? || null(1)?;
    if !ready {
        require_input!(args, 0 => AnyColumnar);
        require_input!(args, 1 => AnyColumnar);
    }

    vtable
        .execute(options, args, ctx)
        .map(ScalarFnExecutionResult::done)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::NumericOperator;
    use crate::scalar_fn::builtin::operators::BooleanOperator;
    use crate::scalar_fn::builtin::operators::CompareOperator;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Comparisons over arithmetic, joined by `AND`: every input is a scalar function, so each
    /// operator executes its inputs through the executor.
    #[test]
    fn executes_non_columnar_inputs_iteratively() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let a = PrimitiveArray::from_iter([1i32, 5, 9, 12]).into_array();
        let b = PrimitiveArray::from_iter([2i32, 2, 2, 2]).into_array();
        let sum = a.numeric(b.clone(), NumericOperator::Add)?;
        let difference = a.numeric(b, NumericOperator::Sub)?;
        let ten = ConstantArray::new(10i32, 4).into_array();
        let result = sum.compare(ten.clone(), CompareOperator::Gt)?.boolean(
            difference.compare(ten, CompareOperator::Lt)?,
            BooleanOperator::And,
        )?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, false, true, false]),
            &mut ctx
        );
        Ok(())
    }
}
