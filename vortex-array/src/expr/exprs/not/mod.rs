// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;

pub use kernel::*;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::Arity;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::scalar::Scalar;

/// Expression that logically inverts boolean values.
pub struct Not;

impl VTable for Not {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.not")
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Not expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "not(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        if !matches!(child_dtype, DType::Bool(_)) {
            vortex_bail!(
                "Not expression expects a boolean child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype.clone())
    }

    fn execute(&self, _data: &Self::Options, mut args: ExecutionArgs) -> VortexResult<ArrayRef> {
        let child = args.inputs.pop().vortex_expect("Missing input child");

        // For constant boolean
        if let Some(scalar) = child.as_constant() {
            let value = match scalar.as_bool().value() {
                Some(b) => Scalar::bool(!b, child.dtype().nullability()),
                None => Scalar::null(child.dtype().clone()),
            };
            return Ok(ConstantArray::new(value, args.row_count).into_array());
        }

        // For boolean array
        if let Some(bool) = child.as_opt::<BoolVTable>() {
            return Ok(BoolArray::new(!bool.to_bit_buffer(), bool.validity()?).into_array());
        }

        // Otherwise, execute and try again
        child.execute::<ArrayRef>(args.ctx)?.not()
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Creates an expression that logically inverts boolean values.
///
/// Returns the logical negation of the input boolean expression.
///
/// ```rust
/// # use vortex_array::expr::{not, root};
/// let expr = not(root());
/// ```
pub fn not(operand: Expression) -> Expression {
    Not.new_expr(EmptyOptions, vec![operand])
}

#[cfg(test)]
mod tests {
    use super::not;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::root::root;
    use crate::expr::test_harness;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            bools
                .to_array()
                .apply(&not_expr)
                .unwrap()
                .to_bool()
                .to_bit_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn test_display_order_of_operations() {
        let a = not(get_item("a", root()));
        let b = get_item("a", not(root()));
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(a.to_string(), "not($.a)");
        assert_eq!(b.to_string(), "not($).a");
    }

    #[test]
    fn dtype() {
        let not_expr = not(root());
        let dtype = DType::Bool(Nullability::NonNullable);
        assert_eq!(
            not_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}
