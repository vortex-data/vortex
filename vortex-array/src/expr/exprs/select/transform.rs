// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_err};

use crate::expr::exprs::get_item::get_item;
use crate::expr::exprs::pack::pack;
use crate::expr::exprs::select::Select;
use crate::expr::transform::rules::{ReduceRule, TypedRuleContext};
use crate::expr::{Expression, ExpressionView};

/// Rule that removes Select expressions by converting them to Pack + GetItem.
///
/// Transforms: `select(["a", "b"], expr)` → `pack(a: get_item("a", expr), b: get_item("b", expr))`
pub struct RemoveSelectRule;

impl ReduceRule<Select, TypedRuleContext> for RemoveSelectRule {
    fn reduce(
        &self,
        select: &ExpressionView<Select>,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>> {
        let child = select.child();
        let child_dtype = child.return_dtype(ctx.dtype())?;
        let child_nullability = child_dtype.nullability();

        let child_dtype = child_dtype.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!(
                "Select child must return a struct dtype, however it was a {}",
                child_dtype
            )
        })?;

        let expr = pack(
            select
                .data()
                .as_include_names(child_dtype.names())
                .map_err(|e| {
                    e.with_context(format!(
                        "Select fields {:?} must be a subset of child fields {:?}",
                        select.data(),
                        child_dtype.names()
                    ))
                })?
                .iter()
                .map(|name| (name.clone(), get_item(name.clone(), child.clone()))),
            child_nullability,
        );

        Ok(Some(expr))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructFields};

    use super::RemoveSelectRule;
    use crate::expr::exprs::pack::Pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::{Select, select};
    use crate::expr::transform::rules::{ReduceRule, TypedRuleContext};

    #[test]
    fn test_remove_select_rule() {
        let dtype = DType::Struct(
            StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
            Nullable,
        );
        let e = select(["a", "b"], root());

        let rule = RemoveSelectRule;
        let ctx = TypedRuleContext::new(dtype.clone());
        let select_view = e.as_::<Select>();
        let result = rule.reduce(&select_view, &ctx).unwrap();

        assert!(result.is_some());
        let transformed = result.unwrap();
        assert!(transformed.is::<Pack>());
        assert!(transformed.return_dtype(&dtype).unwrap().is_nullable());
    }

    #[test]
    fn test_remove_select_rule_exclude_fields() {
        use crate::expr::exprs::select::select_exclude;

        let dtype = DType::Struct(
            StructFields::new(
                ["a", "b", "c"].into(),
                vec![I32.into(), I32.into(), I32.into()],
            ),
            Nullable,
        );
        let e = select_exclude(["c"], root());

        let rule = RemoveSelectRule;
        let ctx = TypedRuleContext::new(dtype.clone());
        let select_view = e.as_::<Select>();
        let result = rule.reduce(&select_view, &ctx).unwrap();

        assert!(result.is_some());
        let transformed = result.unwrap();
        assert!(transformed.is::<Pack>());

        // Should exclude "c" and include "a" and "b"
        let result_dtype = transformed.return_dtype(&dtype).unwrap();
        assert!(result_dtype.is_nullable());
        let fields = result_dtype.as_struct_fields_opt().unwrap();
        assert_eq!(fields.names().as_ref(), &["a", "b"]);
    }
}
