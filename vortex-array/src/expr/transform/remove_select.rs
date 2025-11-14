// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_err};

use crate::expr::exprs::get_item::get_item;
use crate::expr::exprs::pack::pack;
use crate::expr::exprs::select::Select;
use crate::expr::transform::traits::{ReduceRule, RewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};
use crate::expr::{Expression, ExpressionView};

/// Replaces [crate::SelectExpr] with combination of [crate::GetItem] and [crate::Pack] expressions.
pub(crate) fn remove_select(e: Expression, ctx: &DType) -> VortexResult<Expression> {
    e.transform_up(|node| remove_select_transformer(node, ctx))
        .map(|e| e.into_inner())
}

fn remove_select_transformer(
    node: Expression,
    ctx: &DType,
) -> VortexResult<Transformed<Expression>> {
    if let Some(select) = node.as_opt::<Select>() {
        let child = select.child();
        let child_dtype = child.return_dtype(ctx)?;
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

        Ok(Transformed::yes(expr))
    } else {
        Ok(Transformed::no(node))
    }
}

/// Rule that removes Select expressions by converting them to Pack + GetItem.
///
/// Transforms: `select(["a", "b"], expr)` → `pack(a: get_item("a", expr), b: get_item("b", expr))`
pub struct RemoveSelectRule;

impl ReduceRule<Select> for RemoveSelectRule {
    fn reduce(
        &self,
        select: &ExpressionView<Select>,
        ctx: &dyn RewriteContext,
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

    use super::{RemoveSelectRule, remove_select};
    use crate::expr::exprs::pack::Pack;
    use crate::expr::exprs::root::root;
    use crate::expr::exprs::select::{Select, select};
    use crate::expr::session::ExprSession;
    use crate::expr::transform::simplify_typed::apply_child_rules;
    use crate::expr::transform::traits::{ReduceRule, SimpleRewriteContext};

    #[test]
    fn test_remove_select() {
        let dtype = DType::Struct(
            StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
            Nullable,
        );
        let e = select(["a", "b"], root());
        let e = remove_select(e, &dtype).unwrap();

        assert!(e.is::<Pack>());
        assert!(e.return_dtype(&dtype).unwrap().is_nullable());
    }

    #[test]
    fn test_remove_select_rule_direct() {
        let dtype = DType::Struct(
            StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
            Nullable,
        );
        let e = select(["a", "b"], root());

        let rule = RemoveSelectRule;
        let ctx = SimpleRewriteContext { dtype: &dtype };
        let select_view = e.as_::<Select>();
        let result = rule.reduce(&select_view, &ctx).unwrap();

        assert!(result.is_some());
        let transformed = result.unwrap();
        assert!(transformed.is::<Pack>());
        assert!(transformed.return_dtype(&dtype).unwrap().is_nullable());
    }

    #[test]
    fn test_remove_select_via_session() {
        let dtype = DType::Struct(
            StructFields::new(
                ["a", "b", "c"].into(),
                vec![I32.into(), I32.into(), I32.into()],
            ),
            Nullable,
        );

        // Create expression: select(["a", "c"], root())
        let e = select(["a", "c"], root());

        // Use session which has RemoveSelectRule registered
        let session = ExprSession::default();
        let result = apply_child_rules(e, &dtype, &session).unwrap();

        // Should be transformed to Pack
        assert!(result.is::<Pack>());

        // Verify the dtype has only selected fields
        let result_dtype = result.return_dtype(&dtype).unwrap();
        let fields = result_dtype.as_struct_fields_opt().unwrap();
        assert_eq!(fields.names().len(), 2);
        assert_eq!(fields.names()[0].as_ref(), "a");
        assert_eq!(fields.names()[1].as_ref(), "c");
    }
}
