#![allow(unused_imports)]
use std::sync::Arc;

use vortex_dtype::{Field, FieldName};

use crate::{
    col, lit, not, select, BinaryExpr, Column, ExprRef, Identity, Like, Literal, Not, Operator,
    RowFilter, Select, SelectField, VortexExpr, VortexExprExt,
};

/// Restrict expression to only the fields that appear in projection
///
/// TODO(ngates): expressions should have tree-traversal API so this is generic.
/// TODO(joe): remove once layouts are switched over too.
pub fn expr_project(expr: &ExprRef, projection: &[FieldName]) -> Option<ExprRef> {
    if let Some(rf) = expr.as_any().downcast_ref::<RowFilter>() {
        rf.only_fields(projection)
    } else if expr.as_any().downcast_ref::<Literal>().is_some() {
        Some(expr.clone())
    } else if let Some(s) = expr.as_any().downcast_ref::<Select>() {
        match s.fields() {
            SelectField::Include(i) => {
                let fields = i
                    .iter()
                    .filter(|f| projection.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                if projection.len() == 1 {
                    Some(Arc::new(Identity))
                } else {
                    (!fields.is_empty()).then(|| select(fields, s.child().clone()))
                }
            }
            SelectField::Exclude(e) => {
                let fields = projection
                    .iter()
                    .filter(|f| !e.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                if projection.len() == 1 {
                    Some(Arc::new(Identity))
                } else {
                    (!fields.is_empty()).then(|| select(fields, s.child().clone()))
                }
            }
        }
    } else if let Some(c) = expr.as_any().downcast_ref::<Column>() {
        projection.contains(c.field()).then(|| {
            if projection.len() == 1 {
                Arc::new(Identity)
            } else {
                expr.clone()
            }
        })
    } else if let Some(n) = expr.as_any().downcast_ref::<Not>() {
        let own_refs = expr.references();
        if own_refs.iter().all(|p| projection.contains(p)) {
            expr_project(n.child(), projection).map(not)
        } else {
            None
        }
    } else if let Some(bexp) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let lhs_proj = expr_project(bexp.lhs(), projection);
        let rhs_proj = expr_project(bexp.rhs(), projection);
        if bexp.op() == Operator::And {
            match (lhs_proj, rhs_proj) {
                (Some(lhsp), Some(rhsp)) => Some(BinaryExpr::new_expr(lhsp, bexp.op(), rhsp)),
                // Projected lhs and rhs might lose reference to columns if they're simplified to straight column comparisons
                (Some(lhsp), None) => (!bexp
                    .rhs()
                    .references()
                    .intersection(&bexp.lhs().references())
                    .any(|f| projection.contains(f)))
                .then_some(lhsp),
                (None, Some(rhsp)) => (!bexp
                    .lhs()
                    .references()
                    .intersection(&bexp.rhs().references())
                    .any(|f| projection.contains(f)))
                .then_some(rhsp),
                (None, None) => None,
            }
        } else {
            Some(BinaryExpr::new_expr(lhs_proj?, bexp.op(), rhs_proj?))
        }
    } else if let Some(l) = expr.as_any().downcast_ref::<Like>() {
        let child = expr_project(l.child(), projection)?;
        let pattern = expr_project(l.pattern(), projection)?;
        Some(Like::new_expr(
            child,
            pattern,
            l.negated(),
            l.case_insensitive(),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::Field;

    use super::*;
    use crate::{and, ident, lt, or, Identity, Not, Select};

    #[test]
    fn project_and() {
        let band = and(col("a"), col("b"));
        let projection = vec!["b".into()];
        assert_eq!(
            &expr_project(&band, &projection).unwrap(),
            &(Arc::new(Identity) as ExprRef)
        );
    }

    #[test]
    fn project_or() {
        let bor = or(col("a"), col("b"));
        let projection = vec!["b".into()];
        assert!(expr_project(&bor, &projection).is_none());
    }

    #[test]
    fn project_nested() {
        let band = and(lt(col("a"), col("b")), lt(lit(5), col("b")));
        let projection = vec!["b".into()];
        assert!(expr_project(&band, &projection).is_none());
    }

    #[test]
    fn project_multicolumn() {
        let blt = lt(col("a"), col("b"));
        let projection = vec![FieldName::from("a"), FieldName::from("b")];
        assert_eq!(
            &expr_project(&blt, &projection).unwrap(),
            &lt(col("a"), col("b"))
        );
    }

    #[test]
    fn project_select() {
        let include = select(
            vec![
                FieldName::from("a"),
                FieldName::from("b"),
                FieldName::from("c"),
            ],
            ident(),
        );
        let projection = vec![FieldName::from("a"), FieldName::from("b")];
        assert_eq!(
            *expr_project(&include, &projection).unwrap(),
            *select(projection, ident())
        );
    }

    #[test]
    fn project_select_extra_columns() {
        let include = select(
            vec![
                FieldName::from("a"),
                FieldName::from("b"),
                FieldName::from("c"),
            ],
            ident(),
        );
        let projection = vec![FieldName::from("c"), FieldName::from("d")];
        assert_eq!(
            *expr_project(&include, &projection).unwrap(),
            *select(vec![FieldName::from("c")], ident())
        );
    }

    #[test]
    fn project_not() {
        let not_e = not(col("a"));
        let projection = vec![FieldName::from("a"), FieldName::from("b")];
        assert_eq!(&expr_project(&not_e, &projection).unwrap(), &not_e);
    }
}
