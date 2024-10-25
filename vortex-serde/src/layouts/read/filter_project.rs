use std::sync::Arc;

use vortex_dtype::field::Field;
use vortex_expr::{BinaryExpr, Column, Identity, Literal, Operator, Select, VortexExpr};

use crate::layouts::RowFilter;

pub fn filter_project(
    filter: &Arc<dyn VortexExpr>,
    projection: &[Field],
) -> Option<Arc<dyn VortexExpr>> {
    if let Some(rf) = filter.as_any().downcast_ref::<RowFilter>() {
        rf.only_fields(projection).map(|rf| Arc::new(rf) as _)
    } else if filter.as_any().downcast_ref::<Literal>().is_some() {
        Some(filter.clone())
    } else if let Some(s) = filter.as_any().downcast_ref::<Select>() {
        match s {
            Select::Include(i) => {
                let fields = i
                    .iter()
                    .filter(|f| projection.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                match fields.len() {
                    0 => None,
                    1 => Some(Arc::new(Identity)),
                    _ => Some(Arc::new(Select::include(fields))),
                }
            }
            Select::Exclude(e) => {
                let fields = projection
                    .iter()
                    .filter(|f| !e.contains(f))
                    .cloned()
                    .collect::<Vec<_>>();
                match fields.len() {
                    0 => None,
                    1 => Some(Arc::new(Identity)),
                    _ => Some(Arc::new(Select::include(fields))),
                }
            }
        }
    } else if let Some(c) = filter.as_any().downcast_ref::<Column>() {
        projection.contains(c.field()).then(|| {
            if projection.len() == 1 {
                Arc::new(Identity)
            } else {
                Arc::new(Column::new(c.field().clone())) as Arc<dyn VortexExpr>
            }
        })
    } else if let Some(bexp) = filter.as_any().downcast_ref::<BinaryExpr>() {
        let lhs_proj = filter_project(bexp.lhs(), projection);
        let rhs_proj = filter_project(bexp.rhs(), projection);
        if bexp.op() == Operator::And {
            match (lhs_proj, rhs_proj) {
                (Some(lhsp), Some(rhsp)) => Some(Arc::new(BinaryExpr::new(lhsp, bexp.op(), rhsp))),
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
            Some(Arc::new(BinaryExpr::new(lhs_proj?, bexp.op(), rhs_proj?)))
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::field::Field;
    use vortex_expr::{BinaryExpr, Column, Identity, Literal, Operator, Select, VortexExpr};

    use crate::layouts::read::filter_project::filter_project;

    #[test]
    fn project_and() {
        let band = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::And,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("b")];
        assert_eq!(
            *filter_project(&band, &projection).unwrap(),
            *Identity.as_any()
        );
    }

    #[test]
    fn project_or() {
        let bor = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::Or,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("b")];
        assert!(filter_project(&bor, &projection).is_none());
    }

    #[test]
    fn project_nested() {
        let band = Arc::new(BinaryExpr::new(
            Arc::new(BinaryExpr::new(
                Arc::new(Column::new(Field::from("a"))),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(Literal::new(5.into())),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )),
        )) as _;
        let projection = vec![Field::from("b")];
        let option = filter_project(&band, &projection);
        println!("expr: {option:?}");
        assert!(option.is_none());
    }

    #[test]
    fn project_multicolumn() {
        let blt = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(Field::from("a"))),
            Operator::Lt,
            Arc::new(Column::new(Field::from("b"))),
        )) as _;
        let projection = vec![Field::from("a"), Field::from("b")];
        assert_eq!(
            *filter_project(&blt, &projection).unwrap(),
            *BinaryExpr::new(
                Arc::new(Column::new(Field::from("a"))),
                Operator::Lt,
                Arc::new(Column::new(Field::from("b"))),
            )
            .as_any()
        );
    }

    #[test]
    fn project_select() {
        let blt = Arc::new(Select::include(vec![
            Field::from("a"),
            Field::from("b"),
            Field::from("c"),
        ])) as _;
        let projection = vec![Field::from("a"), Field::from("b")];
        assert_eq!(
            *filter_project(&blt, &projection).unwrap(),
            *Select::include(projection).as_any()
        );
    }
}
