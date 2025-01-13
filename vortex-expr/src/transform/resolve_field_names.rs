use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItem};

/// Resolves any [`vortex_dtype::Field::Index`] nodes in the expression to
/// [`vortex_dtype::Field::Name`] nodes.
pub fn resolve_field_names(expr: ExprRef, scope_dtype: &DType) -> VortexResult<ExprRef> {
    let mut visitor = FieldToNameTransform { scope_dtype };
    expr.transform(&mut visitor).map(|node| node.result)
}

struct FieldToNameTransform<'a> {
    scope_dtype: &'a DType,
}

impl MutNodeVisitor for FieldToNameTransform<'_> {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if get_item.field().is_named() {
                return Ok(TransformResult::no(node));
            }

            let child_dtype = get_item.child().return_dtype(self.scope_dtype)?;
            let struct_dtype = child_dtype
                .as_struct()
                .ok_or_else(|| vortex_err!("get_item requires child to have struct dtype"))?;

            return Ok(TransformResult::yes(GetItem::new_expr(
                get_item
                    .field()
                    .clone()
                    .into_named_field(struct_dtype.names())?,
                get_item.child().clone(),
            )));
        }

        Ok(TransformResult::no(node))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructDType};

    use super::*;
    use crate::{get_item, ident};

    #[test]
    fn test_idx_to_name_expr() {
        let dtype = DType::Struct(
            StructDType::new(
                vec!["a".into(), "b".into()].into(),
                vec![
                    DType::Struct(
                        StructDType::new(
                            vec!["c".into(), "d".into()].into(),
                            vec![I32.into(), I32.into()],
                        ),
                        NonNullable,
                    ),
                    DType::Struct(
                        StructDType::new(
                            vec!["e".into(), "f".into()].into(),
                            vec![I32.into(), I32.into()],
                        ),
                        NonNullable,
                    ),
                ],
            ),
            NonNullable,
        );
        let expr = get_item(1, get_item("a", ident()));
        let new_expr = resolve_field_names(expr, &dtype).unwrap();
        assert_eq!(&new_expr, &get_item("d", get_item("a", ident())));

        let expr = get_item(0, get_item(1, ident()));
        let new_expr = resolve_field_names(expr, &dtype).unwrap();
        assert_eq!(&new_expr, &get_item("e", get_item("b", ident())));
    }
}
