use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItem};

pub struct FieldToNameTransform {
    ident_dt: DType,
}

impl FieldToNameTransform {
    fn new(ident_dt: DType) -> Self {
        Self { ident_dt }
    }

    pub fn transform(expr: ExprRef, ident_dt: DType) -> VortexResult<ExprRef> {
        let mut visitor = FieldToNameTransform::new(ident_dt);
        expr.transform(&mut visitor).map(|node| node.result)
    }
}

impl MutNodeVisitor for FieldToNameTransform {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if get_item.field().is_named() {
                return Ok(TransformResult::no(node));
            }

            // TODO(joe) expr::dtype
            let child_dtype = get_item
                .child()
                .evaluate(&Canonical::empty(&self.ident_dt)?.into_array())?
                .dtype()
                .clone();

            let DType::Struct(s_dtype, _) = child_dtype else {
                vortex_bail!(
                    "get_item requires child to have struct dtype, however it was {}",
                    child_dtype
                );
            };

            return Ok(TransformResult::yes(GetItem::new_expr(
                get_item.field().clone().into_named_field(s_dtype.names())?,
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

    use crate::transform::field_type::FieldToNameTransform;
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
        let new_expr = FieldToNameTransform::transform(expr, dtype.clone()).unwrap();
        assert_eq!(&new_expr, &get_item("d", get_item("a", ident())));

        let expr = get_item(0, get_item(1, ident()));
        let new_expr = FieldToNameTransform::transform(expr, dtype).unwrap();
        assert_eq!(&new_expr, &get_item("e", get_item("b", ident())));
    }
}
