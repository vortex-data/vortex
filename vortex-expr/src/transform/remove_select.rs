use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{get_item, pack, ExprRef, Select};

/// Replaces [Select] with combination of [GetItem] and [Pack] expressions.
pub(crate) fn remove_select(e: ExprRef, scope_dt: &DType) -> VortexResult<ExprRef> {
    let mut transform = RemoveSelectTransform {
        scope_dtype: scope_dt,
    };
    e.transform(&mut transform).map(|e| e.result)
}

struct RemoveSelectTransform<'a> {
    scope_dtype: &'a DType,
}

impl MutNodeVisitor for RemoveSelectTransform<'_> {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: ExprRef) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(select) = node.as_any().downcast_ref::<Select>() {
            let child = select.child();
            let child_dtype = child.return_dtype(self.scope_dtype)?;
            let child_dtype = child_dtype.as_struct().ok_or_else(|| {
                vortex_err!(
                    "Select child must return a struct dtype, however it was a {}",
                    child_dtype
                )
            })?;

            let expr = pack(
                select
                    .fields()
                    .as_include_names(child_dtype.names())
                    .map_err(|e| {
                        e.with_context(format!(
                            "Select fields {:?} must be a subset of child fields {:?}",
                            select.fields(),
                            child_dtype.names()
                        ))
                    })?
                    .iter()
                    .map(|name| (name.clone(), get_item(name.clone(), child.clone()))),
            );

            Ok(TransformResult::yes(expr))
        } else {
            Ok(TransformResult::no(node))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructDType};

    use crate::transform::remove_select::remove_select;
    use crate::{ident, select, Pack};

    #[test]
    fn test_remove_select() {
        let dtype = DType::Struct(
            Arc::new(StructDType::new(
                ["a".into(), "b".into()].into(),
                vec![I32.into(), I32.into()],
            )),
            NonNullable,
        );
        let e = select(["a".into(), "b".into()], ident());
        let e = remove_select(e, &dtype).unwrap();

        assert!(e.as_any().is::<Pack>());
    }
}
