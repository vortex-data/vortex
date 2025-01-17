use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{get_item, pack, ExprRef, Select};

/// Select is a useful expression, however it can be defined in terms of get_item & pack,
/// once the expression type is known, this simplifications pass removes the select expression.
pub fn remove_select(e: ExprRef, scope_dt: DType) -> VortexResult<ExprRef> {
    let mut transform = RemoveSelectTransform::new(scope_dt);
    e.transform(&mut transform).map(|e| e.result)
}

struct RemoveSelectTransform {
    ident_dtype: DType,
}

impl RemoveSelectTransform {
    fn new(ident_dtype: DType) -> Self {
        Self { ident_dtype }
    }
}

impl MutNodeVisitor for RemoveSelectTransform {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: ExprRef) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(select) = node.as_any().downcast_ref::<Select>() {
            let child = select.child();
            let child_dtype = child.return_dtype(&self.ident_dtype)?;
            let child_dtype = child_dtype.as_struct().ok_or_else(|| {
                vortex_err!(
                    "Select child must return a struct dtype, however it was a {}",
                    child_dtype
                )
            })?;

            let names = select
                .fields()
                .as_include_names(child_dtype.names())
                .map_err(|e| {
                    vortex_err!(
                        "Select fields must be a subset of child fields, however {}",
                        e
                    )
                })?;

            let pack_children = names
                .iter()
                .map(|name| get_item(name.clone(), child.clone()))
                .collect_vec();

            Ok(TransformResult::yes(pack(names, pack_children)))
        } else {
            Ok(TransformResult::no(node))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructDType};

    use crate::transform::remove_select::remove_select;
    use crate::{ident, select, Pack};

    #[test]
    fn test_remove_select() {
        let dtype = DType::Struct(
            StructDType::new(
                ["a".into(), "b".into()].into(),
                vec![I32.into(), I32.into()],
            ),
            NonNullable,
        );
        let e = select(["a".into(), "b".into()], ident());
        let e = remove_select(e, dtype).unwrap();

        assert!(e.as_any().is::<Pack>());
    }
}
