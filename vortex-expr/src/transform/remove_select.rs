// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexResult, vortex_err};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{DType, ExprRef, SelectVTable, get_item, pack};

/// Replaces [crate::SelectExpr] with combination of [crate::GetItem] and [crate::Pack] expressions.
pub(crate) fn remove_select(e: ExprRef, ctx: &DType) -> VortexResult<ExprRef> {
    let mut transform = RemoveSelectTransform { ctx };
    e.transform(&mut transform).map(|e| e.into_inner())
}

struct RemoveSelectTransform<'a> {
    ctx: &'a DType,
}

impl MutNodeVisitor for RemoveSelectTransform<'_> {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: ExprRef) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(select) = node.as_opt::<SelectVTable>() {
            let child = select.child();
            let child_dtype = child.return_dtype(self.ctx)?;
            let child_nullability = child_dtype.nullability();

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
                child_nullability,
            );

            Ok(TransformResult::yes(expr))
        } else {
            Ok(TransformResult::no(node))
        }
    }
}

#[cfg(test)]
mod tests {

    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructFields};

    use crate::transform::remove_select::remove_select;
    use crate::{PackVTable, root, select};

    #[test]
    fn test_remove_select() {
        let dtype = DType::Struct(
            StructFields::new(["a", "b"].into(), vec![I32.into(), I32.into()]),
            Nullable,
        );
        let e = select(["a", "b"], root());
        let e = remove_select(e, &dtype).unwrap();

        assert!(e.is::<PackVTable>());
        assert!(e.return_dtype(&dtype).unwrap().is_nullable());
    }
}
