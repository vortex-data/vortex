use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, Merge, VortexExpr, get_item, pack};

/// Replaces [Merge] with combination of [GetItem] and [Pack] expressions.
pub(crate) fn remove_merge(e: ExprRef, scope_dt: &DType) -> VortexResult<ExprRef> {
    let mut transform = RemoveMergeTransform {
        scope_dtype: scope_dt,
    };
    e.transform(&mut transform).map(|e| e.result)
}

struct RemoveMergeTransform<'a> {
    scope_dtype: &'a DType,
}

impl MutNodeVisitor for RemoveMergeTransform<'_> {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: ExprRef) -> VortexResult<TransformResult<Self::NodeTy>> {
        if let Some(merge) = node.as_any().downcast_ref::<Merge>() {
            // Try to guess the capacity.
            let mut names = Vec::with_capacity(merge.children().len() * 2);
            let mut children = Vec::with_capacity(merge.children().len() * 2);

            for child in merge.children() {
                let child_dtype = child.return_dtype(self.scope_dtype)?;
                if child_dtype.is_nullable() {
                    todo!("merge nullable structs");
                }
                if !child_dtype.is_struct() {
                    return Err(vortex_err!(
                        "Merge child must return a non-nullable struct dtype, got {}",
                        child_dtype
                    ));
                }
                let child_dtype = child_dtype.as_struct().vortex_expect("expected struct");

                for name in child_dtype.names().iter() {
                    if let Some(idx) = names.iter().position(|n| n == name) {
                        children[idx] = child.clone();
                    } else {
                        names.push(name.clone());
                        children.push(child.clone());
                    }
                }
            }

            let expr = pack(
                names
                    .into_iter()
                    .zip(children)
                    .map(|(name, child)| (name.clone(), get_item(name, child))),
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
    use vortex_dtype::PType::{I32, I64, U32, U64};
    use vortex_dtype::{DType, StructDType};

    use crate::transform::remove_merge::remove_merge;
    use crate::{Pack, get_item, ident, merge};

    #[test]
    fn test_remove_merge() {
        let dtype = DType::Struct(
            Arc::new(StructDType::new(
                ["0".into(), "1".into()].into(),
                vec![
                    DType::Struct(
                        Arc::new(StructDType::new(
                            ["a".into(), "b".into()].into(),
                            vec![I32.into(), I64.into()],
                        )),
                        NonNullable,
                    ),
                    DType::Struct(
                        Arc::new(StructDType::new(
                            ["b".into(), "c".into()].into(),
                            vec![U32.into(), U64.into()],
                        )),
                        NonNullable,
                    ),
                ],
            )),
            NonNullable,
        );

        let e = merge([get_item("0", ident()), get_item("1", ident())]);
        let e = remove_merge(e, &dtype).unwrap();

        assert!(e.as_any().is::<Pack>());
        assert_eq!(
            e.return_dtype(&dtype).unwrap(),
            DType::Struct(
                Arc::new(StructDType::new(
                    ["a".into(), "b".into(), "c".into()].into(),
                    vec![I32.into(), U32.into(), U64.into()],
                )),
                NonNullable,
            )
        );
    }
}
