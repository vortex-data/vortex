// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::traversal::{NodeExt, Transformed};
use crate::{DType, ExprRef, MergeVTable, get_item, pack};

/// Replaces [crate::MergeExpr] with combination of [crate::GetItem] and [crate::Pack] expressions.
pub(crate) fn remove_merge(e: ExprRef, ctx: &DType) -> VortexResult<ExprRef> {
    e.transform_up(|node| merge_transform(node, ctx))
        .map(|t| t.into_inner())
}

fn merge_transform(node: ExprRef, ctx: &DType) -> VortexResult<Transformed<ExprRef>> {
    match node.as_opt::<MergeVTable>() {
        None => Ok(Transformed::no(node)),
        Some(merge) => {
            let merge_dtype = merge.return_dtype(ctx)?;
            let mut names = Vec::with_capacity(merge.children().len() * 2);
            let mut children = Vec::with_capacity(merge.children().len() * 2);
            let mut all_nullable = true;
            for child in merge.children() {
                let child_dtype = child.return_dtype(ctx)?;
                if !child_dtype.is_struct() {
                    return Err(vortex_err!(
                        "Merge child must return a non-nullable struct dtype, got {}",
                        child_dtype
                    ));
                }
                all_nullable = all_nullable && child_dtype.is_nullable();

                let child_dtype = child_dtype
                    .as_struct_fields_opt()
                    .vortex_expect("expected struct");

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
                merge_dtype.nullability(),
            );
            Ok(Transformed::yes(expr))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::{I32, I64, U32, U64};

    use crate::transform::remove_merge::remove_merge;
    use crate::{PackVTable, get_item, merge, root};

    #[test]
    fn test_remove_merge() {
        let dtype = DType::struct_(
            [
                ("0", DType::struct_([("a", I32), ("b", I64)], NonNullable)),
                ("1", DType::struct_([("b", U32), ("c", U64)], NonNullable)),
            ],
            NonNullable,
        );

        let e = merge([get_item("0", root()), get_item("1", root())]);
        let e = remove_merge(e, &dtype).unwrap();

        assert!(e.is::<PackVTable>());
        assert_eq!(
            e.return_dtype(&dtype).unwrap(),
            DType::struct_([("a", I32), ("b", U32), ("c", U64)], NonNullable)
        );
    }
}
