// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::Expression;
use crate::expr::ExpressionView;
use crate::expr::exprs::get_item::get_item;
use crate::expr::exprs::merge::DuplicateHandling;
use crate::expr::exprs::merge::Merge;
use crate::expr::exprs::pack::pack;
use crate::expr::transform::rules::ReduceRule;
use crate::expr::transform::rules::TypedRuleContext;

/// Rule that removes Merge expressions by converting them to Pack + GetItem.
///
/// Transforms: `merge([struct1, struct2])` → `pack(field1: get_item("field1", struct1), field2: get_item("field2", struct2), ...)`
#[derive(Debug, Default)]
pub struct RemoveMergeRule;

impl ReduceRule<Merge, TypedRuleContext> for RemoveMergeRule {
    fn reduce(
        &self,
        merge: &ExpressionView<Merge>,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>> {
        let merge_dtype = merge.return_dtype(ctx.dtype())?;
        let mut names = Vec::with_capacity(merge.children().len() * 2);
        let mut children = Vec::with_capacity(merge.children().len() * 2);
        let mut duplicate_names = HashSet::<_>::new();

        for child in merge.children().iter() {
            let child_dtype = child.return_dtype(ctx.dtype())?;
            if !child_dtype.is_struct() {
                vortex_bail!(
                    "Merge child must return a non-nullable struct dtype, got {}",
                    child_dtype
                )
            }

            let child_dtype = child_dtype
                .as_struct_fields_opt()
                .vortex_expect("expected struct");

            for name in child_dtype.names().iter() {
                if let Some(idx) = names.iter().position(|n| n == name) {
                    duplicate_names.insert(name.clone());
                    children[idx] = child.clone();
                } else {
                    names.push(name.clone());
                    children.push(child.clone());
                }
            }

            if merge.data() == &DuplicateHandling::Error && !duplicate_names.is_empty() {
                vortex_bail!(
                    "merge: duplicate fields in children: {}",
                    duplicate_names.into_iter().format(", ")
                )
            }
        }

        let expr = pack(
            names
                .into_iter()
                .zip(children)
                .map(|(name, child)| (name.clone(), get_item(name, child))),
            merge_dtype.nullability(),
        );

        Ok(Some(expr))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::PType::I64;
    use vortex_dtype::PType::U32;
    use vortex_dtype::PType::U64;

    use super::RemoveMergeRule;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::merge::DuplicateHandling;
    use crate::expr::exprs::merge::Merge;
    use crate::expr::exprs::merge::merge_opts;
    use crate::expr::exprs::pack::Pack;
    use crate::expr::exprs::root::root;
    use crate::expr::transform::rules::ReduceRule;
    use crate::expr::transform::rules::TypedRuleContext;

    #[test]
    fn test_remove_merge() {
        let dtype = DType::struct_(
            [
                ("0", DType::struct_([("a", I32), ("b", I64)], NonNullable)),
                ("1", DType::struct_([("b", U32), ("c", U64)], NonNullable)),
            ],
            NonNullable,
        );

        let e = merge_opts(
            [get_item("0", root()), get_item("1", root())],
            DuplicateHandling::RightMost,
        );

        let ctx = TypedRuleContext::new(dtype.clone());
        let merge_view = e.as_::<Merge>();
        let result = RemoveMergeRule.reduce(&merge_view, &ctx).unwrap();

        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result.is::<Pack>());
        assert_eq!(
            result.return_dtype(&dtype).unwrap(),
            DType::struct_([("a", I32), ("b", U32), ("c", U64)], NonNullable)
        );
    }
}
