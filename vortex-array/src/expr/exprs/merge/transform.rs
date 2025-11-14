// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools as _;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::exprs::get_item::get_item;
use crate::expr::exprs::merge::{DuplicateHandling, Merge};
use crate::expr::exprs::pack::pack;
use crate::expr::transform::rules::{ReduceRule, RewriteContext};
use crate::expr::{Expression, ExpressionView};

/// Rule that removes Merge expressions by converting them to Pack + GetItem.
///
/// Transforms: `merge([struct1, struct2])` → `pack(field1: get_item("field1", struct1), field2: get_item("field2", struct2), ...)`
pub struct RemoveMergeRule;

impl ReduceRule<Merge> for RemoveMergeRule {
    fn reduce(
        &self,
        merge: &ExpressionView<Merge>,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let merge_dtype = merge.return_dtype(ctx.dtype())?;
        let mut names = Vec::with_capacity(merge.children().len() * 2);
        let mut children = Vec::with_capacity(merge.children().len() * 2);
        let mut duplicate_names = HashSet::<_>::new();

        for child in merge.children().iter() {
            let child_dtype = child.return_dtype(ctx.dtype())?;
            if !child_dtype.is_struct() {
                return Err(vortex_err!(
                    "Merge child must return a non-nullable struct dtype, got {}",
                    child_dtype
                ));
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
