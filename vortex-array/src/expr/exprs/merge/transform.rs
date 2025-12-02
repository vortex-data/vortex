// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TODO(ngates): bring this back as an ArrayReduce rule.

use itertools::Itertools as _;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::expr::exprs::merge::Merge;
use crate::expr::transform::rules::ReduceRule;
use crate::expr::transform::rules::TypedRuleContext;
use crate::expr::Expression;

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
    use crate::expr::exprs::merge::merge_opts;
    use crate::expr::exprs::merge::DuplicateHandling;
    use crate::expr::exprs::merge::Merge;
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
