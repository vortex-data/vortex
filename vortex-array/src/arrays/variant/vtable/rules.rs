// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::variant_get::VariantGet;

pub(crate) const PARENT_RULES: ParentRuleSet<Variant> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&VariantGetPushDownRule)]);

/// Rule to push VariantGet through VariantArray to its child encoding.
#[derive(Debug)]
struct VariantGetPushDownRule;

impl ArrayParentReduceRule<Variant> for VariantGetPushDownRule {
    type Parent = ExactScalarFn<VariantGet>;

    fn reduce_parent(
        &self,
        array: &VariantArray,
        parent: ScalarFnArrayView<'_, VariantGet>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let options = parent.options;
        Ok(Some(
            array
                .child()
                .variant_get(&options.path, options.dtype.clone())?,
        ))
    }
}
