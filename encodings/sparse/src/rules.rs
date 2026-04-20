// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::not::NotReduce;
use vortex_array::scalar_fn::fns::not::NotReduceAdaptor;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::Sparse;

static KEYED_PARENT_RULES: [ParentRuleEntry<Sparse>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Sparse)),
    ParentRuleSet::lift_id(CachedId::new("vortex.not"), &NotReduceAdaptor(Sparse)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Sparse> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<Sparse> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);

impl NotReduce for Sparse {
    fn invert(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array.patches().clone().map_values(|values| values.not())?;
        Ok(Some(
            Sparse::try_new_from_patches(inverted_patches, inverted_fill)?.into_array(),
        ))
    }
}
