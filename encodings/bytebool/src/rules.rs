// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::ByteBool;

static KEYED_PARENT_RULES: [ParentRuleEntry<ByteBool>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(ByteBool)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(ByteBool)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(ByteBool)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<ByteBool> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<ByteBool> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
