// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::alp_rd::ALPRD;

static KEYED_PARENT_RULES: [ParentRuleEntry<ALPRD>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(ALPRD)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(ALPRD)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<ALPRD> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<ALPRD> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
