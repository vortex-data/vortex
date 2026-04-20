// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::FSST;

static KEYED_PARENT_RULES: [ParentRuleEntry<FSST>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(FSST)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(FSST)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<FSST> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<FSST> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
