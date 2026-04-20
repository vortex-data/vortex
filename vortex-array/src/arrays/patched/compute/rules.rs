// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Patched;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;

static KEYED_PARENT_RULES: [ParentRuleEntry<Patched>; 2] = [
    ParentRuleSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterReduceAdaptor(Patched),
    ),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Patched)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Patched> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<Patched> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
