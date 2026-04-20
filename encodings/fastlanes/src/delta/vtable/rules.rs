// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::delta::vtable::Delta;

static KEYED_RULES: [ParentRuleEntry<Delta>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Delta)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Delta)),
];

static KEYED_RULES_DENSE: ParentRuleDense<Delta> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<Delta> =
    ParentRuleSet::new_indexed(&KEYED_RULES, &KEYED_RULES_DENSE, &[]);
