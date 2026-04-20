// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::Pco;

static KEYED_PARENT_RULES: [ParentRuleEntry<Pco>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Pco)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Pco)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Pco> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<Pco> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
