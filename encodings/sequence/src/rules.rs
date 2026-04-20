// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::Sequence;

static KEYED_PARENT_RULES: [ParentRuleEntry<Sequence>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Sequence)),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.list_contains"),
        &ListContainsElementReduceAdaptor(Sequence),
    ),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Sequence)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Sequence> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<Sequence> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
