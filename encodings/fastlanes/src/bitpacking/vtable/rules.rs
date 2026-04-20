// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::BitPacked;

static KEYED_RULES: [ParentRuleEntry<BitPacked>; 2] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(BitPacked)),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.slice"),
        &SliceReduceAdaptor(BitPacked),
    ),
];

static KEYED_RULES_DENSE: ParentRuleDense<BitPacked> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<BitPacked> =
    ParentRuleSet::new_indexed(&KEYED_RULES, &KEYED_RULES_DENSE, &[]);
