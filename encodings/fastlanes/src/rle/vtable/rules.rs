// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_session::registry::CachedId;

use crate::RLE;

static KEYED_RULES: [ParentRuleEntry<RLE>; 1] = [ParentRuleSet::lift_id(
    CachedId::new("vortex.cast"),
    &CastReduceAdaptor(RLE),
)];

static KEYED_RULES_DENSE: ParentRuleDense<RLE> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<RLE> =
    ParentRuleSet::new_indexed(&KEYED_RULES, &KEYED_RULES_DENSE, &[]);
