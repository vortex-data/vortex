// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::registry::CachedId;

use crate::arrays::Slice;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;

static KEYED_PARENT_RULES: [ParentRuleEntry<Slice>; 1] = [ParentRuleSet::lift_id(
    CachedId::new("vortex.slice"),
    &SliceReduceAdaptor(Slice),
)];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Slice> = ParentRuleDense::new();

pub(super) static PARENT_RULES: ParentRuleSet<Slice> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);
